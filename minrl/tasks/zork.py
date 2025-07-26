from pathlib import Path
from loguru import logger
from minrl.constants import Conversation, HostType, Sample
from minrl.tasks.dataset import MinRLDataset, Split
from transformers.tokenization_utils_base import PreTrainedTokenizerBase
import textworld
import textworld.gym
from typing import Any, TypedDict
from textworld.gym.envs import TextworldGymEnv
import re
import os
import random

SYSTEM_PROMPT = """
You are an AI agent whose sole task is to play the text-adventure game Zork.  Your primary goal is to explore, solve puzzles, and collect treasures without dying.

INSTRUCTIONS:
1. After each game response, parse the text to update:
   • Your current location  
   • Inventory items  
   • Visible exits and objects  
   • Any puzzles or obstacles described  

2. Decide on exactly one text command per turn (e.g. "north", "take lantern", "open trapdoor").

3. Always choose the highest-value action, balancing exploration, safety (avoid known hazards), and puzzle-solving.

4. Output the next command, formatted as:
   
   <command>your next command</command>

"""


class ZorkSample(TypedDict):
    index: int


class TextWorldAgent:
    """
    Wrapper for an agent that plays an adventure game.
    Stores any state required for the agent to play the game.
    """

    def __init__(self):
        self.inventory = None
        self.observation_history = []
        self.description = None
        self.action_history = []
        self.done = False
        self.score = 0

    def format_conversation(self) -> Conversation:
        """Format conversation used for inference, given current state"""
        user_message = ""
        if self.inventory and len(self.inventory) > 0:
            inventory_formatted = self.inventory.strip("\n") if self.inventory else ""
            user_message += f"\n### Inventory\n{inventory_formatted}"
        if len(self.action_history) > 0:
            action_history_formatted = []
            for action, description in zip(
                self.action_history, self.observation_history
            ):
                action_history_formatted.append(
                    f"Observation: {description}\nCommand: {action}"
                )
            action_history_formatted = "\n".join(action_history_formatted)
            user_message += f"\n### History\n{action_history_formatted}\n"
        if len(self.observation_history) > 0 and self.description:
            last_obs = self.observation_history[-1]
            user_message += f"\n###Description\n{self.description}\n### Current Observation\n{last_obs}\n"
        conv: Conversation = [
            {"role": "system", "content": SYSTEM_PROMPT},
        ]
        if user_message:
            conv.append({"role": "user", "content": user_message})
        return conv

    def update(
        self,
        command: str | None,
        observation: str,
        inventory: str,
        done: bool,
        score: int,
    ):
        """
        Update agent state after a command and the resulting observation.
        """
        if self.done:
            raise ValueError("Agent is done")
        observation = observation.strip("\n")
        if command is not None:
            self.action_history.append(command)
        self.observation_history.append(observation)
        self.inventory = inventory
        self.done = done
        self.score = score


def parse_command(input_string: str) -> str:
    input_string = input_string.strip("\n").replace("COMMAND: ", "")
    command_contents = re.findall(r"<command>(.*?)</command>", input_string, re.DOTALL)
    return command_contents[0]


def zork_reward_func(conversation: Conversation, sample: dict[str, Any]) -> float:
    return 0.0


class ZorkDataset(MinRLDataset):
    """
    Dataset where the agent plays multiple steps of a text adventure game,
    and the reward is the sum of the rewards for each step.
    """

    max_steps: int = 10

    def __init__(
        self,
        split: Split,
        host: HostType,
        tokenizer: PreTrainedTokenizerBase | None = None,
    ):
        super().__init__(split, host, tokenizer)
        self.tokenizer = tokenizer

        games_directory = Path(os.getenv("INFORM_GAMES_DIRECTORY", ""))
        self.game_names = os.listdir(games_directory)

        self.infos = textworld.EnvInfos(
            feedback=True,
            description=True,
            inventory=True,
            verbs=True,
            intermediate_reward=True,
            entities=True,
            facts=True,
        )

        # Register all available games
        self.env_ids: dict[str, str] = {}
        for game_name in self.game_names:
            game_path = games_directory / game_name
            if game_path.exists():
                self.env_ids[game_name] = textworld.gym.register_game(
                    game_path.as_posix(), self.infos
                )
            else:
                logger.warning(f"Game file not found: {game_path}")

        if not self.env_ids:
            raise ValueError(f"No valid game files found for games: {self.game_names}")

        self.envs: dict[int, TextworldGymEnv] = {}
        self.agents: dict[int, TextWorldAgent] = {}
        self.sample_games: dict[int, str] = {}  # Track which game each sample uses

        self.completed_episodes = []

    def conversation(self, _: Sample, sample_index: int) -> Conversation:
        """
        Return the inference conversation for a single turn.
        For new trajectories, randomly selects a game from available games.
        """
        if sample_index not in self.envs:
            # Randomly select a game for this trajectory
            selected_game = random.choice(list(self.env_ids.keys()))
            env_id = self.env_ids[selected_game]

            env: TextworldGymEnv = textworld.gym.make(env_id)
            agent = TextWorldAgent()
            self.envs[sample_index] = env
            self.sample_games[sample_index] = selected_game
            obs, info = env.reset()
            self.agents[sample_index] = agent
            logger.info(
                f"Started new trajectory {sample_index} with game: {selected_game}"
            )
        agent = self.agents[sample_index]
        conv = agent.format_conversation()
        return conv

    def __getitem__(self, index: int) -> ZorkSample:
        """
        Get a sample from the dataset.
        """
        return {"index": index}

    def get_next_state(
        self, sample_index: int, model_response: str
    ) -> tuple[str, bool]:
        """
        After rollout, update any state needed for the next rollout.
        Returns whether the episode is done.
        """
        agent = self.agents[sample_index]
        env: TextworldGymEnv = self.envs[sample_index]
        try:
            action = parse_command(model_response)
        except Exception as e:
            logger.error(f"Error parsing command from {model_response}: {e}")
            return "", True
        obs, score, done, infos = env.step(action)  # type: ignore
        obs = obs.strip("\n")
        logger.info(f"Action: {action}\n Observation: {obs}")
        inventory = infos["inventory"]
        if not done:
            agent.update(action, obs, inventory, done, score)  # type: ignore
        else:
            # Clean up completed episode
            game_name = self.sample_games[sample_index]
            logger.info(
                f"Completed trajectory {sample_index} with game: {game_name}, final score: {score}"
            )
            del self.envs[sample_index]
            del self.agents[sample_index]
            del self.sample_games[sample_index]
        return obs, done

    def __len__(self) -> int:
        return 1000

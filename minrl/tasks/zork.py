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
        self.sample_conversations: dict[int, Conversation] = {}
        self.sample_games: dict[int, str] = {}  # Track which game each sample uses
        self.sample_done: dict[int, bool] = {}
        self.sample_scores: dict[int, int] = {}

        self.completed_episodes = []

    def format_conversation(self, conversation: Conversation) -> Conversation:
        """Format conversation used for inference"""
        return conversation

    def initial_conversation(self, sample: Sample, sample_index: int) -> Conversation:
        """
        Format the initial conversation for inference.
        """
        if sample_index not in self.envs:
            # Randomly select a game for this trajectory
            selected_game = random.choice(list(self.env_ids.keys()))
            env_id = self.env_ids[selected_game]

            env: TextworldGymEnv = textworld.gym.make(env_id)
            self.envs[sample_index] = env
            self.sample_games[sample_index] = selected_game
            obs, info = env.reset()

            # Initialize conversation with system prompt and first observation
            conversation: Conversation = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": f"### Current Observation\n{obs.strip()}"},
            ]

            self.sample_conversations[sample_index] = conversation
            self.sample_done[sample_index] = False
            self.sample_scores[sample_index] = 0
            logger.info(
                f"Started new trajectory {sample_index} with game: {selected_game}"
            )

        return self.sample_conversations[sample_index]

    def __getitem__(self, index: int) -> ZorkSample:
        """
        Get a sample from the dataset.
        """
        return {"index": index}

    def get_next_state(
        self, sample_index: int, conversation: Conversation
    ) -> tuple[str, bool]:
        """
        After rollout, update any state needed for the next rollout.
        Returns whether the episode is done.
        """
        if self.sample_done[sample_index]:
            raise ValueError("Episode is already done")

        env: TextworldGymEnv = self.envs[sample_index]
        conversation = self.sample_conversations[sample_index]

        last_msg = conversation[-1]["content"]

        try:
            action = parse_command(last_msg)
        except Exception as e:
            logger.error(f"Error parsing command from {last_msg}: {e}")
            return "", True

        obs, score, done, infos = env.step(action)  # type: ignore
        obs = obs.strip()
        logger.info(f"Action: {action}\n Observation: {obs}")
        inventory = infos["inventory"]

        if not done:
            # Add the new observation as a user message
            user_content = f"### Current Observation\n{obs}"
            if inventory and len(inventory.strip()) > 0:
                inventory_formatted = inventory.strip()
                user_content += f"\n\n### Inventory\n{inventory_formatted}"

            conversation.append({"role": "user", "content": user_content})
            self.sample_scores[sample_index] = score  # type: ignore
        else:
            # Clean up completed episode
            game_name = self.sample_games[sample_index]
            logger.info(
                f"Completed trajectory {sample_index} with game: {game_name}, final score: {score}"
            )
            self.sample_done[sample_index] = True
            del self.envs[sample_index]
            del self.sample_conversations[sample_index]
            del self.sample_games[sample_index]
            del self.sample_done[sample_index]
            del self.sample_scores[sample_index]

        return obs, done

    def __len__(self) -> int:
        return 1000

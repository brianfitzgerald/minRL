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
from typing import Literal

from minrl.utils import clean_observation

SYSTEM_PROMPT = """
You are an AI agent whose sole task is to play a text adventure game.  Your primary goal is to explore, solve puzzles, and collect treasures without dying.

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


GameSelectMode = Literal["zork", "random", "zork_series"]


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

        self.game_select_mode: GameSelectMode = "random"
        random.seed(42)

        games_directory = Path(os.getenv("INFORM_GAMES_DIRECTORY", ""))
        game_files_found = os.listdir(games_directory)
        logger.info(f"Found {len(game_files_found)} games in {games_directory}")
        selected_games = []
        if self.game_select_mode == "zork_series":
            selected_games = ["zork1.z5", "zork2.z5", "zork3.z5"]
        elif self.game_select_mode == "zork":
            selected_games = ["zork1.z5"]
        elif self.game_select_mode == "random":
            selected_games = random.sample(game_files_found, 128)

        self.infos = textworld.EnvInfos(
            feedback=True,
            description=True,
            inventory=True,
            intermediate_reward=True,
            entities=True,
            facts=True,
        )

        # Register all available games
        self.env_ids: dict[str, str] = {}
        for game_name in selected_games:
            game_path = games_directory / game_name
            if game_path.exists():
                logger.info(f"Registering game: {game_name}")
                self.env_ids[game_name] = textworld.gym.register_game(
                    game_path.as_posix(), self.infos
                )
            else:
                logger.warning(f"Game file not found: {game_path}")

        self.envs: dict[int, TextworldGymEnv] = {}
        self.sample_games: dict[int, str] = {}  # Track which game each sample uses

        self.completed_episodes = []

    def format_conversation(self, conversation: Conversation) -> Conversation:
        """Format conversation used for inference"""
        return conversation

    def initial_conversation(self, sample: Sample, sample_index: int) -> Conversation:
        """
        Format the initial conversation for inference.
        """
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

        logger.info(f"Started new trajectory {sample_index} with game: {selected_game}")

        return conversation

    def __getitem__(self, index: int) -> ZorkSample:
        """
        Get a sample from the dataset.
        """
        return {"index": index}

    def __len__(self) -> int:
        return 128

    def get_next_state(
        self, sample_index: int, conversation: Conversation
    ) -> tuple[str, bool]:
        """
        After rollout, update any state needed for the next rollout.
        Returns whether the episode is done.
        """
        env: TextworldGymEnv = self.envs[sample_index]

        last_msg = conversation[-1]["content"]

        try:
            action = parse_command(last_msg)
        except Exception as e:
            logger.error(f"Error parsing command from {last_msg}: {e}")
            return "", True

        obs, score, done, infos = env.step(action)  # type: ignore

        obs = clean_observation(obs)

        logger.info(f"Action: {action}")
        inventory = infos["inventory"]
        user_content = obs

        if not done:
            # Add the new observation as a user message
            user_content = f"### Current Observation\n{obs}"
            if inventory and len(inventory.strip()) > 0:
                inventory_formatted = inventory.strip()
                user_content += f"\n\n### Inventory\n{inventory_formatted}"

        else:
            # Clean up completed episode
            game_name = self.sample_games[sample_index]
            logger.info(
                f"Completed trajectory {sample_index} with game: {game_name}, final score: {score}"
            )
            del self.envs[sample_index]
            del self.sample_games[sample_index]

        return user_content, done

from pathlib import Path
from loguru import logger
from minrl.constants import (
    Conversation,
    ConversationMessage,
    HostType,
    Sample,
    StepMetadata,
)
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


class SampleRewardState:
    """Encapsulates all reward-related state and logic for a single sample."""

    def __init__(
        self, initial_location: str, initial_inventory: str, initial_score: int
    ):
        # Location tracking
        self.visited_locations: set[str] = {initial_location}

        # Inventory tracking
        self.previous_inventory: str = initial_inventory
        self.inventory_changes: set[str] = {initial_inventory}

        # Score tracking
        self.previous_score: int = initial_score

        # Turn tracking
        self.turn_count: int = 0

    def calculate_reward(
        self, current_location: str, current_inventory: str, current_score: int
    ) -> float:
        """Calculate reward based on state changes according to the reward specification."""
        reward = 0.0

        # 1. Negative per turn reward (-1 point per turn)
        reward += -1.0

        # 2. New area reward (2 points for discovering new area)
        if current_location not in self.visited_locations:
            reward += 2.0
            self.visited_locations.add(current_location)
        # 3. Moving around reward (0.5 points for revisiting area)
        else:
            reward += 0.5

        # 4. Inventory reward (3 points for new inventory change, 0.5 for repeated)
        if current_inventory != self.previous_inventory:
            if current_inventory not in self.inventory_changes:
                reward += 3.0  # New inventory change
                self.inventory_changes.add(current_inventory)
            else:
                reward += 0.5  # Repeated inventory change

        # 5. In-game score reward (score change * 10)
        score_change = current_score - self.previous_score
        if score_change > 0:
            reward += score_change * 10.0

        # Update state for next step
        self.previous_inventory = current_inventory
        self.previous_score = current_score
        self.turn_count += 1

        return reward


def parse_command(input_string: str) -> str:
    input_string = input_string.strip("\n").replace("COMMAND: ", "")
    command_contents = re.findall(r"<command>(.*?)</command>", input_string, re.DOTALL)
    return command_contents[0]


def zork_reward_func(conversation: Conversation, sample: dict[str, Any]) -> float:
    """Extract reward from the last conversation message's step metadata."""
    if not conversation:
        return 0.0

    last_message = conversation[-1]
    if "step_metadata" in last_message and last_message["step_metadata"]:
        metadata = last_message["step_metadata"]
        if "reward" in metadata:
            return metadata["reward"]

    return 0.0


GameSelectMode = Literal["zork", "random", "zork_series", "full", "known_good_games"]

KNOWN_GOOD_GAMES = {
    "zork1.z5",
    "zork3.z5",
    "ztuu.z5",
    "temple.z5",
    "905.z5",
    "omniquest.z5",
    "jewel 2.z5",
    "awaken 2.z5",
    "Advent.z5",
    "zenon.z5",
    "Balances.z5",
    "Murdac.z5",
    "reverb.z5",
    "awaken.z5",
    "spirit.z5",
    "Balances 2.z5",
    "pentari 2.z5",
    "karn 2.z5",
    "library.z5",
    "loose 2.z5",
    "tryst205.z5",
    "deephome.z5",
    "reverb 2.z5",
    "detective 2.z5",
    "gold.z5",
    "acorncourt.z5",
    "enter.z5",
    "theatre.z5",
    "curses.z5",
    "Adventureland 2.z5",
    "zork2.z5",
    "Advent 2.z5",
    "jewel.z5",
    "sherbet.z5",
    "karn.z5",
    "inhumane.z5",
    "acorncourt 2.z5",
    "loose.z5",
    "night.z5",
    "pentari.z5",
    "Adventureland.z5",
    "detective.z5",
    "ludicorp.z5",
}


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

        self.game_select_mode: GameSelectMode = "zork_series"
        self.samples_per_game = 4

        games_directory = Path(os.getenv("INFORM_GAMES_DIRECTORY", ""))
        game_files_found = os.listdir(games_directory)
        logger.info(f"Found {len(game_files_found)} games in {games_directory}")
        selected_games = []
        if self.game_select_mode == "zork_series":
            selected_games = ["zork1.z5", "zork2.z5", "zork3.z5"]
        elif self.game_select_mode == "zork":
            selected_games = ["zork1.z5"]
        elif self.game_select_mode == "random":
            selected_games = random.sample(game_files_found, 64)
        elif self.game_select_mode == "full":
            selected_games = game_files_found
        elif self.game_select_mode == "known_good_games":
            selected_games = list(KNOWN_GOOD_GAMES)

        self.infos = textworld.EnvInfos(
            feedback=True,
            description=True,
            inventory=True,
            location=True,
            intermediate_reward=True,
            entities=True,
            moves=True,
            score=True,
            max_score=True,
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

        # map of sample index to environment
        self.envs: dict[int, TextworldGymEnv] = {}
        # max score for each game ID
        self.max_scores: dict[str, int] = {}
        self.sample_games: dict[int, str] = {}  # Track which game each sample uses

        # State tracking for reward calculation per sample
        self.sample_reward_states: dict[
            int, SampleRewardState
        ] = {}  # sample_index -> reward state

    def format_conversation(self, conversation: Conversation) -> Conversation:
        """Format conversation used for inference"""
        return conversation

    def initial_conversation(self, sample: Sample, sample_index: int) -> Conversation:
        """
        Format the initial conversation for inference.
        """
        # Deterministically select a game for this trajectory to ensure N samples per game
        game_names = list(self.env_ids.keys())
        game_index = sample_index // self.samples_per_game
        selected_game = game_names[game_index % len(game_names)]
        env_id = self.env_ids[selected_game]

        env: TextworldGymEnv = textworld.gym.make(env_id)
        self.envs[sample_index] = env
        self.sample_games[sample_index] = selected_game
        obs, info = env.reset()

        self.max_scores[selected_game] = info["max_score"]

        # Initialize reward tracking state
        current_location = info["location"].name if info["location"] else "unknown"
        current_inventory = info["inventory"] or ""
        current_score = info["score"]
        self.sample_reward_states[sample_index] = SampleRewardState(
            initial_location=current_location,
            initial_inventory=current_inventory,
            initial_score=current_score,
        )

        # Initialize conversation with system prompt and first observation
        conversation: Conversation = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": obs},
        ]

        logger.info(f"Started new trajectory {sample_index} with game: {selected_game}")

        return conversation

    def __getitem__(self, index: int) -> ZorkSample:
        """
        Get a sample from the dataset.
        """
        return {"index": index}

    def __len__(self) -> int:
        return len(self.env_ids) * self.samples_per_game

    def step(
        self, sample_index: int, conversation: Conversation
    ) -> tuple[ConversationMessage, bool]:
        """
        After rollout, update any state needed for the next rollout.
        Returns whether the episode is done and step metadata.
        """
        env: TextworldGymEnv = self.envs[sample_index]

        last_msg = conversation[-1]["content"]

        try:
            action = parse_command(last_msg)
        except Exception as e:
            logger.error(f"Error parsing command from {last_msg}: {e}")
            return {"role": "user", "content": last_msg}, True

        obs, score, done, infos = env.step(action)  # type: ignore

        obs = clean_observation(obs)

        inventory: str = infos["inventory"]
        current_location = infos["location"].name if infos["location"] else ""
        # Handle score which might be a dict or int
        if isinstance(score, dict):
            current_score = score.get("score", 0)
        else:
            current_score = int(score)

        # Calculate reward based on state changes
        reward_state = self.sample_reward_states[sample_index]
        reward = reward_state.calculate_reward(
            current_location, inventory, current_score
        )

        user_content = obs
        step_metadata: StepMetadata = {
            "observation": obs,
            "inventory": inventory,
            "score": score,  # pyright: ignore[reportAssignmentType]
            "moves": infos["moves"],
            "location": current_location,
            "reward": reward,  # Add reward to metadata
        }

        if not done:
            # Add the new observation as a user message
            user_content = obs
            if inventory and len(inventory.strip()) > 0:
                inventory_formatted = inventory.strip()
                user_content += f"\n\n Inventory: {inventory_formatted}"

        else:
            # Clean up completed episode
            game_name = self.sample_games[sample_index]
            logger.info(
                f"Completed trajectory {sample_index} with game: {game_name}, final score: {score}"
            )
            del self.envs[sample_index]
            del self.sample_games[sample_index]
            del self.max_scores[game_name]

            # Clean up reward tracking state
            if sample_index in self.sample_reward_states:
                del self.sample_reward_states[sample_index]

        return {
            "role": "user",
            "content": user_content,
            "step_metadata": step_metadata,
        }, done

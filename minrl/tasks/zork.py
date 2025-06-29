from loguru import logger
from minrl.constants import HostType
from minrl.tasks.dataset import Episode, MinRLDataset, MiniBatch, Split
from transformers.tokenization_utils_base import PreTrainedTokenizerBase
import textworld
import textworld.gym
from openai.types.chat.chat_completion_message_param import ChatCompletionMessageParam
from typing import Literal, TypedDict
import requests
from pathlib import Path
from textworld.gym.envs import TextworldGymEnv
import re

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

4. Never output internal reasoning.  Only output the next command, prefixed with:
   
   <command>your next command</command>

5. If you ever die or become stuck, output:
   
   <command>restart</command>

EXAMPLE TURN:
You are in a dimly lit room.  To the north is a heavy oak door.  A rusty key lies on the floor.

Your response should be:
   
   <command>take key</command>

That's all.  Ready?  Begin by reading the game's opening description.  When you're ready, output your first COMMAND.
"""

ZGameName = Literal["zork1"]


class ZGame(TypedDict):
    url: str
    filename: str


Z_GAMES: dict[ZGameName, ZGame] = {
    "zork1": {
        "url": "https://github.com/danielricks/textplayer/raw/refs/heads/master/games/zork1.z5",
        "filename": "zork1.z5",
    }
}


class TextWorldAgent:
    """
    Wrapper for an agent that plays an adventure game.
    Contains any state required as well as a list of commands that the agent can take.
    """

    def __init__(self):
        self.commands = []
        self.inventory = None
        self.observation_history = []
        self.description = None
        self.action_history = []
        self.game_name: ZGameName = "zork1"

    def conversation(self) -> list[ChatCompletionMessageParam]:
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
        return [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": user_message,
            },
        ]

    def update(
        self, command: str | None, description: str, observation: str, inventory: str
    ):
        """
        Update agent state after a command and the resulting observation.
        """
        observation = observation.strip("\n")
        if command is not None:
            self.action_history.append(command)
        self.observation_history.append(description)
        self.description = description.strip("\n")
        self.inventory = inventory


def parse_command(input_string: str) -> str:
    input_string = input_string.strip("\n").replace("COMMAND: ", "")
    command_contents = re.findall(r"<command>(.*?)</command>", input_string, re.DOTALL)
    return command_contents[0]


class ZorkDataset(MinRLDataset):
    """
    Dataset where the agent plays multiple steps of a text adventure game,
    and the reward is the sum of the rewards for each step.
    """

    def __init__(
        self,
        split: Split,
        host: HostType,
        tokenizer: PreTrainedTokenizerBase | None = None,
    ):
        super().__init__(split, host, tokenizer)
        self.tokenizer = tokenizer
        # N concurrent game states
        self.n_environments = 1
        self.game_name: ZGameName = "zork1"
        self.game_metadata = Z_GAMES[self.game_name]
        self._download_game_if_needed(self.game_name)
        self.infos = textworld.EnvInfos(
            feedback=True,
            description=True,
            inventory=True,
            verbs=True,
            intermediate_reward=True,
            entities=True,
            facts=True,
        )
        env_id = textworld.gym.register_game(
            f"games/{self.game_metadata['filename']}", self.infos
        )

        self.envs: list[TextworldGymEnv] = [
            textworld.gym.make(env_id) for _ in range(self.n_environments)
        ]
        self.agents: list[TextWorldAgent] = [
            TextWorldAgent() for _ in range(self.n_environments)
        ]

    def _download_game_if_needed(self, game_name: ZGameName):
        games_dir = Path.cwd() / "games"
        games_dir.mkdir(exist_ok=True)

        game_path = games_dir / self.game_metadata["filename"]

        if game_path.exists():
            logger.info(f"Game {game_name} already exists at {game_path}")
            return

        logger.info(f"Downloading game {game_name} to {game_path}")

        response = requests.get(self.game_metadata["url"])
        response.raise_for_status()

        with open(game_path, "wb") as f:
            f.write(response.content)

    def __getitem__(self, i: int) -> dict:
        # Generate a sample for the given index
        # This is a placeholder implementation - you'll need to customize based on your needs
        env_idx = i % self.n_environments
        agent = self.agents[env_idx]
        if len(agent.observation_history) == 0:
            obs, info = self.envs[env_idx].reset()
            agent.update(None, info["description"], obs, info["inventory"])
        conv = agent.conversation()

        return {
            "id": i,
            "conversation": conv,
        }

    def post_generate(self, episode: Episode):
        command = parse_command(episode.text)
        env_idx = episode.batch_index % self.n_environments
        obs, score, done, infos = self.envs[env_idx].step(command)  # type: ignore
        self.agents[env_idx].update(
            command, infos["description"], obs, infos["inventory"].strip("\n")
        )

    def __len__(self) -> int:
        return 10000

    def collate_fn(self, batch: list[dict]) -> MiniBatch:
        """
        Collate examples into a batch.
        Used during training only, requires a tokenizer.
        """
        assert len(batch) == self.n_environments, (
            "Batch size must be >= n_environments, cannot have multiple games in a batch"
        )
        if self.tokenizer is None:
            raise ValueError("Tokenizer is not set")
        prefixes = []
        prefix_token_ids = []
        for item in batch:
            prefix: str = self.tokenizer.apply_chat_template(
                item["conversation"],  # type: ignore
                tokenize=False,
                enable_thinking=False,
            )
            prefixes.append(prefix)
            prefix_token_ids.append(self.tokenizer.encode(prefix))

        return MiniBatch(
            prefixes=prefixes,
            prefix_token_ids=prefix_token_ids,
            samples=batch,
        )

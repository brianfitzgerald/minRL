from minrl.constants import HostType
from minrl.tasks.dataset import MinRLDataset, MiniBatch, Split
from transformers.tokenization_utils_base import PreTrainedTokenizerBase
import textworld
from textworld.core import Environment
import textworld.gym
from openai.types.chat.chat_completion_message_param import ChatCompletionMessageParam

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
   
   COMMAND: <your next command>

5. If you ever die or become stuck, output:
   
   COMMAND: restart

EXAMPLE TURN:
(Game says: "You are in a dimly lit room.  To the north is a heavy oak door.  A rusty key lies on the floor.")

Your response should be:
   
   COMMAND: take key

That's all.  Ready?  Begin by reading the game's opening description.  When you're ready, output your first COMMAND.
"""


class TextWorldAgent:
    """
    Wrapper for an agent that plays an adventure game.
    Contains any state required as well as a list of commands that the agent can take.
    """

    def __init__(self):
        self.commands = []
        self.inventory = None
        self.observation_history = []
        self.action_history = []

    def conversation(self, description: str) -> list[ChatCompletionMessageParam]:
        """Format conversation used for infernce, given current state"""
        action_history_formatted = []
        for action, description in zip(self.action_history, self.observation_history):
            action_history_formatted.append(f"Observation: {description}\n{action}")
        action_history_formatted = "Actions:\n" + "\n".join(action_history_formatted)
        inventory_formatted = self.inventory if self.inventory else ""
        user_msg = f"### History{action_history_formatted}\n### Inventory\n{inventory_formatted}\n### Current state\n{description}"
        print(user_msg)
        return [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": user_msg,
            },
        ]

    def update(self, description: str, inventory: str, action: str) -> str:
        """Update the agent's state based on the description"""
        self.observation_history.append(description.strip("\n\r"))
        self.inventory = inventory
        self.action_history.append(action.replace("COMMAND: ", "").strip("\n\r"))
        return description


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
        self.n_concurrent = 32
        self.envs: list[Environment] = [
            textworld.start("games/zork.z5") for _ in range(self.n_concurrent)
        ]

    def __getitem__(self, i: int) -> dict:
        # Generate a sample for the given index
        # This is a placeholder implementation - you'll need to customize based on your needs
        return {
            "id": i,
            "sample_data": f"sample_{i}",
            # Add other fields as needed for your nethack task
        }

    def __len__(self) -> int:
        return 0

    def conversation(self, sample: dict) -> list[ChatCompletionMessageParam]:
        return []

    def collate_fn(self, batch: list[dict]) -> MiniBatch:
        """
        Collate examples into a batch.
        Used during training only, requires a tokenizer.
        """
        assert len(batch) >= self.n_concurrent, (
            "Batch size must be >= n_environments, cannot have multiple games in a batch"
        )
        if self.tokenizer is None:
            raise ValueError("Tokenizer is not set")
        prefixes = []
        prefix_token_ids = []
        for item in batch:
            prefix: str = self.tokenizer.apply_chat_template(
                self.conversation(item),  # type: ignore
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

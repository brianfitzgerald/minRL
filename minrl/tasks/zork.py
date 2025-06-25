from minrl.constants import HostType
from minrl.tasks.dataset import MinRLDataset, MiniBatch, Split
from transformers.tokenization_utils_base import PreTrainedTokenizerBase
from typing import Any
import textworld
from textworld.core import Environment
import textworld.gym


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

    def conversation(self, sample: dict) -> list[dict[str, Any]]:
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
            prefix: str = self.tokenizer.apply_chat_template(  # type: ignore
                self.conversation(item),
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

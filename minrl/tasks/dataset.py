from abc import abstractmethod
from dataclasses import dataclass
from typing import Any, Literal

from torch.utils.data import Dataset
from transformers.tokenization_utils_base import PreTrainedTokenizerBase
from openai.types.chat.chat_completion_message_param import ChatCompletionMessageParam

from minrl.constants import HostType


Split = Literal["train", "test", "eval"]


@dataclass
class Episode:
    """Store all relevant information of an episode."""

    # Index of sample in batch
    batch_index: int
    answer_index: int
    prefix: str
    # prefix + generated text
    text: str
    # Token IDs of the prefix
    prefix_token_ids: list[int]
    # Token IDs of the generated text
    generated_token_ids: list[int]
    is_finished: bool
    reward: float
    sample: dict[str, Any]


@dataclass
class MiniBatch:
    """Batch of data for each training step."""

    prefixes: list[str]
    prefix_token_ids: list[list[int]]
    samples: list[Any]


class MinRLDataset(Dataset):
    def __init__(
        self,
        split: Split,
        host: HostType,
        tokenizer: PreTrainedTokenizerBase | None = None,
        batch_size: int = 4,
    ):
        self.split = split
        self.tokenizer = tokenizer
        self.host = host
        self.batch_size = batch_size

    @abstractmethod
    def collate_fn(self, batch: list[dict]) -> MiniBatch:
        pass

    @abstractmethod
    def conversation(self, sample: dict[str, Any]) -> list[ChatCompletionMessageParam]:
        """Conversation used to generate the prefix, or the prompt for evals."""
        pass

    @abstractmethod
    def post_generate(self, episode: Episode):
        """Some datasets have an internal state that needs to be updated after generation."""
        pass

    @abstractmethod
    def reward_function(self, response: str, sample: dict[str, Any]) -> float:
        """Reward function for the dataset."""
        pass

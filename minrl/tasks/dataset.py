from abc import abstractmethod
from dataclasses import dataclass
from typing import Any, Literal

from torch.utils.data import Dataset
from transformers.tokenization_utils_base import PreTrainedTokenizerBase
from openai.types.chat.chat_completion_message import ChatCompletionMessage

from minrl.constants import HostType


Split = Literal["train", "test", "eval"]


@dataclass
class Episode:
    """Store all relevant information of an episode."""

    prefix: str
    text: str
    # Token IDs of the prefix
    prefix_token_ids: list[int]
    # Token IDs of the generated text
    generated_token_ids: list[int]
    is_finished: bool
    reward: float
    reward_info: dict[str, float]


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
    ):
        self.split = split
        self.tokenizer = tokenizer
        self.host = host

    @abstractmethod
    def collate_fn(self, batch: list[dict]) -> MiniBatch:
        pass

    @abstractmethod
    def conversation(self, sample: dict[str, Any]) -> list[ChatCompletionMessage]:
        """Conversation used to generate the prefix, or the prompt for evals."""
        pass

from abc import abstractmethod
from dataclasses import dataclass
from typing import Literal

from torch.utils.data import Dataset
from transformers.tokenization_utils_base import PreTrainedTokenizerBase
from openai.types.chat.chat_completion_message_param import ChatCompletionMessageParam

from minrl.constants import Conversation, HostType, Sample


Split = Literal["train", "test", "eval"]


@dataclass
class Episode:
    """Store all relevant information of an episode."""

    # Index of sample in batch
    batch_index: int
    answer_index: int
    finished: bool
    reward: float
    conversation: Conversation


@dataclass
class MiniBatch:
    """Batch of data for each training step."""

    conversations: list[Conversation]
    samples: list[Sample]


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
    def conversation(self, sample: Sample) -> Conversation:
        """Conversation used to generate a completion."""
        pass

    @abstractmethod
    def post_generate(self, episode: Episode):
        """Some datasets have an internal state that needs to be updated after generation."""
        pass

    @abstractmethod
    def reward_function(self, response: str, sample: Sample) -> float:
        """Reward function for the dataset."""
        pass

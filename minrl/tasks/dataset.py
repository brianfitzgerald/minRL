from abc import abstractmethod
from dataclasses import dataclass
from typing import Literal

from torch.utils.data import Dataset
from transformers.tokenization_utils_base import PreTrainedTokenizerBase

from minrl.constants import Conversation, HostType, Sample


Split = Literal["train", "test", "eval"]

EpisodeStatus = Literal["finished", "terminated"]


@dataclass
class Episode:
    """Store all relevant information of an episode."""

    # Index of group in batch
    group_index: int
    # Index of answer in group
    answer_index: int
    # Whether the episode finished or terminated early
    reward: float
    conversation: Conversation
    sample: Sample


class MinRLDataset(Dataset):
    """
    Base class for all datasets.
    Each dataset has a step wise inference function and a max number of steps.
    """

    max_steps: int = 1
    max_tokens: int = 1024

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
    def initial_conversation(self, sample: Sample, sample_index: int) -> Conversation:
        """
        Given a sample, format the initial conversation for inference.
        """
        raise NotImplementedError("format_initial_conversation is not implemented")

    @abstractmethod
    def get_next_state(
        self, sample_index: int, conversation: Conversation
    ) -> tuple[str, bool]:
        """
        After each turn, get the next state of the environment based on the model response.
        Returns (obs, done)
        """
        raise NotImplementedError("post_generation is not implemented")

from abc import abstractmethod
from dataclasses import dataclass
from typing import Literal

from torch.utils.data import Dataset
from transformers.tokenization_utils_base import PreTrainedTokenizerBase

from minrl.constants import Conversation, HostType, InferenceFunction, Sample


Split = Literal["train", "test", "eval"]


@dataclass
class Episode:
    """Store all relevant information of an episode."""

    # Index of group in batch
    group_index: int
    # Index of answer in group
    answer_index: int
    finished: bool
    reward: float
    conversation: Conversation


@dataclass
class MiniBatch:
    """Batch of data for each training step."""

    conversations: list[Conversation]
    samples: list[Sample]

    def __len__(self) -> int:
        return len(self.samples)


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
    def conversation(self, sample: Sample, sample_index: int) -> Conversation:
        """
        Given a sample, format the conversation for inference.
        """

    def post_rollout(self, sample_index: int, model_response: str) -> bool:
        """
        After rollout, update any state needed for the next rollout.
        Return whether the episode is done, true by default for single turn inference.
        """
        return True

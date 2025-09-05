from abc import abstractmethod
from typing import Literal

from torch.utils.data import Dataset
from transformers.tokenization_utils_base import PreTrainedTokenizerBase

from minrl.constants import Conversation, HostType, Sample, StepMetadata


Split = Literal["train", "test", "eval"]

EpisodeStatus = Literal["finished", "terminated"]


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
    def step(
        self, sample_index: int, conversation: Conversation
    ) -> tuple[str, bool, StepMetadata]:
        """
        After each turn, get the next state of the environment based on the model response.
        Returns (obs, done)
        """
        raise NotImplementedError("post_generation is not implemented")

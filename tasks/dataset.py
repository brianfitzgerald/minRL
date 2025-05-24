from abc import abstractmethod
from dataclasses import dataclass
import torch

from typing import Any, Literal

from torch import Tensor
from torch.utils.data import Dataset
from transformers.tokenization_utils_base import PreTrainedTokenizerBase


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
    # Only populated with vllm
    generated_logprobs: Tensor


@dataclass
class MiniBatch:
    """Batch of data for each training step."""

    prefixes: list[str]
    prefix_token_ids: list[list[int]]
    samples: list[dict[str, Any]]


class MinRLDataset(Dataset):
    def __init__(self, split: Split, tokenizer: PreTrainedTokenizerBase | None = None):
        self.split = split
        self.tokenizer = tokenizer

    @abstractmethod
    def collate_fn(self, batch: list[dict]) -> MiniBatch:
        pass


def batch_to_samples(batch: dict) -> list:
    """Convert a batch of data to a list of samples."""
    first_key = next(iter(batch.values()))
    batch_size = len(first_key)
    return [{k: v[i] for k, v in batch.items()} for i in range(batch_size)]

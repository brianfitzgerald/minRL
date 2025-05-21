from dataclasses import dataclass

from typing import Any
from torch import Tensor


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
    prefix_tokens: list[list[str]]
    prefix_token_ids: list[list[int]]
    samples: list[dict[str, Any]]
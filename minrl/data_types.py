from dataclasses import dataclass
from typing import Dict, List

from torch import Tensor


@dataclass
class Episode:
    """Store all relevant information of an episode."""

    prefix: str
    text: str
    prefix_token_ids: list[int]
    generated_token_ids: list[int]
    is_finished: bool
    reward: float
    reward_info: Dict[str, float]
    # Only populated with vllm
    generated_logprobs: Tensor


@dataclass
class MiniBatch:
    """Batch of data for each training step."""

    prefixes: list[str]
    prefix_tokens: list[list[str]]
    prefix_token_ids: list[list[int]]
    answer: list[str]
    answer_groups: list[list[str]]
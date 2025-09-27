import pytest
import torch
from minrl.algorithms import update_policy, normalize_rewards_per_group
from minrl.constants import TrainerConfig, Conversation
from minrl.constants import Episode
from transformers.tokenization_utils_base import PreTrainedTokenizerBase
from vllm import LLM

import torch.nn as nn


@pytest.fixture
def sample_episodes() -> list[Episode]:
    """Fixture that creates sample episodes for testing."""
    conversations: list[Conversation] = [
        [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there!"},
        ],
        [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hey!"},
        ],
    ]

    episodes = [
        Episode(
            group_index=0,
            answer_index=0,
            sample={"id": 0, "prompt": "Hello"},
            reward=1.0,
            conversation=conversations[0],
        ),
        Episode(
            group_index=0,
            answer_index=1,
            sample={"id": 0, "prompt": "Hello"},
            reward=0.5,
            conversation=conversations[1],
        ),
    ]
    return episodes


def test_normalize_rewards_per_group(sample_episodes):
    """Test reward normalization per group."""
    normalized_episodes = normalize_rewards_per_group(sample_episodes)

    # Check that we have the same number of episodes
    assert len(normalized_episodes) == len(sample_episodes)

    # Check that rewards are normalized (mean should be 0)
    rewards = [e.reward for e in normalized_episodes]
    assert abs(sum(rewards) / len(rewards)) < 1e-6  # Mean should be ~0

    # Check that other episode attributes are preserved
    for orig, norm in zip(sample_episodes, normalized_episodes):
        assert orig.group_index == norm.group_index
        assert orig.answer_index == norm.answer_index
        assert orig.sample == norm.sample
        assert orig.conversation == norm.conversation


def test_update_policy_grpo_algorithm(
    hf_model: nn.Module,
    config: TrainerConfig,
    tokenizer: PreTrainedTokenizerBase,
    device: torch.device,
    sample_episodes: list[Episode],
):
    """Test update_policy with GRPO algorithm."""
    config.algorithm = "grpo"

    pad_token_id: int = tokenizer.pad_token_id or 0  # type: ignore
    optimizer = torch.optim.Adam(hf_model.parameters(), lr=0.001)  # type: ignore

    results = update_policy(
        model=hf_model,
        optimizer=optimizer,
        episodes=sample_episodes,
        micro_batch_size=2,
        pad_token_id=pad_token_id,
        max_grad_norm=1.0,
        device=device,
        algorithm=config.algorithm,
        tokenizer=tokenizer,
        apply_loss=True,
    )

    assert results is not None

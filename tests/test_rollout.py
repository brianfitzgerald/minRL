import pytest
from minrl.algorithms import rollout
from minrl.constants import Conversation
from vllm import LLM
from transformers.tokenization_utils_base import PreTrainedTokenizerBase
from minrl.constants import TrainerConfig


def test_rollout_with_sample_batch(
    config: TrainerConfig,
    tokenizer: PreTrainedTokenizerBase,
    vllm_model: LLM,
):
    """Test rollout with a sample batch containing conversations."""
    conversations: list[Conversation] = [
        [
            {"role": "user", "content": "Hello, how are you?"},
        ]
    ]
    samples = [{"id": 0, "prompt": "Hello, how are you?"}]
    episodes = rollout(
        config=config,
        tokenizer=tokenizer,
        group_size=1,
        max_steps=1,
        conversations=conversations,
        samples=samples,
        reward_function=lambda x, y: 0.5,
        vllm_model=vllm_model,
    )

    assert isinstance(episodes, list)
    assert len(episodes) > 0

    # Check episode structure
    episode = episodes[0]
    assert episode.reward == 0.5


def test_rollout_with_multiple_turns(
    config: TrainerConfig,
    tokenizer: PreTrainedTokenizerBase,
    vllm_model: LLM,
):
    """Test rollout with a sample batch containing conversations."""
    conversations: list[Conversation] = [
        [
            {"role": "user", "content": "Hello, how are you?"},
        ]
    ]
    samples = [{"id": 0, "prompt": "Hello, how are you?"}]
    episodes = rollout(
        config=config,
        tokenizer=tokenizer,
        group_size=4,
        max_steps=3,
        conversations=conversations,
        samples=samples,
        reward_function=lambda x, y: 0.5,
        vllm_model=vllm_model,
    )

    assert isinstance(episodes, list)
    assert len(episodes) > 0

    # Check episode structure
    episode = episodes[0]
    assert episode.reward == 0.5

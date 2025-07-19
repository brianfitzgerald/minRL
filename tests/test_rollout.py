import pytest
import torch
from minrl.algorithms import rollout
from minrl.constants import SMOL_LM_2_135M, TrainerConfig, Conversation
from vllm import LLM
from minrl.tasks.dataset import MiniBatch
from transformers.models.auto.tokenization_auto import AutoTokenizer
from loguru import logger

from transformers.tokenization_utils_base import PreTrainedTokenizerBase


@pytest.fixture(scope="session")
def vllm_model():
    """Fixture that creates a vLLM model instance once per test session."""
    logger.info("Creating vLLM model")
    model = LLM(
        model=SMOL_LM_2_135M,
        device="cuda:0" if torch.cuda.is_available() else "cpu",
        dtype="bfloat16",
        enforce_eager=True,
        gpu_memory_utilization=0.2,
    )
    logger.info("vLLM model created")
    return model


@pytest.fixture
def tokenizer():
    """Fixture that creates a tokenizer for testing."""
    return AutoTokenizer.from_pretrained(SMOL_LM_2_135M)


@pytest.fixture
def config():
    """Fixture that creates a TrainerConfig for testing."""
    return TrainerConfig()


@pytest.fixture
def sample_batch():
    """Fixture that creates a sample MiniBatch with conversations."""
    conversations: list[Conversation] = [
        [
            {"role": "user", "content": "Hello, how are you?"},
        ]
    ]
    samples = [{"id": 0, "prompt": "Hello, how are you?"}]
    return MiniBatch(conversations=conversations, samples=samples)


def test_rollout_with_sample_batch(
    config: TrainerConfig,
    tokenizer: PreTrainedTokenizerBase,
    sample_batch: MiniBatch,
    vllm_model: LLM,
):
    """Test rollout with a sample batch containing conversations."""
    episodes = rollout(
        config=config,
        tokenizer=tokenizer,
        batch=sample_batch,
        max_new_tokens=50,
        num_answers_per_question=1,
        max_turns=1,
        reward_function=lambda x, y: 0.5,
        vllm_model=vllm_model,
    )

    assert isinstance(episodes, list)
    assert len(episodes) > 0

    # Check episode structure
    episode = episodes[0]
    assert episode.reward == 0.5

import pytest
import torch
from minrl.constants import SMOL_LM_2_135M, TrainerConfig
from vllm import LLM
from vllm.envs import set_vllm_use_v1
from transformers.models.auto.tokenization_auto import AutoTokenizer
from loguru import logger
from transformers.models.auto.modeling_auto import AutoModelForCausalLM
import torch.nn as nn

from minrl.metrics import DummyMetricsWrapper


@pytest.fixture(scope="session")
def vllm_model():
    """Fixture that creates a vLLM model instance once per test session."""
    logger.info("Creating vLLM model")
    # Disable V1 engine to match trainer behavior
    set_vllm_use_v1(False)
    # vLLM only supports CUDA and CPU, not MPS
    model = LLM(
        model=SMOL_LM_2_135M,
        enforce_eager=True,
    )
    logger.info("vLLM model created")
    return model


@pytest.fixture(scope="session")
def hf_model() -> nn.Module:
    """Fixture that creates a Hugging Face model instance once per test session."""
    logger.info("Creating HF model")
    # Determine device for HF model (can use MPS, unlike vLLM)
    device = (
        "cuda:0"
        if torch.cuda.is_available()
        else ("mps" if torch.backends.mps.is_available() else "cpu")
    )
    logger.info(f"Creating HF model on device {device}")
    model = AutoModelForCausalLM.from_pretrained(
        SMOL_LM_2_135M,
        device_map={"": device},  # Load directly on target device
        dtype=torch.bfloat16,
        attn_implementation=(
            "flash_attention_2" if torch.cuda.is_available() else "eager"
        ),
    )

    logger.info("Hugging Face model created")
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
def device():
    """Fixture that provides a device for testing."""
    if torch.cuda.is_available():
        return torch.device("cuda:0")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")


class MockVLLMOutput:
    """Mock vLLM output that mimics the structure of vLLM's RequestOutput."""

    def __init__(self, text: str):
        self.text = text


class MockVLLMRequestOutput:
    """Mock vLLM RequestOutput that contains multiple outputs."""

    def __init__(self, outputs: list[MockVLLMOutput]):
        self.outputs = outputs


MOCK_VLLM_RESPONSE = "Mock response"


class MockVLLMModel:
    """Mock vLLM model that always returns a constant response."""

    def generate(self, prompts, sampling_params):
        """Mock generate method that returns consistent responses."""
        # Extract the number of completions from sampling_params
        n = getattr(sampling_params, "n", 1)

        # Create mock outputs for each prompt
        mock_outputs = []
        for _ in prompts:
            # Create n outputs per prompt (for group_size > 1)
            if n == 1:
                # For single response, don't append number
                outputs = [MockVLLMOutput(MOCK_VLLM_RESPONSE)]
            else:
                # For multiple responses, append numbers to distinguish them
                outputs = [
                    MockVLLMOutput(f"{MOCK_VLLM_RESPONSE} batch_idx={j}")
                    for j in range(n)
                ]
            mock_outputs.append(MockVLLMRequestOutput(outputs))

        return mock_outputs


@pytest.fixture
def mock_vllm_model():
    """Fixture that provides a mock vLLM model for testing."""
    return MockVLLMModel()


@pytest.fixture
def dummy_metrics_wrapper():
    """Fixture that provides a dummy metrics wrapper for testing."""
    return DummyMetricsWrapper(
        logger_choice="tensorboard",
        task="connections",
        trainer_config=TrainerConfig(),
        run_name="test",
    )

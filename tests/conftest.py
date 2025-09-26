import pytest
import torch
from minrl.constants import SMOL_LM_2_135M, TrainerConfig
from vllm import LLM
from transformers.models.auto.tokenization_auto import AutoTokenizer
from loguru import logger
from transformers.models.auto.modeling_auto import AutoModelForCausalLM
import torch.nn as nn


@pytest.fixture(scope="session")
def vllm_model():
    """Fixture that creates a vLLM model instance once per test session."""
    logger.info("Creating vLLM model")
    # vLLM only supports CUDA and CPU, not MPS
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    model = LLM(
        model=SMOL_LM_2_135M,
        enforce_eager=True,
        gpu_memory_utilization=0.2,
    )
    logger.info("vLLM model created")
    return model


@pytest.fixture(scope="session")
def hf_model() -> nn.Module:
    """Fixture that creates a Hugging Face model instance once per test session."""
    logger.info("Creating Hugging Face model")
    # Determine device for HF model (can use MPS, unlike vLLM)
    device = (
        "cuda:0"
        if torch.cuda.is_available()
        else ("mps" if torch.backends.mps.is_available() else "cpu")
    )

    model = AutoModelForCausalLM.from_pretrained(
        SMOL_LM_2_135M,
        device_map={"": device},  # Load directly on target device
        torch_dtype=torch.bfloat16,
        attn_implementation=(
            "flash_attention_2" if torch.cuda.is_available() else "sdpa"
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

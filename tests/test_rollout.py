from minrl.algorithms import rollout
from minrl.constants import Conversation
from transformers.tokenization_utils_base import PreTrainedTokenizerBase
from minrl.constants import TrainerConfig
from tests.conftest import MOCK_VLLM_RESPONSE, MockVLLMModel


def test_rollout_with_sample_batch(
    config: TrainerConfig,
    tokenizer: PreTrainedTokenizerBase,
    mock_vllm_model: MockVLLMModel,
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
        vllm_model=mock_vllm_model,  # type: ignore
    )

    assert isinstance(episodes, list)
    assert len(episodes) > 0

    # Check episode structure
    episode = episodes[0]
    assert episode.reward == 0.5

    # Verify that the mock model returned the expected text
    # The conversation should have the assistant's response
    assert len(episode.conversation) >= 2  # user message + assistant response
    assistant_message = episode.conversation[1]
    assert assistant_message["role"] == "assistant"
    assert assistant_message["content"] == MOCK_VLLM_RESPONSE


def test_rollout_with_multiple_turns(
    config: TrainerConfig,
    tokenizer: PreTrainedTokenizerBase,
    mock_vllm_model: MockVLLMModel,
):
    """Test rollout with a sample batch containing conversations."""

    group_size = 4
    conversations: list[Conversation] = [
        [
            {"role": "user", "content": "Hello, how are you?"},
        ]
    ]
    samples = [{"id": 0, "prompt": "Hello, how are you?"}]
    episodes = rollout(
        config=config,
        tokenizer=tokenizer,
        group_size=group_size,
        max_steps=3,
        conversations=conversations,
        samples=samples,
        reward_function=lambda x, y: 0.5,
        vllm_model=mock_vllm_model,  # type: ignore
    )

    assert isinstance(episodes, list)
    assert len(episodes) == len(conversations) * group_size

    # Check episode structure
    episode = episodes[0]
    assert episode.reward == 0.5

    # Verify that the mock model returned the expected text
    # The conversation should have the assistant's response
    assert len(episode.conversation) >= 2  # user message + assistant response
    assistant_message = episode.conversation[1]
    assert assistant_message["role"] == "assistant"
    assert assistant_message["content"] == f"{MOCK_VLLM_RESPONSE} batch_idx=0"

    # Test that each episode uses the correct response index
    for i, episode in enumerate(episodes):
        assert episode.group_index == 0
        assert episode.answer_index == i
        assert len(episode.conversation) == 4
        for j in range(1, 4):
            assert episode.conversation[j]["role"] == "assistant"
        # The first assistant message should have the correct response index
        first_assistant_message = episode.conversation[1]
        assert (
            first_assistant_message["content"] == f"{MOCK_VLLM_RESPONSE} batch_idx={i}"
        )

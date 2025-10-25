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
    episodes, rollout_duration = rollout(
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

    group_size, n_groups = 4, 6
    conversations: list[Conversation] = [
        [
            {"role": "user", "content": f"Hello, how are you? Group {i}"},
        ]
        for i in range(n_groups)
    ]
    samples = [
        {"id": i, "prompt": f"Hello, how are you? Group {i}"} for i in range(n_groups)
    ]

    episodes, rollout_duration = rollout(
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
    assert isinstance(rollout_duration, float)
    assert rollout_duration >= 0
    assert len(episodes) == len(conversations) * group_size

    # Test that each episode uses the correct group and answer indices
    for i, episode in enumerate(episodes):
        expected_group_index = i // group_size
        expected_answer_index = i % group_size

        assert episode.group_index == expected_group_index
        assert episode.answer_index == expected_answer_index
        assert len(episode.conversation) == 4  # 1 user message + 3 assistant messages

        # All assistant messages should have the correct role
        for j in range(1, 4):
            assert episode.conversation[j]["role"] == "assistant"

        # The first assistant message should have the correct response index
        # For the first step (step_idx=0), vLLM generates group_size responses per conversation
        # The mock model appends "batch_idx={j}" where j is the response index within the group
        first_assistant_message = episode.conversation[1]
        assert (
            first_assistant_message["content"]
            == f"{MOCK_VLLM_RESPONSE} batch_idx={expected_answer_index}"
        )

        # Verify the conversation starts with the correct user message for this group
        user_message = episode.conversation[0]
        assert user_message["role"] == "user"
        assert (
            user_message["content"]
            == f"Hello, how are you? Group {expected_group_index}"
        )

import torch
from minrl.algorithms import preprocess_batch, get_token_ids_and_assistant_mask
from minrl.constants import Episode, Conversation
from transformers.tokenization_utils_base import PreTrainedTokenizerBase
import torch.nn as nn


def get_pad_token_id(tokenizer: PreTrainedTokenizerBase) -> int:
    """Helper to safely get pad token ID."""
    pad_id = tokenizer.pad_token_id
    if pad_id is None:
        pad_id = tokenizer.eos_token_id
    assert isinstance(pad_id, int), "pad_token_id must be an int"
    return pad_id


def create_test_episode(
    conversation: Conversation,
    group_index: int = 0,
    answer_index: int = 0,
    reward: float = 0.5,
) -> Episode:
    """Helper to create test episodes."""
    return Episode(
        group_index=group_index,
        answer_index=answer_index,
        reward=reward,
        conversation=conversation,
        sample={"id": group_index},
    )


def test_process_batch_single_turn(
    hf_model: nn.Module,
    tokenizer: PreTrainedTokenizerBase,
    device: torch.device,
):
    """Test process_batch with a single user-assistant turn."""
    # Create simple single-turn episodes
    episodes = [
        create_test_episode(
            group_index=0,
            answer_index=0,
            conversation=[
                {"role": "user", "content": "Hello"},
                {"role": "assistant", "content": "Hi there"},
            ],
            reward=1.0,
        ),
        create_test_episode(
            group_index=0,
            answer_index=1,
            conversation=[
                {"role": "user", "content": "Hello"},
                {"role": "assistant", "content": "Greetings"},
            ],
            reward=0.5,
        ),
    ]

    pad_token_id = get_pad_token_id(tokenizer)

    logprobs, target_masks, batch_rewards, entropy, n_target_tokens = preprocess_batch(
        model=hf_model,
        episodes=episodes,
        tokenizer=tokenizer,
        pad_token_id=pad_token_id,
        device=device,
    )

    # Check output shapes
    assert logprobs.shape[0] == 2  # batch size
    assert target_masks.shape[0] == 2  # batch size
    assert logprobs.shape == target_masks.shape
    assert batch_rewards.shape == (2,)
    assert isinstance(entropy, torch.Tensor)
    assert isinstance(n_target_tokens, int)

    # Check rewards are correct
    assert batch_rewards[0] == 1.0
    assert batch_rewards[1] == 0.5

    # Check that target_masks only mark assistant tokens
    # For each episode, verify that some tokens are masked (assistant tokens)
    for i in range(2):
        # Get the token IDs and assistant mask for verification
        token_ids, assistant_mask = get_token_ids_and_assistant_mask(
            episodes[i].conversation, tokenizer
        )

        # After shifting for next-token prediction, we should have masks
        # that correspond to the assistant portion
        episode_mask = target_masks[i]

        # At least some tokens should be masked (assistant content)
        assert episode_mask.sum() > 0, f"Episode {i} has no masked tokens"

        # Not all tokens should be masked (user prompt exists)
        assert episode_mask.sum() < episode_mask.shape[0], (
            f"Episode {i} has all tokens masked"
        )


def test_process_batch_multi_turn(
    hf_model: nn.Module,
    tokenizer: PreTrainedTokenizerBase,
    device: torch.device,
):
    """Test process_batch with multi-turn conversations."""
    episodes = [
        create_test_episode(
            group_index=0,
            answer_index=0,
            conversation=[
                {"role": "user", "content": "What is 2+2?"},
                {"role": "assistant", "content": "4"},
                {"role": "user", "content": "And 3+3?"},
                {"role": "assistant", "content": "6"},
            ],
            reward=2.0,
        ),
    ]

    pad_token_id = get_pad_token_id(tokenizer)

    logprobs, target_masks, batch_rewards, entropy, n_target_tokens = preprocess_batch(
        model=hf_model,
        episodes=episodes,
        tokenizer=tokenizer,
        pad_token_id=pad_token_id,
        device=device,
    )

    # Check output shapes
    assert logprobs.shape[0] == 1
    assert target_masks.shape[0] == 1
    assert batch_rewards.shape == (1,)
    assert batch_rewards[0] == 2.0

    # Verify assistant tokens are marked
    episode_mask = target_masks[0]
    assert episode_mask.sum() > 0

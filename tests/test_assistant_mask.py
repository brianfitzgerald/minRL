import pytest
from transformers import AutoTokenizer
from minrl.algorithms import get_token_ids_and_assistant_mask
from minrl.constants import Conversation, GEMMA_3_1B, QWEN_3_0_6B


def visualize_token_mask(token_ids, assistant_mask, tokenizer, title=None):
    """Visualize token mask with colors showing which tokens contribute to loss."""

    # The assistant_mask is already the loss mask (True for assistant tokens)
    loss_mask = assistant_mask

    # ANSI color codes
    LOSS_COLOR = "\033[91m"  # Red for tokens that contribute to loss
    NO_LOSS_COLOR = "\033[90m"  # Gray for tokens that don't contribute to loss
    RESET = "\033[0m"

    print("\n" + "=" * 80)
    if title:
        print(f"LOSS MASK VISUALIZATION: {title}")
    else:
        print("LOSS MASK VISUALIZATION (Color-coded by loss contribution)")
    print("=" * 80)

    # Print tokens with color coding based on loss mask
    print("Decoded conversation with loss mask colors:")
    print(
        f"Legend: {LOSS_COLOR}Contributes to Loss{RESET} | {NO_LOSS_COLOR}No Loss Contribution{RESET}"
    )
    print("-" * 80)

    # Build colored output by grouping consecutive tokens with same loss contribution
    output = ""
    current_contributes_to_loss = None
    current_tokens = []

    for token_id, contributes_to_loss in zip(token_ids, loss_mask):
        if contributes_to_loss != current_contributes_to_loss:
            # Print accumulated tokens with previous group's color
            if current_tokens:
                token_text = tokenizer.decode(current_tokens)
                color = LOSS_COLOR if current_contributes_to_loss else NO_LOSS_COLOR
                output += f"{color}{token_text}{RESET}"

            current_contributes_to_loss = contributes_to_loss
            current_tokens = [token_id]
        else:
            current_tokens.append(token_id)

    # Print final group
    if current_tokens:
        token_text = tokenizer.decode(current_tokens)
        color = LOSS_COLOR if current_contributes_to_loss else NO_LOSS_COLOR
        output += f"{color}{token_text}{RESET}"

    print(output)
    print("-" * 80)

    # Print loss contribution summary
    print("\nLOSS CONTRIBUTION SEGMENTS:")
    print("-" * 40)

    current_contributes_to_loss = None
    segment_start = 0
    loss_tokens = sum(loss_mask)
    total_tokens = len(loss_mask)

    for i, contributes_to_loss in enumerate(loss_mask + [None]):
        if contributes_to_loss != current_contributes_to_loss:
            if current_contributes_to_loss is not None:
                # Print previous segment
                segment_tokens = token_ids[segment_start:i]
                segment_text = tokenizer.decode(segment_tokens)
                color = LOSS_COLOR if current_contributes_to_loss else NO_LOSS_COLOR
                status = "LOSS" if current_contributes_to_loss else "NO_LOSS"
                print(
                    f"{color}{status:7s}{RESET} ({segment_start:3d}-{i - 1:3d}): {repr(segment_text)}"
                )

            current_contributes_to_loss = contributes_to_loss
            segment_start = i

    print("-" * 40)
    print(
        f"Loss tokens: {loss_tokens}/{total_tokens} ({100 * loss_tokens / total_tokens:.1f}%)"
    )


def test_get_token_ids_and_assistant_mask(tokenizer):
    """Test get_token_ids_and_assistant_mask function with mask visualization."""

    # Create a test conversation with different roles
    conversation: Conversation = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "What is 2+2?"},
        {"role": "assistant", "content": "The answer is 4."},
    ]

    # Get token IDs and assistant mask
    token_ids, assistant_mask = get_token_ids_and_assistant_mask(
        conversation, tokenizer
    )

    # Basic assertions
    assert len(token_ids) == len(assistant_mask), (
        "Token IDs and assistant mask must have same length"
    )
    assert len(token_ids) > 0, "Should have tokens"
    assert len(assistant_mask) > 0, "Should have assistant mask"

    # Check assistant token distribution
    assert True in assistant_mask, "Should have assistant tokens"
    assert False in assistant_mask, "Should have non-assistant tokens"

    # Verify token sequences - first tokens should be non-assistant, last should be assistant
    assert not assistant_mask[0], "First token should be non-assistant"
    assert assistant_mask[-1], "Last token should be assistant"

    # Visualize the token mask
    visualize_token_mask(
        token_ids, assistant_mask, tokenizer, "Standard 3-message conversation"
    )


@pytest.mark.parametrize("model_id", [GEMMA_3_1B, QWEN_3_0_6B])
def test_assistant_mask_with_different_models(model_id):
    """Test that get_token_ids_and_assistant_mask works correctly with different model tokenizers."""

    # Create a tokenizer for the specific model
    tokenizer = AutoTokenizer.from_pretrained(model_id)

    # Create a test conversation with system, user, and assistant messages
    conversation: Conversation = [
        {"role": "system", "content": "You are a helpful AI assistant."},
        {"role": "user", "content": "What is the capital of France?"},
        {"role": "assistant", "content": "The capital of France is Paris."},
    ]

    # Get token IDs and assistant mask
    token_ids, assistant_mask = get_token_ids_and_assistant_mask(
        conversation, tokenizer
    )

    # Basic assertions
    assert len(token_ids) == len(assistant_mask), (
        f"Token IDs and assistant mask must have same length for {model_id}"
    )
    assert len(token_ids) > 0, f"Should have tokens for {model_id}"
    assert len(assistant_mask) > 0, f"Should have assistant mask for {model_id}"

    # Check that we have both True and False in the mask
    assert True in assistant_mask, (
        f"Should have assistant tokens (True) in mask for {model_id}"
    )
    assert False in assistant_mask, (
        f"Should have non-assistant tokens (False) in mask for {model_id}"
    )

    # Verify that the first tokens (system/user) are not marked as assistant
    assert not assistant_mask[0], f"First token should be non-assistant for {model_id}"

    # Verify that the last tokens (assistant response) are marked as assistant
    assert assistant_mask[-1], f"Last token should be assistant for {model_id}"

    # Count assistant tokens - should be less than total (since we have system + user messages)
    num_assistant_tokens = sum(assistant_mask)
    total_tokens = len(assistant_mask)
    assert 0 < num_assistant_tokens < total_tokens, (
        f"Assistant tokens should be a subset of total tokens for {model_id}. "
        f"Got {num_assistant_tokens}/{total_tokens}"
    )

    # Visualize the token mask for debugging
    visualize_token_mask(token_ids, assistant_mask, tokenizer, f"Model: {model_id}")

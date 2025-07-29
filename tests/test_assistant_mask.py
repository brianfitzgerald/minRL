import pytest
from minrl.algorithms import get_token_ids_and_assistant_mask
from minrl.constants import Conversation


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
    assert assistant_mask[0] == False, "First token should be non-assistant"
    assert assistant_mask[-1] == True, "Last token should be assistant"

    # Visualize the token mask
    visualize_token_mask(
        token_ids, assistant_mask, tokenizer, "Standard 3-message conversation"
    )


def test_get_token_ids_and_assistant_mask_error_cases(tokenizer):
    """Test error cases for get_token_ids_and_assistant_mask."""

    # Test with empty conversation
    with pytest.raises(ValueError, match="Conversation must have at least 1 message"):
        get_token_ids_and_assistant_mask([], tokenizer)

    # Test with single user message (should work now)
    token_ids, assistant_mask = get_token_ids_and_assistant_mask(
        [{"role": "user", "content": "Hello"}], tokenizer
    )
    assert len(token_ids) > 0
    assert len(assistant_mask) == len(token_ids)
    assert not any(assistant_mask)  # All should be False since no assistant messages


def test_get_token_ids_and_assistant_mask_minimal(tokenizer):
    """Test with minimal valid conversation."""

    conversation: Conversation = [
        {"role": "user", "content": "Hi"},
        {"role": "assistant", "content": "Hello"},
    ]

    token_ids, assistant_mask = get_token_ids_and_assistant_mask(
        conversation, tokenizer
    )

    assert len(token_ids) == len(assistant_mask)
    assert True in assistant_mask, "Should have assistant tokens"
    assert False in assistant_mask, "Should have non-assistant tokens"

    # Should start with non-assistant tokens and end with assistant tokens
    assert assistant_mask[0] == False, "First token should be non-assistant"
    assert assistant_mask[-1] == True, "Last token should be assistant"

    # Visualize the token mask
    visualize_token_mask(
        token_ids, assistant_mask, tokenizer, "Minimal 2-message conversation"
    )

    print(f"\nSummary - Total tokens: {len(token_ids)}")
    assistant_tokens = sum(assistant_mask)
    non_assistant_tokens = len(assistant_mask) - assistant_tokens
    print(
        f"Token distribution: assistant={assistant_tokens}, non-assistant={non_assistant_tokens}"
    )


def test_get_token_ids_and_assistant_mask_complex(tokenizer):
    """Test with a more complex multi-turn conversation."""

    conversation: Conversation = [
        {"role": "system", "content": "You are an expert mathematician."},
        {"role": "user", "content": "Solve this equation: x^2 + 5x + 6 = 0"},
        {
            "role": "assistant",
            "content": "I'll solve this quadratic equation step by step.",
        },
        {"role": "user", "content": "Can you show the work?"},
        {
            "role": "assistant",
            "content": "Sure! Using the quadratic formula: x = (-5 ± √(25-24))/2 = (-5 ± 1)/2. So x = -2 or x = -3.",
        },
    ]

    token_ids, assistant_mask = get_token_ids_and_assistant_mask(
        conversation, tokenizer
    )

    # Visualize the token mask
    visualize_token_mask(
        token_ids, assistant_mask, tokenizer, "Complex multi-turn conversation"
    )

    print(f"\nSummary - Total tokens: {len(token_ids)}")

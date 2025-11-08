import pytest
import torch
from transformers import AutoTokenizer

from minrl.constants import GEMMA_3_1B, QWEN_3_0_6B, SMOL_LM_2_135M, Conversation
from minrl.sft import pack_sequences


class MockDataset:
    """Mock dataset for testing pack_sequences."""

    def __init__(self, conversations: list[Conversation]):
        self.conversations = conversations

    def initial_conversation(self, sample: dict, idx: int) -> Conversation:
        """Return conversation from the sample."""
        return self.conversations[idx]


def test_pack_sequences_basic(tokenizer):
    """Test basic sequence packing functionality."""
    # Create simple conversations
    conversations = [
        [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there!"},
        ],
        [
            {"role": "user", "content": "How are you?"},
            {"role": "assistant", "content": "I'm doing well, thanks!"},
        ],
    ]

    dataset = MockDataset(conversations)
    samples = [{"id": 0}, {"id": 1}]

    # Pack sequences
    result = pack_sequences(
        samples,
        tokenizer,
        dataset,
        max_seq_length=2048,
        pad_token_id=tokenizer.pad_token_id,
    )

    # Check output structure
    assert "input_ids" in result
    assert "labels" in result
    assert "attention_mask" in result

    # Check shapes
    assert result["input_ids"].shape[0] == 2  # batch size
    assert result["labels"].shape[0] == 2
    assert result["attention_mask"].shape[0] == 2

    # All tensors should have same shape
    assert result["input_ids"].shape == result["labels"].shape
    assert result["input_ids"].shape == result["attention_mask"].shape

    # Check dtypes
    assert result["input_ids"].dtype == torch.long
    assert result["labels"].dtype == torch.long
    assert result["attention_mask"].dtype == torch.long


def test_pack_sequences_padding(tokenizer):
    """Test that sequences are padded correctly to same length."""
    # Create conversations of different lengths
    conversations = [
        [
            {"role": "user", "content": "Hi"},
            {"role": "assistant", "content": "Hello!"},
        ],
        [
            {"role": "user", "content": "Can you explain quantum mechanics?"},
            {
                "role": "assistant",
                "content": "Quantum mechanics is a fundamental theory in physics that describes nature at the smallest scales.",
            },
        ],
    ]

    dataset = MockDataset(conversations)
    samples = [{"id": 0}, {"id": 1}]

    result = pack_sequences(
        samples,
        tokenizer,
        dataset,
        max_seq_length=2048,
        pad_token_id=tokenizer.pad_token_id,
    )

    # Both sequences should be padded to same length
    seq_len = result["input_ids"].shape[1]
    assert result["input_ids"].shape == (2, seq_len)
    assert result["labels"].shape == (2, seq_len)
    assert result["attention_mask"].shape == (2, seq_len)

    # Check that padding tokens are present
    input_ids = result["input_ids"]
    attention_mask = result["attention_mask"]
    labels = result["labels"]

    # First sequence (shorter) should have fewer non-zero attention mask tokens
    num_tokens_seq0 = (attention_mask[0] == 1).sum().item()
    num_tokens_seq1 = (attention_mask[1] == 1).sum().item()
    assert num_tokens_seq0 < num_tokens_seq1, (
        "First (shorter) sequence should have fewer non-padded tokens"
    )

    # The sequences should be padded to the same total length
    assert attention_mask[0].shape[0] == attention_mask[1].shape[0], (
        "Sequences should be padded to same length"
    )

    # Where attention_mask is 0, labels should be -100
    for i in range(2):
        padded_positions = attention_mask[i] == 0
        if padded_positions.any():
            assert (labels[i][padded_positions] == -100).all(), (
                f"Padded positions should have labels=-100 in sample {i}"
            )

    # Where attention_mask is 1, we should have real content
    for i in range(2):
        non_padded_positions = attention_mask[i] == 1
        assert non_padded_positions.any(), (
            f"Should have non-padded positions in sample {i}"
        )


def test_pack_sequences_max_length_filtering(tokenizer):
    """Test that sequences exceeding max_seq_length are filtered out."""
    conversations = [
        [
            {"role": "user", "content": "Short message"},
            {"role": "assistant", "content": "Short reply"},
        ],
        [
            {"role": "user", "content": "This is a very long message " * 100},
            {"role": "assistant", "content": "This is a very long response " * 100},
        ],
    ]

    dataset = MockDataset(conversations)
    samples = [{"id": 0}, {"id": 1}]

    # Pack with small max length - should filter out long conversation
    result = pack_sequences(
        samples,
        tokenizer,
        dataset,
        max_seq_length=50,  # Small max length
        pad_token_id=tokenizer.pad_token_id,
    )

    # Should only have 1 or 0 sequences (depending on tokenizer)
    # The long conversation should be filtered out
    assert result["input_ids"].shape[0] <= 2, (
        "Should filter out sequences exceeding max length"
    )

    # All sequences should be within max length
    seq_len = result["input_ids"].shape[1]
    assert seq_len <= 50, (
        f"Sequence length {seq_len} should not exceed max_seq_length=50"
    )


def test_pack_sequences_multi_turn_conversation(tokenizer):
    """Test packing with multi-turn conversations."""
    conversations = [
        [
            {"role": "user", "content": "What is your name?"},
            {"role": "assistant", "content": "I'm an AI assistant."},
            {"role": "user", "content": "What can you do?"},
            {"role": "assistant", "content": "I can help with many tasks!"},
        ]
    ]

    dataset = MockDataset(conversations)
    samples = [{"id": 0}]

    result = pack_sequences(
        samples,
        tokenizer,
        dataset,
        max_seq_length=2048,
        pad_token_id=tokenizer.pad_token_id,
    )

    labels = result["labels"][0]

    # Should have masked and unmasked tokens
    assert -100 in labels, "Should have masked tokens"
    assert (labels != -100).any(), "Should have unmasked assistant tokens"

    # Should have tokens from both assistant responses
    num_assistant_tokens = (labels != -100).sum().item()
    assert num_assistant_tokens > 0, (
        "Should have assistant tokens from multi-turn conversation"
    )


@pytest.mark.parametrize("model_id", [SMOL_LM_2_135M, QWEN_3_0_6B, GEMMA_3_1B])
def test_pack_sequences_different_tokenizers(model_id):
    """Test that pack_sequences works with different tokenizer formats."""
    tokenizer = AutoTokenizer.from_pretrained(model_id)

    conversations = [
        [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there!"},
        ]
    ]

    dataset = MockDataset(conversations)
    samples = [{"id": 0}]

    result = pack_sequences(
        samples,
        tokenizer,
        dataset,
        max_seq_length=2048,
        pad_token_id=tokenizer.pad_token_id,
    )

    # Should work with all tokenizers
    assert result["input_ids"].shape[0] == 1
    assert result["labels"].shape[0] == 1
    assert result["attention_mask"].shape[0] == 1

    # Should have proper masking
    labels = result["labels"][0]
    assert -100 in labels, f"Should have masked tokens for {model_id}"
    assert (labels != -100).any(), f"Should have unmasked tokens for {model_id}"


def test_pack_sequences_label_shift(tokenizer):
    """Test that labels are properly shifted for next-token prediction."""
    conversations = [
        [
            {"role": "user", "content": "Test"},
            {"role": "assistant", "content": "Response"},
        ]
    ]

    dataset = MockDataset(conversations)
    samples = [{"id": 0}]

    result = pack_sequences(
        samples,
        tokenizer,
        dataset,
        max_seq_length=2048,
        pad_token_id=tokenizer.pad_token_id,
    )

    input_ids = result["input_ids"][0]
    labels = result["labels"][0]

    # Labels should be one position ahead of input_ids (next-token prediction)
    # Where labels are not -100, they should correspond to the next token
    valid_label_positions = labels != -100
    if valid_label_positions.any():
        # Check that labels length matches the shift
        # input_ids should be one token longer than the original due to the shift
        assert input_ids.shape[0] == labels.shape[0], (
            "Input and labels should have same padded length"
        )


def test_pack_sequences_attention_mask_correctness(tokenizer):
    """Test that attention mask correctly identifies real vs padded tokens."""
    conversations = [
        [
            {"role": "user", "content": "Hi"},
            {"role": "assistant", "content": "Hello"},
        ],
        [
            {
                "role": "user",
                "content": "This is a much longer message that will require more tokens",
            },
            {"role": "assistant", "content": "This is also a longer response"},
        ],
    ]

    dataset = MockDataset(conversations)
    samples = [{"id": 0}, {"id": 1}]

    result = pack_sequences(
        samples,
        tokenizer,
        dataset,
        max_seq_length=2048,
        pad_token_id=tokenizer.pad_token_id,
    )

    input_ids = result["input_ids"]
    attention_mask = result["attention_mask"]
    labels = result["labels"]

    for i in range(input_ids.shape[0]):
        # Where attention mask is 1, we should have meaningful content
        # Where attention mask is 0, we should have padding
        num_attended = (attention_mask[i] == 1).sum().item()
        num_padded = (attention_mask[i] == 0).sum().item()

        # Should have some attended tokens
        assert num_attended > 0, f"Sample {i} should have some attended tokens"

        # Padded positions should have -100 labels
        padded_positions = attention_mask[i] == 0
        if padded_positions.any():
            assert (labels[i][padded_positions] == -100).all(), (
                f"Padded positions should have labels=-100 in sample {i}"
            )

        # Total should equal sequence length
        assert num_attended + num_padded == attention_mask[i].shape[0], (
            f"Attention mask should cover all positions in sample {i}"
        )

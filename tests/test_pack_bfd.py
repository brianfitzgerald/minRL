#!/usr/bin/env python3
"""Test script for the _pack_bfd function implementation."""

import torch
from minrl.sft import _pack_bfd


def test_pack_bfd_basic():
    """Test basic BFD packing functionality."""
    print("Testing basic BFD packing...")

    # Create sample data with different sequence lengths
    batch_size = 5
    max_len = 20
    seq_length = 50  # Packing target length

    # Create input_ids with varying lengths
    input_ids = torch.zeros(batch_size, max_len, dtype=torch.long)
    labels = torch.full((batch_size, max_len), -100, dtype=torch.long)

    # Fill with different sequence lengths
    lengths = [5, 8, 12, 3, 15]
    for i, length in enumerate(lengths):
        input_ids[i, :length] = torch.arange(1, length + 1)  # Non-zero tokens
        labels[i, :length] = torch.arange(1, length + 1)  # Valid labels

    examples = {"input_ids": input_ids, "labels": labels}

    # Test packing
    result = _pack_bfd(examples, seq_length)

    print(f"Input shape: {input_ids.shape}")
    print(f"Output input_ids shape: {result['input_ids'].shape}")
    print(f"Output labels shape: {result['labels'].shape}")
    print(f"Output attention_mask shape: {result['attention_mask'].shape}")
    print(f"Output seq_lengths shape: {result['seq_lengths'].shape}")

    # Verify output structure
    assert "input_ids" in result
    assert "labels" in result
    assert "attention_mask" in result
    assert "seq_lengths" in result

    # Verify shapes
    assert result["input_ids"].shape[1] == seq_length
    assert result["labels"].shape[1] == seq_length
    assert result["attention_mask"].shape[1] == seq_length
    assert result["input_ids"].shape[0] == result["labels"].shape[0]
    assert result["input_ids"].shape[0] == result["attention_mask"].shape[0]

    print("✓ Basic BFD packing test passed!")
    return result


def test_pack_bfd_efficiency():
    """Test that BFD packing is more efficient than simple padding."""
    print("\nTesting BFD packing efficiency...")

    # Create sequences that would benefit from packing
    batch_size = 6
    max_len = 30
    seq_length = 100  # Large packing target

    input_ids = torch.zeros(batch_size, max_len, dtype=torch.long)
    labels = torch.full((batch_size, max_len), -100, dtype=torch.long)

    # Create sequences of different lengths
    lengths = [10, 15, 20, 25, 30, 5]
    for i, length in enumerate(lengths):
        input_ids[i, :length] = torch.arange(1, length + 1)
        labels[i, :length] = torch.arange(1, length + 1)

    examples = {"input_ids": input_ids, "labels": labels}

    # Test BFD packing
    bfd_result = _pack_bfd(examples, seq_length)

    # Calculate efficiency
    total_input_length = sum(lengths)
    num_bins = bfd_result["input_ids"].shape[0]
    total_packed_length = num_bins * seq_length

    efficiency = (
        total_input_length / total_packed_length if total_packed_length > 0 else 0
    )

    print(f"Total input length: {total_input_length}")
    print(f"Number of bins: {num_bins}")
    print(f"Total packed length: {total_packed_length}")
    print(f"Packing efficiency: {efficiency:.2%}")

    # BFD should be more efficient than naive padding
    naive_bins = batch_size  # Each sequence in its own bin
    naive_efficiency = total_input_length / (naive_bins * seq_length)

    print(f"Naive padding efficiency: {naive_efficiency:.2%}")
    print(
        f"BFD improvement: {(efficiency - naive_efficiency) / naive_efficiency * 100:.1f}%"
    )

    assert efficiency > naive_efficiency, (
        "BFD should be more efficient than naive padding"
    )
    print("✓ BFD efficiency test passed!")


def test_pack_bfd_edge_cases():
    """Test edge cases for BFD packing."""
    print("\nTesting BFD packing edge cases...")

    # Test with empty input
    empty_examples = {
        "input_ids": torch.empty(0, 10, dtype=torch.long),
        "labels": torch.empty(0, 10, dtype=torch.long),
    }
    result = _pack_bfd(empty_examples, 50)
    assert result["input_ids"].shape[0] == 0
    print("✓ Empty input test passed!")

    # Test with all sequences too long
    long_examples = {
        "input_ids": torch.ones(3, 10, dtype=torch.long),  # All non-zero
        "labels": torch.ones(3, 10, dtype=torch.long),
    }
    result = _pack_bfd(long_examples, 5)  # seq_length < sequence length
    assert result["input_ids"].shape[0] == 0
    print("✓ All sequences too long test passed!")

    # Test with single sequence
    single_examples = {
        "input_ids": torch.tensor([[1, 2, 3, 0, 0]], dtype=torch.long),
        "labels": torch.tensor([[1, 2, 3, -100, -100]], dtype=torch.long),
    }
    result = _pack_bfd(single_examples, 50)
    assert result["input_ids"].shape[0] == 1
    assert result["input_ids"].shape[1] == 50
    print("✓ Single sequence test passed!")


def main():
    """Run all tests."""
    print("Running BFD packing tests...\n")

    try:
        test_pack_bfd_basic()
        test_pack_bfd_efficiency()
        test_pack_bfd_edge_cases()
        print("\n🎉 All tests passed!")

    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()

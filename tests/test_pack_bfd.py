#!/usr/bin/env python3
"""Test script for the _pack_bfd function implementation."""

import torch
from minrl.packing import pack_bfd


def _compute_efficiency(lengths, bfd_result, seq_length):
    total_input_length = sum(lengths)
    num_bins = bfd_result["input_ids"].shape[0]
    total_packed_length = num_bins * seq_length

    efficiency = (
        total_input_length / total_packed_length if total_packed_length > 0 else 0
    )
    return efficiency


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
    result = pack_bfd(examples, seq_length)

    efficiency = _compute_efficiency(lengths, result, seq_length)

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

    assert efficiency > 0.8

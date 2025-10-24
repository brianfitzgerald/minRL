from collections import defaultdict, deque
import torch
from typing import Any
from minrl.tasks.dataset import MinRLDataset
from minrl.algorithms import get_token_ids_and_assistant_mask


class _SegmentTree:
    """Segment tree for efficient space management in BFD packing."""

    def __init__(self, max_size: int):
        self.max_size = max_size
        self.tree = [0] * (4 * max_size)
        self.lazy = [0] * (4 * max_size)

    def _update_lazy(self, node: int, start: int, end: int):
        """Propagate lazy updates."""
        if self.lazy[node] != 0:
            self.tree[node] += self.lazy[node]
            if start != end:
                self.lazy[2 * node + 1] += self.lazy[node]
                self.lazy[2 * node + 2] += self.lazy[node]
            self.lazy[node] = 0

    def _update_range(
        self, node: int, start: int, end: int, left: int, right: int, val: int
    ):
        """Update range [left, right] with value val."""
        self._update_lazy(node, start, end)

        if start > end or start > right or end < left:
            return

        if start >= left and end <= right:
            self.tree[node] += val
            if start != end:
                self.lazy[2 * node + 1] += val
                self.lazy[2 * node + 2] += val
            return

        mid = (start + end) // 2
        self._update_range(2 * node + 1, start, mid, left, right, val)
        self._update_range(2 * node + 2, mid + 1, end, left, right, val)
        self._update_lazy(2 * node + 1, start, mid)
        self._update_lazy(2 * node + 2, mid + 1, end)
        self.tree[node] = max(self.tree[2 * node + 1], self.tree[2 * node + 2])

    def _query_max(self, node: int, start: int, end: int, left: int, right: int) -> int:
        """Query maximum in range [left, right]."""
        self._update_lazy(node, start, end)

        if start > end or start > right or end < left:
            return 0

        if start >= left and end <= right:
            return self.tree[node]

        mid = (start + end) // 2
        return max(
            self._query_max(2 * node + 1, start, mid, left, right),
            self._query_max(2 * node + 2, mid + 1, end, left, right),
        )

    def add(self, size: int):
        """Add available space of given size."""
        if 0 < size <= self.max_size:
            self._update_range(0, 0, self.max_size, size, size, 1)

    def remove(self, size: int):
        """Remove available space of given size."""
        if 0 < size <= self.max_size:
            self._update_range(0, 0, self.max_size, size, size, -1)

    def search(self, required_size: int) -> int:
        """Find the smallest available space >= required_size."""
        if required_size > self.max_size:
            return self.max_size

        # Binary search for the smallest available space
        left, right = required_size, self.max_size
        result = self.max_size

        while left <= right:
            mid = (left + right) // 2
            if self._query_max(0, 0, self.max_size, required_size, mid) > 0:
                result = mid
                right = mid - 1
            else:
                left = mid + 1

        return result


def _pack_bfd(examples: dict, seq_length: int) -> dict:
    """Pack sequences using Best Fit Decreasing strategy.

    Args:
        examples: Dictionary containing 'input_ids' and 'labels' tensors
        seq_length: Maximum sequence length for packing

    Returns:
        Dictionary with packed sequences and metadata
    """
    input_ids = examples["input_ids"]
    labels = examples["labels"]

    # Get sequence lengths (excluding padding)
    attention_mask = (input_ids != 0).long()  # Assuming 0 is pad token
    seq_lengths = attention_mask.sum(dim=1)

    # Filter out sequences that are too long
    valid_mask = seq_lengths <= seq_length
    if not valid_mask.any():
        # All sequences too long, return empty
        return {
            "input_ids": torch.empty(0, seq_length, dtype=torch.long),
            "labels": torch.empty(0, seq_length, dtype=torch.long),
            "attention_mask": torch.empty(0, seq_length, dtype=torch.long),
            "seq_lengths": torch.empty(0, dtype=torch.long),
        }

    valid_input_ids = input_ids[valid_mask]
    valid_labels = labels[valid_mask]
    valid_lengths = seq_lengths[valid_mask]

    # Sort by length (descending)
    sorted_indices = torch.argsort(valid_lengths, descending=True)

    # Initialize segment tree and bin management
    segment_tree = _SegmentTree(seq_length)
    segment_tree.add(seq_length)  # Max bin is always available
    space_to_bin = defaultdict(deque)

    # Bins represented as dicts with ids and total length
    bins = []

    # Process sequences in descending order of length
    for idx in sorted_indices:
        length = valid_lengths[idx].item()

        # Find best fit space
        space = segment_tree.search(length)

        if space < seq_length:
            # Use existing bin
            bin_dict = space_to_bin[space].popleft()
        else:
            # Create new bin
            bin_dict = {"ids": [], "length": 0}
            bins.append(bin_dict)

        # Add sequence to bin
        bin_dict["ids"].append(idx.item())
        bin_dict["length"] += length

        # Update segment tree
        if space < seq_length and not space_to_bin[space]:
            segment_tree.remove(space)

        # Update remaining space
        remaining_space = space - length
        space_to_bin[remaining_space].append(bin_dict)
        if remaining_space > 0:
            segment_tree.add(remaining_space)

    # Pack sequences according to bin order
    packed_input_ids = []
    packed_labels = []
    packed_attention_masks = []
    packed_seq_lengths = []

    for bin_dict in bins:
        bin_input_ids = valid_input_ids[bin_dict["ids"]]
        bin_labels = valid_labels[bin_dict["ids"]]
        bin_lengths = valid_lengths[bin_dict["ids"]]

        # Concatenate sequences in this bin
        concatenated_input_ids = torch.cat(
            [bin_input_ids[i][: bin_lengths[i]] for i in range(len(bin_input_ids))]
        )
        concatenated_labels = torch.cat(
            [bin_labels[i][: bin_lengths[i]] for i in range(len(bin_labels))]
        )

        # Pad to seq_length
        pad_length = seq_length - len(concatenated_input_ids)
        if pad_length > 0:
            pad_input_ids = torch.zeros(pad_length, dtype=torch.long)
            pad_labels = torch.full((pad_length,), -100, dtype=torch.long)
            concatenated_input_ids = torch.cat([concatenated_input_ids, pad_input_ids])
            concatenated_labels = torch.cat([concatenated_labels, pad_labels])

        # Create attention mask
        attention_mask = torch.cat(
            [
                torch.ones(len(concatenated_input_ids) - pad_length, dtype=torch.long),
                torch.zeros(pad_length, dtype=torch.long),
            ]
        )

        packed_input_ids.append(concatenated_input_ids)
        packed_labels.append(concatenated_labels)
        packed_attention_masks.append(attention_mask)
        packed_seq_lengths.append(torch.tensor(bin_dict["length"], dtype=torch.long))

    if not packed_input_ids:
        # No valid sequences
        return {
            "input_ids": torch.empty(0, seq_length, dtype=torch.long),
            "labels": torch.empty(0, seq_length, dtype=torch.long),
            "attention_mask": torch.empty(0, seq_length, dtype=torch.long),
            "seq_lengths": torch.empty(0, dtype=torch.long),
        }

    return {
        "input_ids": torch.stack(packed_input_ids),
        "labels": torch.stack(packed_labels),
        "attention_mask": torch.stack(packed_attention_masks),
        "seq_lengths": torch.stack(packed_seq_lengths),
    }


def pack_sequences(
    samples: list[Any],
    tokenizer: Any,
    dataset: MinRLDataset,
    max_seq_length: int = 2048,
    pad_token_id: int = 0,
) -> dict[str, torch.Tensor]:
    all_input_ids = []
    all_labels = []

    for i, sample in enumerate(samples):
        # Get the full conversation including ground truth
        conversation = dataset.initial_conversation(sample, i)

        # Get token IDs and mask for assistant responses
        token_ids, assistant_mask = get_token_ids_and_assistant_mask(
            conversation, tokenizer
        )

        # Create labels: -100 for non-assistant tokens (ignored in loss)
        # Shift left by 1 for next-token prediction
        labels = []
        for j in range(len(token_ids) - 1):
            if assistant_mask[j + 1]:  # Next token is assistant token
                labels.append(token_ids[j + 1])
            else:
                labels.append(-100)

        # Input is all tokens except the last
        input_ids = token_ids[:-1]

        # Add to lists if not too long
        if len(input_ids) <= max_seq_length:
            all_input_ids.append(input_ids)
            all_labels.append(labels)

    # Find max length in this batch
    max_len = max(len(ids) for ids in all_input_ids) if all_input_ids else 1

    # Pad sequences
    padded_input_ids = []
    padded_labels = []
    attention_masks = []

    for input_ids, labels in zip(all_input_ids, all_labels):
        # Pad input_ids and labels
        pad_len = max_len - len(input_ids)

        padded_input_ids.append(input_ids + [pad_token_id] * pad_len)
        padded_labels.append(labels + [-100] * pad_len)
        attention_masks.append([1] * len(input_ids) + [0] * pad_len)

    return {
        "input_ids": torch.tensor(padded_input_ids, dtype=torch.long),
        "labels": torch.tensor(padded_labels, dtype=torch.long),
        "attention_mask": torch.tensor(attention_masks, dtype=torch.long),
    }

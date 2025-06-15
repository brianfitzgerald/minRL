from typing import Any, List, Dict, Tuple, Union, TypedDict
from transformers.tokenization_utils_base import PreTrainedTokenizerBase
import random
from minrl.tasks.dataset import MinRLDataset, MiniBatch, Split

SYSTEM_PROMPT = """
You are a helpful assistant. Solve this puzzle for me.
There are three pegs and n disks of different sizes stacked on the first peg. The disks are
numbered from 1 (smallest) to n (largest). Disk moves in this puzzle should follow:
1. Only one disk can be moved at a time.
2. Each move consists of taking the upper disk from one stack and placing it on top of
another stack.
3. A larger disk may not be placed on top of a smaller disk.
The goal is to move the entire stack to the third peg.
Example: With 3 disks numbered 1 (smallest), 2, and 3 (largest), the initial state is [[3, 2, 1],
[], []], and a solution might be:
moves = [[1 , 0 , 2] , [2 , 0 , 1] , [1 , 2 , 1] , [3 , 0 , 2] ,
[1 , 1 , 0] , [2 , 1 , 2] , [1 , 0 , 2]]
This means: Move disk 1 from peg 0 to peg 2, then move disk 2 from peg 0 to peg 1, and so on.
Requirements:
• When exploring potential solutions in your thinking process, always include the corresponding complete list of moves.
• The positions are 0-indexed (the leftmost peg is 0).
• Ensure your final answer includes the complete list of moves in the format:
moves = [[disk id, from peg, to peg], ...]
"""


def user_prompt(num_disks: int) -> str:
    return f"""
I have a puzzle with {num_disks} disks of different sizes with
Initial configuration:
• Peg 0: {num_disks} (bottom), . . . 2, 1 (top)
• Peg 1: (empty)
• Peg 2: (empty)
17
Goal configuration:
• Peg 0: (empty)
• Peg 1: (empty)
• Peg 2: {num_disks} (bottom), . . . 2, 1 (top)
Rules:
• Only one disk can be moved at a time.
• Only the top disk from any stack can be moved.
• A larger disk may not be placed on top of a smaller disk.
Find the sequence of moves to transform the initial configuration into the goal configuration.
"""


class TowerOfHanoi:
    def __init__(self, num_disks: int = 3) -> None:
        if num_disks < 1:
            raise ValueError("Number of disks must be at least 1")

        self.num_disks: int = num_disks
        self.moves_count: int = 0
        self.stacks: List[List[int]] = []
        self.reset()

    def reset(self) -> None:
        self.stacks = [
            list(range(self.num_disks, 0, -1)),
            [],
            [],
        ]
        self.moves_count = 0

    def get_state(self) -> Dict[str, Union[List[List[int]], int]]:
        return {
            "stacks": [stack.copy() for stack in self.stacks],
            "moves_count": self.moves_count,
        }

    def is_valid_move(self, from_stack: int, to_stack: int) -> Tuple[bool, str]:
        if from_stack not in [0, 1, 2]:
            return False, f"Invalid source stack: {from_stack}. Must be 0, 1, or 2."

        if to_stack not in [0, 1, 2]:
            return False, f"Invalid destination stack: {to_stack}. Must be 0, 1, or 2."

        if from_stack == to_stack:
            return False, "Cannot move disk to the same stack."

        if not self.stacks[from_stack]:
            return False, f"Stack {from_stack} is empty."

        if self.stacks[to_stack]:
            top_from = self.stacks[from_stack][-1]
            top_to = self.stacks[to_stack][-1]
            if top_from > top_to:
                return False, f"Cannot place disk {top_from} on smaller disk {top_to}."

        return True, ""

    def make_move(self, from_stack: int, to_stack: int) -> bool:
        is_valid, error_msg = self.is_valid_move(from_stack, to_stack)

        if not is_valid:
            return False

        disk = self.stacks[from_stack].pop()
        self.stacks[to_stack].append(disk)
        self.moves_count += 1

        return True

    def is_solved(self) -> bool:
        return len(self.stacks[2]) == self.num_disks and self.stacks[2] == list(
            range(self.num_disks, 0, -1)
        )

    def get_minimum_moves(self) -> int:
        return (2**self.num_disks) - 1


class HanoiSampleTokenized(TypedDict):
    prefix: str
    prefix_token_ids: list[int]


class HanoiSample(TypedDict):
    n_disks: int


def tokenize_hanoi_sample(
    sample: HanoiSample, tokenizer: PreTrainedTokenizerBase
) -> HanoiSampleTokenized:
    prefix: str = tokenizer.apply_chat_template(  # type: ignore
        [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt(sample["n_disks"])},
        ],
    )
    return {
        "prefix": prefix,
        "prefix_token_ids": tokenizer.encode(prefix),
    }


class HanoiDataset(MinRLDataset):
    def __init__(
        self,
        split: Split,
        tokenizer: PreTrainedTokenizerBase | None = None,
    ):
        super().__init__(split, tokenizer)
        self.tokenizer = tokenizer

    def __getitem__(self, _: int) -> HanoiSample:
        n_disks = random.randint(1, 10)
        return {"n_disks": n_disks}

    def collate_fn(self, batch: List[HanoiSample]) -> MiniBatch:
        """
        Collate examples into a batch.
        Used during training / only, requires a tokenizer.
        """
        if self.tokenizer is None:
            raise ValueError("Tokenizer is not set")
        tokenized = [tokenize_hanoi_sample(item, self.tokenizer) for item in batch]
        prefixes = [item["prefix"] for item in tokenized]
        prefix_token_ids = [item["prefix_token_ids"] for item in tokenized]
        return MiniBatch(
            prefixes=prefixes,
            prefix_token_ids=prefix_token_ids,
            samples=batch,
        )


def hanoi_reward_func(response: str, sample: dict[str, Any]) -> float:
    # TODO: Implement reward function
    return 1.0

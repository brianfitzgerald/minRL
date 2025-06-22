from typing import Any, List, Dict, Tuple, Union, TypedDict
from transformers.tokenization_utils_base import PreTrainedTokenizerBase
import random
from minrl.tasks.dataset import MinRLDataset, MiniBatch, Split
from loguru import logger
import re
import ast
import numpy as np
from minrl.constants import HostType

SYSTEM_PROMPT = """
You are a helpful assistant. Solve this puzzle for me.
There are three pegs and n disks of different sizes stacked on the first peg. The disks are
numbered from 1 (smallest) to n (largest). Disk moves in this puzzle should follow:
1. Only one disk can be moved at a time.
2. Each move consists of taking the upper disk from one stack and placing it on top of
another stack.
3. A larger disk may not be placed on top of a smaller disk.
The goal is to move the entire stack to the third peg.
Requirements:
• When exploring potential solutions in your thinking process, always include the corresponding complete list of moves.
• The positions are 0-indexed (the leftmost peg is 0).
• Ensure your final answer includes the complete list of moves in the format:
moves = [[disk id, from peg, to peg], ...]

DO NOT write any code. Just return the list of moves.
Always format your response within <result> tags.

### Example 
Initial state: <state>[[3, 2, 1], [], []]</state>
Response: <result>[[1, 0, 2], [2, 0, 1], [1, 2, 1], [3, 0, 2], [1, 1, 0], [2, 1, 2], [1, 0, 2]]</result>
This means: Move disk 1 from peg 0 to peg 2, then move disk 2 from peg 0 to peg 1, and so on.
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


def create_hanoi_state(n: int) -> list[list[int]]:
    return [
        list(range(n, 0, -1)),
        [],
        [],
    ]


def make_distributions(n: int) -> tuple[list[float], list[float]]:
    x = np.linspace(0, 1, n)
    w1 = 1 - x
    w2 = x
    return (w1 / w1.sum()).tolist(), (w2 / w2.sum()).tolist()


def blend(w1: list[float], w2: list[float], alpha: float) -> list[float]:
    arr = (1 - alpha) * np.array(w1) + alpha * np.array(w2)
    return (arr / arr.sum()).tolist()


class HanoiDataset(MinRLDataset):
    def __init__(
        self,
        split: Split,
        host: HostType,
        tokenizer: PreTrainedTokenizerBase | None = None,
    ):
        super().__init__(split, host, tokenizer)
        self.tokenizer = tokenizer
        self.n_samples = 10**3
        self.seed = 42
        random.seed(self.seed)
        self.interpolate_weights = False
        self.n_disks_range = range(3, 15)

    def __getitem__(self, i: int) -> HanoiSample:
        if self.interpolate_weights:
            w1, w2 = make_distributions(9)
            w = blend(w1, w2, i / self.n_samples)
        else:
            w = [1.0] * len(self.n_disks_range)
        n_disks = random.choices(self.n_disks_range, k=1, weights=w)[0]
        return {"n_disks": n_disks}

    def __len__(self) -> int:
        # mock value to satisfy dataloader
        # 10k samples
        return self.n_samples

    def conversation(self, sample: HanoiSample) -> list[dict[str, Any]]:
        return [
            {
                "role": "system",
                "content": SYSTEM_PROMPT,
            },
            {
                "role": "user",
                "content": f"Initial state: {create_hanoi_state(sample['n_disks'])}",
            },
        ]

    def collate_fn(self, batch: List[HanoiSample]) -> MiniBatch:
        """
        Collate examples into a batch.
        Used during training only, requires a tokenizer.
        """
        if self.tokenizer is None:
            raise ValueError("Tokenizer is not set")
        prefixes = []
        prefix_token_ids = []
        for item in batch:
            prefix: str = self.tokenizer.apply_chat_template(  # type: ignore
                self.conversation(item),
                tokenize=False,
                enable_thinking=False,
            )
            prefixes.append(prefix)
            prefix_token_ids.append(self.tokenizer.encode(prefix))

        return MiniBatch(
            prefixes=prefixes,
            prefix_token_ids=prefix_token_ids,
            samples=batch,
        )


def extract_result_list(s: str) -> list[list[int]]:
    m = re.search(r"<result>(.*?)</result>", s, re.DOTALL)
    if not m:
        return []
    return ast.literal_eval(m.group(1))


def hanoi_reward_func(response: str, sample: dict[str, Any]) -> float:
    try:
        game = TowerOfHanoi(sample["n_disks"])
        moves = extract_result_list(response)
        n_valid_moves, required_moves, solved = 0, game.get_minimum_moves(), False
        for i, move in enumerate(moves):
            from_stack, to_stack = move[1], move[2]
            valid = game.make_move(from_stack, to_stack)
            if valid:
                n_valid_moves += 1
            if not valid:
                break
            if game.is_solved():
                solved = True
                break
            if i > required_moves:
                return -1.0
        if not solved and n_valid_moves > 0:
            return round(n_valid_moves / required_moves, 2)
        if solved:
            return 1.0
        else:
            return 0.0
    except Exception as e:
        logger.error(f"Error in hanoi_reward_func: {e}")
        return 0.0
    return 0.0

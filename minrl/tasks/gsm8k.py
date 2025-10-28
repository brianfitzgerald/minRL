from transformers.tokenization_utils_base import PreTrainedTokenizerBase
from minrl.tasks.dataset import MinRLDataset, Split
import re
from minrl.constants import Conversation, HostType, Sample
from datasets import load_dataset

from minrl.utils import log_conversation

_SOLUTION_CLIP_CHARS = 300

TEMPLATE = """
You are a helpful assistant. Reason about an answer to the following question and provide the final answer after ####.
Question: {question}
"""


# https://github.com/volcengine/verl/blob/main/verl/utils/reward_score/gsm8k.py#L20
def extract_solution(solution_str, method="strict"):
    assert method in ["strict", "flexible"]

    # Optimization: Regular expression matching on very long strings can be slow.
    # For math problems, the final answer is usually at the end.
    # We only match on the last 300 characters, which is a safe approximation for 300 tokens.
    if len(solution_str) > _SOLUTION_CLIP_CHARS:
        solution_str = solution_str[-_SOLUTION_CLIP_CHARS:]

    if method == "strict":
        # this also tests the formatting of the model
        solutions = re.findall("#### (\\-?[0-9\\.\\,]+)", solution_str)
        if len(solutions) == 0:
            final_answer = None
        else:
            # take the last solution
            final_answer = solutions[-1].replace(",", "").replace("$", "")
    elif method == "flexible":
        answer = re.findall("(\\-?[0-9\\.\\,]+)", solution_str)
        final_answer = None
        if len(answer) == 0:
            # no reward is there is no answer
            pass
        else:
            invalid_str = ["", "."]
            # find the last number that is not '.'
            for final_answer in reversed(answer):
                if final_answer not in invalid_str:
                    break
    return final_answer


def compute_score(
    solution_str: str,
    ground_truth: str,
    method: str = "flexible",
    format_score: float = 0.1,
    score: float = 1.0,
) -> float:
    answer = extract_solution(solution_str=solution_str, method=method)
    if answer is None:
        return 0
    else:
        if answer == ground_truth:
            return score
        else:
            return format_score


class GSM8KDataset(MinRLDataset):
    def __init__(
        self,
        split: Split,
        host: HostType,
        tokenizer: PreTrainedTokenizerBase | None = None,
    ):
        super().__init__(split, host, tokenizer)
        self.tokenizer = tokenizer
        split_name: str = "test" if self.split == "eval" else self.split
        self.dataset = load_dataset("openai/gsm8k", "main", split=split_name)
        self.iter = iter(self.dataset)

    def __getitem__(self, i: int) -> Sample:
        return next(self.iter)

    def __len__(self) -> int:
        # mock value to satisfy dataloader
        # 10k samples
        return len(self.dataset)  # type: ignore

    def initial_conversation(self, sample: Sample, sample_index: int) -> Conversation:
        return [
            {
                "role": "user",
                "content": TEMPLATE.format(question=sample["question"]),
            },
        ]

    @staticmethod
    def reward_function(conversation: Conversation, sample: Sample) -> float:
        answer = conversation[-1]["content"]
        ground_truth = extract_solution(sample["answer"], method="strict")
        if ground_truth is None:
            return 0.0
        return float(compute_score(answer, ground_truth))

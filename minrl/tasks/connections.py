import itertools
import math
import re
from typing import Any, List, TypedDict

import pandas as pd
from loguru import logger
from sklearn.model_selection import train_test_split
from transformers.tokenization_utils_base import PreTrainedTokenizerBase

from minrl.constants import Conversation, HostType, Sample
from minrl.tasks.dataset import MinRLDataset, Split

SYSTEM_MESSAGE = (
    "You are a helpful assistant. You first think about the reasoning process "
    "in your mind and then provide the user with the answer."
)
RESPONSE_PROMPT = "Let me solve this step by step.\n<think>"

N_CONNECTION_GROUPS = 4
CONNECTION_GROUP_SIZE = 4

CONNECTIONS_PROMPT = """
Find four groups of four words that are related to each other.
You may only use each word in one group.
Always return four groups of four words.
Respond in the following format:
<answer>
<group>
...
</group>
<group>
...
</group>
</answer>

"""

MOCK_PROMPT = """candle, crayon, honeycomb, seal, kettles, mittens, raindrops, whiskers, canine, fang, molar, tusk, defense, excuse, out, reason"""

MOCK_ANSWER = """
<answer>
<group>candle, crayon, honeycomb, seal</group>
<group>kettles, mittens, raindrops, whiskers</group>
<group>canine, fang, molar, tusk</group>
<group>defense, excuse, out, reason</group>
</answer>
"""


class ConnectionsTrainSample(TypedDict):
    words: list[str]
    groups: list[dict[str, list[str]]]


class ConnectionsSample(TypedDict):
    answer_groups: list[list[str]]
    prompt: str
    answer: str


class ConnectionsSampleTokenized(TypedDict):
    prefix: str
    prefix_token_ids: list[int]
    answer: str
    answer_groups: list[list[str]]


class ConnectionsEvalAnswer(TypedDict):
    level: int
    group: str
    members: List[str]


class ConnectionsEvalSample(TypedDict):
    id: int
    date: str
    answers: List[ConnectionsEvalAnswer]


def _map_generated_sample(sample: ConnectionsTrainSample) -> ConnectionsSample:
    words = sample["words"]
    words_formatted = ", ".join(words)
    answer = []
    answer_groups = []
    for group in sample["groups"]:
        answer.append(", ".join(group["words"]))
        answer_groups.append(group["words"])
    answer_formatted = "\n".join([f"<group>{a}</group>" for a in answer])
    return {
        "prompt": words_formatted,
        "answer": f"<answer>{answer_formatted}</answer>",
        "answer_groups": answer_groups,
    }


def _map_canonical_sample(sample: ConnectionsEvalSample) -> ConnectionsSample:
    """Eval samples are from a different dataset, so we need to map them to the same format as train samples."""
    prompt = []
    answer_groups = []
    for answer in sample["answers"]:
        prompt.append(", ".join(answer["members"]))
        answer_groups.append(answer["members"])
    answer_formatted = "\n".join([f"<group>{a}</group>" for a in prompt])
    return {
        "prompt": ", ".join(prompt),
        "answer": f"<answer>{answer_formatted}</answer>",
        "answer_groups": answer_groups,
    }


class ConnectionsDataset(MinRLDataset):
    def __init__(
        self,
        split: Split,
        host: HostType,
        tokenizer: PreTrainedTokenizerBase | None = None,
    ):
        super().__init__(split, host, tokenizer)
        dataset_dir = "/data/" if host == "modal" else "data/"
        generated_prompts_path = f"{dataset_dir}/connections_generated.jsonl"
        logger.info(f"Loading generated prompts from {generated_prompts_path}")
        prompts_pd = pd.read_json(generated_prompts_path, lines=True)
        df_groups = pd.json_normalize(prompts_pd["solution"], "groups")  # type: ignore
        self.tokenizer = tokenizer
        num_samples = 1000

        logger.info(f"Generating {num_samples} samples")

        groups = [
            {
                "groups": (
                    g := df_groups.sample(4, replace=False).reset_index(drop=True)
                ).to_dict(orient="records"),
                "words": list(itertools.chain.from_iterable(g["words"].dropna())),
            }
            for _ in range(num_samples)
        ]

        samples_generated = pd.DataFrame(groups)
        samples_generated = samples_generated.apply(_map_generated_sample, axis=1)
        canonical_prompts_path = f"{dataset_dir}/connections_canonical.json"
        logger.info(f"Loading canonical prompts from {canonical_prompts_path}")
        samples_canonical = pd.read_json(canonical_prompts_path)
        samples_canonical = samples_canonical.apply(_map_canonical_sample, axis=1)

        # concat both datasets
        samples_generated = pd.concat([samples_generated, samples_canonical])

        # Only use generated samples for now
        train_data, val_data = train_test_split(  # type: ignore
            samples_generated, test_size=0.1, random_state=42
        )
        self.dataframe: pd.DataFrame = train_data if split == "train" else val_data  # pyright: ignore[reportAttributeAccessIssue]

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        item = self.dataframe.iloc[idx]
        return item  # type: ignore

    def initial_conversation(self, sample: Sample, _: int) -> Conversation:
        return [
            {
                "role": "system",
                "content": CONNECTIONS_PROMPT,
            },
            {"role": "user", "content": MOCK_PROMPT},
            {"role": "assistant", "content": MOCK_ANSWER},
            {"role": "user", "content": f"{sample['prompt']}"},
        ]

    @staticmethod
    def reward_function(conversation: Conversation, sample: dict[str, Any]) -> float:
        answer_str = conversation[-1]["content"]
        groups = parse_groups(answer_str)
        format_score = strict_format_reward_func(answer_str)
        hard_score = score_connections_hard(sample["answer_groups"], groups)
        soft_score = score_connections_soft(sample["answer_groups"], groups)
        score = (hard_score + soft_score) / 2 + format_score
        if math.isnan(score):
            return 0.0
        return score


def strict_format_reward_func(response: str) -> float:
    """Reward function that checks if the completion has the right format, with strict spacing."""
    pattern = r"^<reasoning>.*?</reasoning>\s*<answer>.*?</answer>\s*$"
    match = re.match(pattern, response, flags=re.DOTALL)
    return 0.25 if match else 0.0


def parse_groups(input_string: str) -> list[list[str]]:
    # Find all occurrences of text within <group>...</group>
    group_contents = re.findall(r"<group>(.*?)</group>", input_string, re.DOTALL)

    groups = []
    for content in group_contents:
        # Split on commas and trim each word
        words = [word.strip() for word in content.split(",") if word.strip()]
        groups.append(words)

    return groups


def score_connections_soft(
    solution_groups: list[list[str]], submitted_groups: list[list[str]]
) -> float:
    """Return the best match count for each solution group."""
    solution_sets = [set(group) for group in solution_groups]
    submitted_sets = [
        set(group) for group in submitted_groups if len(group) == CONNECTION_GROUP_SIZE
    ]

    if len(submitted_sets) > N_CONNECTION_GROUPS:
        return 0.0

    if len(submitted_groups) == 0 or len(solution_groups) == 0:
        return 0.0

    # Get highest match count for each solution group
    best_match_counts = []
    if submitted_sets:
        for sol_set in solution_sets:
            if submitted_sets:
                best_match_counts.append(
                    max(
                        len(sol_set.intersection(submitted))
                        for submitted in submitted_sets
                    )
                )
            else:
                best_match_counts.append(0)
    else:
        best_match_counts = [0] * len(solution_sets)
    return float(sum(best_match_counts) / len(solution_groups)) / len(submitted_groups)


def score_connections_hard(
    solution_groups: list[list[str]], submitted_groups: list[list[str]]
) -> float:
    """Return the number of correct groups."""
    hard_score = 0
    correct_group_indices = []  # Track indices of correctly solved solution groups.

    solution_set = [set(group) for group in solution_groups]
    solved = set()

    if len(submitted_groups) > N_CONNECTION_GROUPS:
        return 0.0

    for submitted_group in submitted_groups:
        for i, correct_set in enumerate(solution_set):
            if set(submitted_group) == correct_set and i not in solved:
                hard_score += 1
                correct_group_indices.append(i)
                solved.add(i)
                break

    if len(submitted_groups) == 0:
        return 0.0
    return float(hard_score) / len(submitted_groups)

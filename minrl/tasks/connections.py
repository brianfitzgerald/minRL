import itertools
import re
from typing import Any, Dict, List, TypedDict

import pandas as pd
from loguru import logger
from sklearn.model_selection import train_test_split
from transformers.tokenization_utils_base import PreTrainedTokenizerBase
from minrl.tasks.dataset import MinRLDataset, MiniBatch, Split
from minrl.constants import HostType

SYSTEM_MESSAGE = (
    "You are a helpful assistant. You first think about the reasoning process "
    "in your mind and then provide the user with the answer."
)
RESPONSE_PROMPT = "Let me solve this step by step.\n<think>"

N_GROUPS = 4
GROUP_SIZE = 4


CONNECTIONS_PROMPT = """
You are an expert puzzle solving model.
Find groups of words that are related to each other. Each group is four words long. There are exactly four groups in total.
You may only use each word in one group.
Respond in the following format:
<think>
...
</think>
<answer>
<group>
...
</group>
<group>
...
</group>
</answer>
# Example

User: candle, crayon, honeycomb, seal, defense, excuse, out, reason, kettles, mittens, raindrops, whiskers, canine, fang, molar, tusk
Assistant: <think>
For Group 1, the words related to wax: candle, crayon, honeycomb, seal — connecting to wax in various forms, such as crayon and candles being wax-based.
For Group 2, "My Favorite Things" connects to lyrics mentioning kettles, mittens, raindrops, whiskers.
Group 3 relates to teeth, covering canine, fang, molar, tusk. Group 4 involves "no" related phrases like "no excuse," "no defense."
</think>
<answer>
<group> candle, crayon, honeycomb, seal</group>
<group> kettles, mittens, raindrops, whiskers</group>
<group> canine, fang, molar, tusk</group>
<group> defense, excuse, out, reason</group>
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
        generated_prompts_path = (
            "/data/connections_generated.jsonl"
            if host == "modal"
            else "data/connections_generated.jsonl"
        )
        logger.info(f"Loading generated prompts from {generated_prompts_path}")
        prompts_pd = pd.read_json(generated_prompts_path, lines=True)
        df_groups = pd.json_normalize(prompts_pd["solution"], "groups")  # type: ignore
        self.tokenizer = tokenizer
        num_samples = 10000

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
        canonical_prompts_path = (
            "/data/connections_canonical.json"
            if host == "modal"
            else "data/connections_canonical.json"
        )
        logger.info(f"Loading canonical prompts from {canonical_prompts_path}")
        samples_canonical = pd.read_json(canonical_prompts_path)
        samples_canonical = samples_canonical.apply(_map_canonical_sample, axis=1)
        train_data, val_data = train_test_split(
            samples_generated, test_size=0.1, random_state=42
        )
        self.dataframe: pd.DataFrame = train_data if split == "train" else val_data

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        item = self.dataframe.iloc[idx]
        return item  # type: ignore

    def conversation(self, sample: dict[str, Any]) -> List[dict[str, Any]]:
        return [
            {
                "role": "system",
                "content": CONNECTIONS_PROMPT,
            },
            {"role": "user", "content": sample["prompt"]},
        ]

    def collate_fn(self, batch: List[ConnectionsSample]) -> MiniBatch:
        """
        Collate examples into a batch.
        Used during training / only, requires a tokenizer.
        """
        if self.tokenizer is None:
            raise ValueError("Tokenizer is not set")
        prefixes, prefix_token_ids = [], []
        for sample in batch:
            prefix: str = self.tokenizer.apply_chat_template(
                self.conversation(sample),  # type: ignore
                tokenize=False,
                enable_thinking=False,
            )  # type: ignore
            tokens = self.tokenizer.encode(prefix)
            prefixes.append(prefix)
            prefix_token_ids.append(tokens)
        return MiniBatch(
            prefixes=prefixes,
            prefix_token_ids=prefix_token_ids,
            samples=batch,
        )


def strict_format_reward_func(
    response: str, samples: dict[str, Any]
) -> Dict[str, float]:
    """Reward function that checks if the completion has the right format, with strict spacing."""
    pattern = r"^<reasoning>.*?</reasoning>\s*<answer>.*?</answer>\s*$"
    match = re.match(pattern, response, flags=re.DOTALL)
    return {"reward": 0.25 if match else 0.0}


def parse_groups(input_string) -> list[list[str]]:
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
        set(group) for group in submitted_groups if len(group) == GROUP_SIZE
    ]

    if len(submitted_sets) > N_GROUPS:
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

    if len(submitted_groups) > N_GROUPS:
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


def connections_reward_func(response: str, sample: dict[str, Any]) -> float:
    """Reward the number of correct groups."""
    groups = parse_groups(response)
    hard_score = score_connections_hard(sample["answer_groups"], groups)
    soft_score = score_connections_soft(sample["answer_groups"], groups)
    return hard_score + soft_score

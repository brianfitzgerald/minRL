import itertools
import re
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional
from torch.utils.data import Dataset

import pandas as pd
from loguru import logger
from sklearn.model_selection import train_test_split
from tokenizers import Tokenizer

from data_types import MiniBatch

SYSTEM_MESSAGE = (
    "You are a helpful assistant. You first think about the reasoning process "
    "in your mind and then provide the user with the answer."
)
USER_TEMPLATE = (
    "Using the numbers {numbers}, create an equation that equals {target}. "
    "You can use basic arithmetic operations (+, -, *, /) and each number can only be used once. "
    "Show your work in <think> </think> tags. "
    "And return the final answer in <answer> </answer> tags, for example <answer> (1 + 2) / 3 </answer>."
)
RESPONSE_PROMPT = "Let me solve this step by step.\n<think>"


class CountdownTasksDataset(Dataset):
    """Prepare Countdown Tasks for training"""

    def __init__(
        self,
        tokenizer: Tokenizer,
        data_path: str,
        split: str = "train",
        test_size: int = 100,
    ):
        data = pd.read_parquet(Path(data_path) / "data")
        # use the last `test_size` examples for testing
        self.data = (
            data.iloc[:-test_size] if split == "train" else data.iloc[-test_size:]
        )
        self.tokenizer: Tokenizer = tokenizer

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data.iloc[idx].to_dict()
        item.update(self.encode_prefix(item["nums"], item["target"]))
        return item

    def encode_prefix(self, numbers: List[int], target: int):
        """Prefix is the *actual* input to the model."""
        user_message = USER_TEMPLATE.format(numbers=numbers, target=target)
        prefix = self.tokenizer.apply_chat_template(
            [
                {"role": "system", "content": SYSTEM_MESSAGE},
                {"role": "user", "content": user_message},
            ],
            RESPONSE_PROMPT,
        )
        tokens = self.tokenizer.encode(prefix)
        return {
            "prefix": prefix,
            "prefix_tokens": tokens.tokens,
            "prefix_token_ids": tokens.ids,
        }

    @staticmethod
    def collate_fn(batch: List[Dict[str, Any]]) -> MiniBatch:
        """Collate examples into a batch."""
        numbers = [item["nums"] for item in batch]
        target = [item["target"] for item in batch]
        prefix = [item["prefix"] for item in batch]
        prefix_tokens = [item["prefix_tokens"] for item in batch]
        prefix_token_ids = [item["prefix_token_ids"] for item in batch]
        return MiniBatch(
            numbers=numbers,
            target=target,
            prefix=prefix,
            prefix_tokens=prefix_tokens,
            prefix_token_ids=prefix_token_ids,
        )


def strict_format_reward_func(prompts, completions, **kwargs) -> list[float]:
    """Reward function that checks if the completion has the right format, with strict spacing."""
    pattern = r"^<reasoning>.*?</reasoning>\s*<answer>.*?</answer>\s*$"
    responses = [completion[0]["content"] for completion in completions]
    matches = [re.match(pattern, r, flags=re.DOTALL) for r in responses]
    rewards = [0.25 if match else 0.0 for match in matches]
    logger.info(f"Strict format rewards: {rewards}")
    return rewards


CONNECTIONS_PROMPT = """
You are an expert puzzle solving model.
Find groups of words that are related to each other. Each group is four words long. There are exactly four groups in total.
You may only use each word in one group.
Respond in the following format:
<reasoning>
...
</reasoning>
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
Assistant: <reasoning>
I'll start with breaking down the reasoning. For Group 1, the words related to wax: candle, crayon, honeycomb, seal — connecting to wax in various forms, such as crayon and candles being wax-based.
For Group 2, "My Favorite Things" connects to lyrics mentioning kettles, mittens, raindrops, whiskers.
Group 3 relates to teeth, covering canine, fang, molar, tusk. Group 4 involves "no" related phrases like "no excuse," "no defense."
</reasoning>
<answer>
<group> candle, crayon, honeycomb, seal</group>
<group> kettles, mittens, raindrops, whiskers</group>
<group> canine, fang, molar, tusk</group>
<group> defense, excuse, out, reason</group>
</answer>

"""


def _connections_map(example: dict) -> dict:
    words = example["words"]
    words_formatted = ", ".join(words)
    answer = []
    answer_groups = []
    for group in example["groups"]:
        answer.append(", ".join(group["words"]))
        answer_groups.append(group["words"])
    answer_formatted = "\n".join([f"<group>{a}</group>" for a in answer])
    return {
        "prompt": [
            {
                "role": "system",
                "content": CONNECTIONS_PROMPT,
            },
            {"role": "user", "content": words_formatted},
        ],
        "answer": f"<answer>{answer_formatted}</answer>",
        "answer_groups": answer_groups,
    }


class ConnectionsDataset(Dataset):
    def __init__(self, data: pd.DataFrame, tokenizer: Tokenizer):
        self.dataframe: pd.DataFrame = data
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx: int) -> dict:
        item = self.dataframe.loc[idx].to_dict()
        mapped = _connections_map(item)
        return mapped

    def get_collate_fn(self) -> Callable[[list[dict]], MiniBatch]:
        def _collate_fn(batch: list[dict]) -> MiniBatch:
            prompts = [item["prompt"] for item in batch]
            prefix_tokens = [self.tokenizer.encode(p).tokens for p in prompts]
            prefix_token_ids = [self.tokenizer.encode(p).ids for p in prompts]
            answer = [item["answer"] for item in batch]
            return MiniBatch(
                prefix=prompts,
                prefix_tokens=prefix_tokens,
                prefix_token_ids=prefix_token_ids,
                target=answer,
            )
        return _collate_fn


def create_connections_datasets(
    tokenizer: Tokenizer,
    jsonl_path: str = "connections_prompts.jsonl",
    num_samples: int = 10000,
    seed: int = 42,
) -> tuple[ConnectionsDataset, ConnectionsDataset]:
    # Load and process data
    prompts_pd = pd.read_json(jsonl_path, lines=True)
    df_groups = pd.json_normalize(prompts_pd["solution"], "groups")  # type: ignore

    # Generate samples
    groups = [
        {
            "groups": (
                g := df_groups.sample(4, replace=False).reset_index(drop=True)
            ).to_dict(orient="records"),
            "words": list(itertools.chain.from_iterable(g["words"].dropna())),
        }
        for _ in range(num_samples)
    ]

    # Create DataFrame and split
    groups_pd = pd.DataFrame(groups)
    train_data, val_data = train_test_split(groups_pd, test_size=0.1, random_state=seed)

    # Create datasets
    train_dataset = ConnectionsDataset(train_data, tokenizer)
    val_dataset = ConnectionsDataset(val_data, tokenizer)

    return train_dataset, val_dataset

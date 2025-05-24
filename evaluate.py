import os
import fire
from dotenv import load_dotenv
from torch.utils.data import DataLoader
from tasks import TASK_DEFINITIONS, TaskChoice
from litellm import batch_completion
from litellm.types.utils import ModelResponse
from typing import TypedDict
import pandas as pd
from loguru import logger

from tasks.connections import CONNECTIONS_PROMPT

"""
Evaluate a series of OSS models against prompts and evals for a specific task.
"""

load_dotenv(".env")

INFERENCE_MODELS = ["gpt-4.1-mini"]


class OutRow(TypedDict):
    model: str
    prompt: str
    response: str
    score: float


def main(task: TaskChoice = "connections"):
    task_definition = TASK_DEFINITIONS[task]
    dataset, reward_function = (
        task_definition["dataset"]("eval"),
        task_definition["reward_function"],
    )
    loader = DataLoader(dataset, batch_size=4, shuffle=False)

    out_rows: list[OutRow] = []

    os.makedirs("eval_results", exist_ok=True)

    for batch in loader:
        for model in INFERENCE_MODELS:
            convs = [
                [
                    {
                        "role": "system",
                        "content": CONNECTIONS_PROMPT,
                    },
                    {"role": "user", "content": p},
                ]
                for p in batch["prompt"]
            ]
            responses: list[ModelResponse] = batch_completion(
                model=model,
                messages=convs,
            )  # type: ignore
            for response in responses:
                response_content: str = response.choices[0].message.content  # type: ignore
                score = reward_function(response_content, batch)
                logger.info(f"Response: {response_content}, Score: {score}")
                out_rows.append(
                    {
                        "model": model,
                        "prompt": batch["prompt"],
                        "response": response_content,
                        "score": score,
                    }
                )

    out_rows_pd = pd.DataFrame(out_rows)

    print(out_rows_pd)
    out_rows_pd.to_csv(f"eval_results/eval_{task}.csv", index=False)


if __name__ == "__main__":
    fire.Fire(main)

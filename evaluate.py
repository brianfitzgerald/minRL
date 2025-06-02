import os
from typing import TypedDict, cast
from tqdm import tqdm

import fire
import pandas as pd
from dotenv import load_dotenv
from loguru import logger
from openai import AsyncOpenAI
from torch.utils.data import DataLoader

from minrl.tasks import TASK_DEFINITIONS, TaskChoice
from minrl.tasks.connections import (
    CONNECTIONS_PROMPT,
    ConnectionsSample,
)
from minrl.tasks.dataset import batch_to_samples

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


async def main(task: TaskChoice = "connections"):
    task_definition = TASK_DEFINITIONS[task]
    dataset, reward_function = (
        task_definition["dataset"]("eval"),
        task_definition["reward_function"],
    )
    loader = DataLoader(dataset, batch_size=4, shuffle=False)

    out_rows: list[OutRow] = []

    os.makedirs("eval_results", exist_ok=True)
    client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    for batch in tqdm(loader):
        batch: list[ConnectionsSample] = batch_to_samples(batch)  # type: ignore
        for model in INFERENCE_MODELS:
            convs = []
            for sample in batch:
                convs.append(
                    [
                        {
                            "role": "system",
                            "content": CONNECTIONS_PROMPT,
                        },
                        {"role": "user", "content": sample["prompt"]},
                    ]
                )

            responses = await asyncio.gather(
                *[
                    client.chat.completions.create(
                        model=model,
                        messages=conv,
                    )
                    for conv in convs
                ]
            )

            for sample, response in zip(batch, responses):
                assert response.choices[0].message.content is not None, (
                    "No response content"
                )
                response_content = response.choices[0].message.content
                score = reward_function(response_content, cast(dict, sample))
                logger.info(f"Score: {score}")
                out_rows.append(
                    {
                        "model": model,
                        "prompt": sample["prompt"],
                        "response": response_content,
                        "score": score,
                    }
                )

    out_rows_pd = pd.DataFrame(out_rows)

    logger.info(out_rows_pd)
    out_rows_pd.to_csv(f"eval_results/eval_{task}.csv", index=False)


if __name__ == "__main__":
    import asyncio

    asyncio.run(fire.Fire(main))

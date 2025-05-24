import os
import fire
from dotenv import load_dotenv
from torch.utils.data import DataLoader
from tasks import TASK_DEFINITIONS, TaskChoice
from openai import AsyncOpenAI
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

    for batch in loader:
        for model in INFERENCE_MODELS:
            logger.info(f"Evaluating {model} on {task}")
            for prompt in batch["prompt"]:
                conv = [
                    {
                        "role": "system",
                        "content": CONNECTIONS_PROMPT,
                    },
                    {"role": "user", "content": prompt},
                ]

                response = await client.chat.completions.create(
                    model=model,
                    messages=conv,
                )

                response_content: str = response.choices[0].message.content  # type: ignore
                score = reward_function(response_content, batch)
                logger.info(f"Response: {response_content}, Score: {score}")
                out_rows.append(
                    {
                        "model": model,
                        "prompt": prompt,
                        "response": response_content,
                        "score": score,
                    }
                )

    out_rows_pd = pd.DataFrame(out_rows)

    print(out_rows_pd)
    out_rows_pd.to_csv(f"eval_results/eval_{task}.csv", index=False)


if __name__ == "__main__":
    import asyncio

    asyncio.run(fire.Fire(main))

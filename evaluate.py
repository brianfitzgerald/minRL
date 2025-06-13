import os
from typing import Literal, NotRequired, TypedDict, cast
from tqdm import tqdm

import fire
import pandas as pd
from dotenv import load_dotenv
from loguru import logger
from openai import AsyncOpenAI
from openai.types.chat.chat_completion import ChatCompletion
from torch.utils.data import DataLoader
from vllm import LLM, RequestOutput

from minrl.tasks import TASK_DEFINITIONS, TaskChoice
from minrl.tasks.connections import (
    CONNECTIONS_PROMPT,
    ConnectionsSample,
)
from minrl.constants import QWEN_3_0_6B
from minrl.tasks.dataset import batch_to_samples

"""
Evaluate against any task in the minrl.tasks module.
"""

load_dotenv(".env")

ModelType = Literal["openrouter", "openai", "huggingface", "finetuned"]


class EvalModel(TypedDict):
    type: ModelType
    model_id: str
    base_model_id: NotRequired[str]


ModelName = Literal[
    "gemini-2.0-flash", "gpt-4o-mini", "Qwen3.0-6B", "qwen-grpo", "qwen-reinforce"
]

EVAL_MODELS: dict[ModelName, EvalModel] = {
    "gemini-2.0-flash": {
        "type": "openrouter",
        "model_id": "google/gemini-2.0-flash-001",
    },
    "gpt-4o-mini": {"type": "openai", "model_id": "gpt-4o-mini"},
    "Qwen3.0-6B": {"type": "huggingface", "model_id": QWEN_3_0_6B},
    "qwen-grpo": {
        "type": "finetuned",
        "model_id": "Qwen3_0.6B-grpo-20250612_211716_step_000050",
        "base_model_id": QWEN_3_0_6B,
    },
    "qwen-reinforce": {
        "type": "finetuned",
        "model_id": "Qwen3_0.6B-reinforce-20250612_213402_step_000900",
        "base_model_id": QWEN_3_0_6B,
    },
}


class OutRow(TypedDict):
    model: str
    prompt: str
    response: str
    score: float


async def main(
    task: TaskChoice = "connections",
    model_name: ModelName = "qwen-grpo",
    batch_size: int = 8,
):

    task_definition = TASK_DEFINITIONS[task]
    dataset, reward_function = (
        task_definition["dataset"]("eval"),
        task_definition["reward_function"],
    )
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    model = EVAL_MODELS[model_name]
    vllm_model: LLM | None = None
    using_vllm = model["type"] == "finetuned"
    if using_vllm:
        model_path = os.path.join(".", "checkpoints", model["model_id"])
        logger.info(f"Loading finetuned model from {model_path}")
        assert (
            "base_model_id" in model
        ), "Base model ID is required for finetuned models"
        vllm_model = LLM(
            model=model_path,
            tokenizer=model["base_model_id"],
            device="cuda",
            gpu_memory_utilization=0.2,
            max_model_len=1024,
            enforce_eager=True,
        )
    else:
        openai_client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    out_rows: list[OutRow] = []

    os.makedirs("eval_results", exist_ok=True)

    for batch in tqdm(loader):
        batch: list[ConnectionsSample] = batch_to_samples(batch)  # type: ignore
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

        if vllm_model is not None:
            responses = vllm_model.chat(convs)
        else:
            responses = await asyncio.gather(
                *[
                    openai_client.chat.completions.create(
                        model=model["model_id"],
                        messages=conv,
                    )
                    for conv in convs
                ]
            )

        for sample, response in zip(batch, responses):
            response_content = ""
            if using_vllm:
                assert isinstance(response, RequestOutput)
                response_content = response.outputs[0].text
            else:
                assert isinstance(response, ChatCompletion)
                response_content = response.choices[0].message.content
            assert response_content is not None, "No response content"
            score = reward_function(response_content, cast(dict, sample))
            logger.info(f"Score: {score}")
            out_rows.append(
                {
                    "model": model_name,
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

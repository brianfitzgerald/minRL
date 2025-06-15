import os
from typing import Any, Literal, NotRequired, TypedDict, cast
from tqdm import tqdm

import fire
import pandas as pd
from dotenv import load_dotenv
from loguru import logger
from openai import AsyncOpenAI
from openai.types.chat.chat_completion import ChatCompletion
from torch.utils.data import DataLoader
from vllm import LLM, RequestOutput, SamplingParams

from minrl.tasks import TASK_DEFINITIONS, TaskChoice
from minrl.constants import QWEN_3_0_6B

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
    "gemini_2_flash",
    "gpt_4.1_mini",
    "Qwen3.0-6B",
    "qwen_grpo",
    "qwen_reinforce",
    "magistral_medium",
]

EVAL_MODELS: dict[ModelName, EvalModel] = {
    "gemini_2_flash": {
        "type": "openrouter",
        "model_id": "google/gemini-2.0-flash-001",
    },
    "gpt_4.1_mini": {"type": "openai", "model_id": "gpt-4.1-mini"},
    "Qwen3.0-6B": {"type": "huggingface", "model_id": QWEN_3_0_6B},
    "qwen_grpo": {
        "type": "finetuned",
        "model_id": "Qwen3_0.6B-grpo-20250612_211716_step_000050",
        "base_model_id": QWEN_3_0_6B,
    },
    "qwen_reinforce": {
        "type": "finetuned",
        "model_id": "Qwen3_0.6B-reinforce-20250612_213402_step_000900",
        "base_model_id": QWEN_3_0_6B,
    },
    "magistral_medium": {
        "type": "openrouter",
        "model_id": "mistralai/magistral-medium-2506:thinking",
    },
}


class OutRow(TypedDict):
    model: str
    response: str
    score: float
    sample: dict[str, Any]


async def main(
    task: TaskChoice = "hanoi",
    model_name: ModelName = "gemini_2_flash",
    batch_size: int = 8,
):

    task_definition = TASK_DEFINITIONS[task]
    dataset, reward_function = (
        task_definition["dataset"]("eval"),
        task_definition["reward_function"],
    )
    loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=False, collate_fn=lambda x: x
    )

    model = EVAL_MODELS[model_name]
    vllm_model: LLM | None = None
    model_type = model["type"]
    if model_type == "finetuned":
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
    elif model_type == "openai":
        openai_client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    elif model_type == "openrouter":
        openai_client = AsyncOpenAI(
            api_key=os.getenv("OPENROUTER_API_KEY"),
            base_url="https://openrouter.ai/api/v1",
        )
    else:
        raise ValueError(f"Invalid model type: {model_type}")

    out_rows: list[OutRow] = []

    os.makedirs("eval_results", exist_ok=True)

    for i, batch in enumerate(tqdm(loader)):
        conversations = []
        for sample in batch:
            conversations.append(dataset.conversation(sample))

        logger.info(f"Requesting batch {i} of {len(conversations)} completions")
        if vllm_model is not None:
            sampling_params = SamplingParams(max_tokens=1024)
            responses = vllm_model.chat(conversations, sampling_params=sampling_params)
        else:
            responses = await asyncio.gather(
                *[
                    openai_client.chat.completions.create(
                        model=model["model_id"],
                        messages=conv,
                    )
                    for conv in conversations
                ]
            )

        for sample, response in zip(batch, responses):
            response_content = ""
            if model_type == "finetuned":
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
                    "response": response_content,
                    "score": score,
                    "sample": sample,
                }
            )

    out_rows_pd = pd.DataFrame(out_rows)

    logger.info(out_rows_pd)
    out_rows_pd.to_parquet(f"eval_results/eval_{task}_{model_name}.parquet")


if __name__ == "__main__":
    import asyncio

    asyncio.run(fire.Fire(main))

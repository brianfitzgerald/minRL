import asyncio
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
import modal
from modal.volume import FileEntryType
from pathlib import Path

from minrl.tasks import TASK_DEFINITIONS, TaskChoice
from minrl.constants import MODAL_MODELS_VOLUME_NAME, QWEN_3_0_6B

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
    "gpt_4o",
]

EVAL_MODELS: dict[ModelName, EvalModel] = {
    "gemini_2_flash": {
        "type": "openrouter",
        "model_id": "google/gemini-2.0-flash-001",
    },
    "gpt_4.1_mini": {"type": "openai", "model_id": "gpt-4.1-mini"},
    "gpt_4o": {"type": "openai", "model_id": "gpt-4o-2024-08-06"},
    "Qwen3.0-6B": {"type": "huggingface", "model_id": QWEN_3_0_6B},
    "qwen_grpo": {
        "type": "finetuned",
        "model_id": "Qwen3_0.6B-grpo-connections-0620_221447_step_003500",
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


def _download_checkpoint_from_modal(checkpoint_name: str):
    vol = modal.Volume.from_name(MODAL_MODELS_VOLUME_NAME)
    for file in vol.iterdir(f"checkpoints/{checkpoint_name}"):
        if file.type == FileEntryType.FILE:
            # remove checkpoints/ from the path as it's already in the path
            local_path = os.path.join(".", file.path)
            if os.path.exists(local_path):
                logger.info(f"Skipping {file.path} as it already exists locally.")
                continue
            logger.info(f"Downloading {file.path} to {local_path}")
            Path(local_path).parent.mkdir(parents=True, exist_ok=True)
            try:
                with open(local_path, "wb") as file_obj:
                    for chunk in tqdm(vol.read_file(file.path), desc="Downloading"):
                        file_obj.write(chunk)
            except Exception as e:
                logger.error(f"Error downloading {file.path}: {e}")
                raise e


async def main(
    task: TaskChoice = "connections",
    model_name: ModelName = "gemini_2_flash",
    batch_size: int = 32,
):
    task_definition = TASK_DEFINITIONS[task]
    dataset, reward_function = (
        task_definition["dataset"]("eval", host="local"),
        task_definition["reward_function"],
    )
    loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=False, collate_fn=lambda x: x
    )

    if model_name not in EVAL_MODELS:
        raise ValueError(f"Invalid model name: {model_name}")
    model = EVAL_MODELS[model_name]
    vllm_model: LLM | None = None
    model_type = model["type"]
    if model_type in ["finetuned", "huggingface"]:
        tokenizer_model_id = model["model_id"]
        if model_type == "finetuned":
            model_path = os.path.join(".", "checkpoints", model["model_id"])
            logger.info(f"Loading finetuned model from {model_path}")
            assert (
                "base_model_id" in model
            ), "Base model ID is required for finetuned models"
            tokenizer_model_id = model["base_model_id"]
            if not os.path.exists(model_path):
                logger.info(
                    f"{model['model_id']} not found locally, downloading from modal"
                )
                _download_checkpoint_from_modal(model["model_id"])
        else:
            model_path = model["model_id"]
        vllm_model = LLM(
            model=model_path,
            tokenizer=tokenizer_model_id,
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
            if model_type in ["finetuned", "huggingface"]:
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

    file_path = f"eval_results/eval_{task}_{model_name}.parquet"
    logger.info(f"Saving results to {file_path}")
    out_rows_pd.to_parquet(file_path)


if __name__ == "__main__":
    fire.Fire(main)

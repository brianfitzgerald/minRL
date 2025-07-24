import asyncio
import os
from typing import Any, Literal, TypedDict
import aiohttp
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

from minrl.tasks import TASK_DATASETS, TaskChoice
from minrl.constants import (
    MODAL_MODELS_VOLUME_NAME,
    Conversation,
    ConversationMessage,
    ModelName,
    INFERENCE_MODELS,
)

"""
Evaluate against any task in the minrl.tasks module.
"""

load_dotenv(".env")

Status = Literal["running", "done", "error"]


class OutRow(TypedDict):
    model: str
    # Parsed actions
    actions: list[str]
    # Outputs from the environment
    observations: list[str]
    # Full responses from inference
    full_responses: list[ConversationMessage]
    status: Status


def _save_results(out_rows: list[OutRow], task: TaskChoice, model_name: ModelName):
    out_rows = [row for row in out_rows if row["status"] == "done"]
    df = pd.DataFrame(out_rows)
    file_path = f"eval_results/eval_{task}_{model_name}.parquet"
    logger.info(f"Saving results to {file_path}")
    df.to_parquet(file_path)


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


async def _openrouter_request(
    model_id: str,
    conv: list[dict[str, str]],
    api_key: str,
    reasoning_effort: str | None = None,
):
    url = "https://openrouter.ai/api/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    data = {"model": model_id, "messages": conv}
    if reasoning_effort:
        data["reasoning"] = {"effort": reasoning_effort, "enabled": True}
    async with aiohttp.ClientSession() as session:
        async with session.post(url, headers=headers, json=data) as response:
            return await response.json()


async def main(
    task: TaskChoice = "zork",
    model_name: ModelName = "gpt-4.1-mini",
    batch_size: int = 4,
):
    dataset_cls = TASK_DATASETS[task]["dataset"]
    dataset = dataset_cls(split="eval", host="local")
    loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=False, collate_fn=lambda x: x
    )

    openai_client = None
    api_key = None
    if model_name not in INFERENCE_MODELS:
        raise ValueError(f"Invalid model name: {model_name}")
    model = INFERENCE_MODELS[model_name]
    vllm_model: LLM | None = None
    model_type = model["type"]
    if model_type in ["finetuned", "huggingface"]:
        tokenizer_model_id = model["model_id"]
        if model_type == "finetuned":
            model_path = os.path.join(".", "checkpoints", model["model_id"])
            logger.info(f"Loading finetuned model from {model_path}")
            assert "base_model_id" in model, (
                "Base model ID is required for finetuned models"
            )
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
        api_key = os.getenv("OPENROUTER_API_KEY")
        if not api_key:
            raise ValueError("OPENROUTER_API_KEY is not set")
    else:
        raise ValueError(f"Invalid model type: {model_type}")

    os.makedirs("eval_results", exist_ok=True)

    for i, batch in enumerate(tqdm(loader)):
        # Process responses
        batch_out_rows: list[OutRow] = [
            {
                "model": model_name,
                "actions": [],
                "observations": [],
                "full_responses": [],
                "status": "running",
            }
            for _ in batch
        ]

        while not all(row["status"] == "done" for row in batch_out_rows):
            # Create batch of conversations
            conversation_batch: list[Conversation] = []
            for j, (sample, row) in enumerate(zip(batch, batch_out_rows)):
                if row["status"] == "done":
                    continue
                conversation_batch.append(dataset.conversation(sample, j))

            # Perform inference
            if model_type in ("finetuned", "huggingface"):
                sampling_params = SamplingParams(max_tokens=dataset.max_tokens)
                assert vllm_model is not None, "VLLM model is not initialized"
                responses = vllm_model.chat(
                    conversation_batch,  # type: ignore
                    sampling_params=sampling_params,
                )
            elif model_type == "openai":
                assert openai_client is not None, "OpenAI client is not initialized"
                responses = await asyncio.gather(
                    *[
                        openai_client.chat.completions.create(
                            model=model["model_id"],
                            messages=conv,  # type: ignore
                        )
                        for conv in conversation_batch
                    ]
                )
            elif model_type == "openrouter":
                assert api_key is not None, "OpenRouter API key is not set"
                responses = await asyncio.gather(
                    *[
                        _openrouter_request(model["model_id"], conv, api_key, "medium")  # type: ignore
                        for conv in conversation_batch
                    ]
                )
            else:
                raise ValueError(f"Invalid model type: {model_type}")

            # Process responses
            for i, (response, row) in enumerate(zip(responses, batch_out_rows)):
                reasoning_content = None
                if row["status"] == "done":
                    continue
                response_content = ""
                if model_type in ["finetuned", "huggingface"]:
                    assert isinstance(response, RequestOutput)
                    response_content = response.outputs[0].text
                elif model_type == "openai":
                    assert isinstance(response, ChatCompletion)
                    response_content = response.choices[0].message.content
                elif model_type == "openrouter":
                    assert isinstance(response, dict)
                    response_content = response["choices"][0]["message"]["content"]
                    reasoning_content = None
                    if "reasoning" in response["choices"][0]["message"]:
                        reasoning_content = response["choices"][0]["message"][
                            "reasoning"
                        ]
                assert response_content is not None, "No response content"

                batch_out_rows[i]["full_responses"].append(
                    {
                        "role": "assistant",
                        "content": response_content,
                        "reasoning": reasoning_content,
                    }
                )

                obs, done = dataset.post_generation(i, response_content)
                batch_out_rows[i]["observations"].append(obs)
                batch_out_rows[i]["actions"].append(response_content)
                if done and row["status"] == "running":
                    _save_results(batch_out_rows, task, model_name)
                    batch_out_rows[i]["status"] = "done"

        _save_results(batch_out_rows, task, model_name)


if __name__ == "__main__":
    fire.Fire(main)

import asyncio
import os
from pathlib import Path

import aiohttp
import fire
import pandas as pd
from dotenv import load_dotenv
from loguru import logger
from openai import AsyncOpenAI
from openai.types.chat.chat_completion import ChatCompletion
from torch.utils.data import DataLoader
from tqdm import tqdm
from vllm import LLM, RequestOutput, SamplingParams

from minrl.constants import (
    INFERENCE_MODELS,
    Conversation,
    EvalsOutRow,
    ModelName,
)
from minrl.tasks import TASK_DATASETS, TaskChoice
from minrl.modal_utils import download_checkpoint_from_modal

"""
Evaluate against any task in the minrl.tasks module.
"""

load_dotenv(".env")


def _save_results(out_rows: list[EvalsOutRow], task: TaskChoice, model_name: ModelName):
    out_rows = [row for row in out_rows if row["status"] == "done"]
    df = pd.DataFrame(out_rows)
    file_path = f"eval_results/{task}/eval_{model_name}.parquet"
    if not Path(file_path).parent.exists():
        Path(file_path).parent.mkdir(parents=True, exist_ok=True)
    logger.info(f"Saving results to {file_path}")
    df.to_parquet(file_path)


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
            assert (
                "base_model_id" in model
            ), "Base model ID is required for finetuned models"
            tokenizer_model_id = model["base_model_id"]
            if not os.path.exists(model_path):
                logger.info(
                    f"{model['model_id']} not found locally, downloading from modal"
                )
                download_checkpoint_from_modal(model["model_id"])
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
        batch_out: list[EvalsOutRow] = [
            {
                "model": model_name,
                "conversation": [],
                "status": "running",
            }
            for _ in batch
        ]
        # Create batch of conversations
        conversation_batch: list[Conversation] = []
        for j, sample in enumerate(batch):
            conversation_batch.append(dataset.initial_conversation(sample, j))

        # iterate through conversation steps
        while not all(row["status"] == "done" for row in batch_out):
            for j, (sample, row) in enumerate(zip(batch, batch_out)):
                if row["status"] == "done":
                    continue

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
            for i, (response, row) in enumerate(zip(responses, batch_out)):
                reasoning_content = None
                if row["status"] == "done":
                    continue
                response_str = ""
                if model_type in ["finetuned", "huggingface"]:
                    assert isinstance(response, RequestOutput)
                    response_str = response.outputs[0].text
                elif model_type == "openai":
                    assert isinstance(response, ChatCompletion)
                    response_str = response.choices[0].message.content
                elif model_type == "openrouter":
                    assert isinstance(response, dict)
                    response_str = response["choices"][0]["message"]["content"]
                    reasoning_content = None
                    if "reasoning" in response["choices"][0]["message"]:
                        reasoning_content = response["choices"][0]["message"][
                            "reasoning"
                        ]
                assert response_str is not None, "No response content"

                conversation_batch[i].append(
                    {
                        "role": "assistant",
                        "content": response_str,
                        "reasoning": reasoning_content,
                    }
                )

                obs, done = dataset.get_next_state(i, conversation_batch[i])
                if done and row["status"] == "running":
                    _save_results(batch_out, task, model_name)
                    batch_out[i]["status"] = "done"
                    batch_out[i]["conversation"] = conversation_batch[i]
                conversation_batch[i].append(
                    {
                        "role": "user",
                        "content": obs,
                    }
                )

        _save_results(batch_out, task, model_name)


if __name__ == "__main__":
    fire.Fire(main)

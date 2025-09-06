import asyncio
from datetime import datetime
import random
import os
from pathlib import Path
from typing import cast

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
    EvalSample,
    ModelName,
)
from minrl.tasks import TASK_DATASETS, TaskChoice
from minrl.tasks.dataset import MinRLDataset
from minrl.tasks.zork import ZorkDataset
from minrl.modal_utils import download_checkpoint_from_modal

"""
Evaluate against any task in the minrl.tasks module.
"""

load_dotenv(".env")


def _save_all_results(all_outs: dict[ModelName, list[EvalSample]], task: TaskChoice):
    """Save all results from all models in a single parquet file."""
    # Collect all results from all models
    all_rows = []
    for _, out_rows in all_outs.items():
        # Include both done and error samples for analysis
        valid_rows = [row for row in out_rows if row["status"] in ["done", "error"]]
        all_rows.extend(valid_rows)

    if not all_rows:
        logger.info("No results to save")
        return

    df = pd.DataFrame(all_rows)
    # Use MMDDHHMM format for timestamp (month day hour minute)
    timestamp_str = datetime.now().strftime("%m%d%H%M")
    file_path = f"eval_results/{task}/eval_all_models_{timestamp_str}.parquet"
    if not Path(file_path).parent.exists():
        Path(file_path).parent.mkdir(parents=True, exist_ok=True)
    logger.info(f"Saved all results to {file_path}")
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


def _sanitize_conversation(conversation: Conversation) -> Conversation:
    return [
        {
            "role": message["role"],
            "content": message["content"],
        }
        for message in conversation
    ]


async def main(
    task: TaskChoice = "zork",
    model_names: list[ModelName] | None = None,
    batch_size: int = 4,
):
    """Run evaluation for one or more models in a single batched loop.

    Backwards compatible: if model_name is provided, it will be used.
    """
    # Resolve models list
    if model_names is None:
        model_names = ["gemini_2.5_flash", "magistral_medium", "gpt-4.1-mini"]

    # Validate models
    for m in model_names:
        if m not in INFERENCE_MODELS:
            raise ValueError(f"Invalid model name: {m}")

    dataset_cls = TASK_DATASETS[task]["dataset"]

    datasets: dict[ModelName, MinRLDataset] = {
        m: dataset_cls(split="eval", host="local") for m in model_names
    }

    # Use the first dataset instance to drive batching
    reference_dataset = datasets[model_names[0]]
    loader = DataLoader(
        reference_dataset, batch_size=batch_size, shuffle=False, collate_fn=lambda x: x
    )

    random.seed(42)

    # Initialize backends per model
    vllm_models: dict[ModelName, LLM] = {}
    openai_clients: dict[ModelName, AsyncOpenAI] = {}
    openrouter_keys: dict[str, str] = {}

    model_types: dict[ModelName, str] = {}
    model_cfgs = {m: INFERENCE_MODELS[cast(ModelName, m)] for m in model_names}

    for m, cfg in model_cfgs.items():
        model_types[cast(ModelName, m)] = cfg["type"]
        if cfg["type"] in ["finetuned", "huggingface"]:
            tokenizer_model_id = cfg["model_id"]
            if cfg["type"] == "finetuned":
                model_path = os.path.join(".", "checkpoints", cfg["model_id"])
                logger.info(f"Loading finetuned model from {model_path}")
                assert "base_model_id" in cfg, (
                    "Base model ID is required for finetuned models"
                )
                tokenizer_model_id = cfg["base_model_id"]
                if not os.path.exists(model_path):
                    logger.info(
                        f"{cfg['model_id']} not found locally, downloading from modal"
                    )
                    download_checkpoint_from_modal(cfg["model_id"])
            else:
                model_path = cfg["model_id"]
            vllm_models[cast(ModelName, m)] = LLM(
                model=model_path,
                tokenizer=tokenizer_model_id,
                device="cuda",
                gpu_memory_utilization=0.2,
                max_model_len=1024,
                enforce_eager=True,
            )
        elif cfg["type"] == "openai":
            openai_clients[cast(ModelName, m)] = AsyncOpenAI(
                api_key=os.getenv("OPENAI_API_KEY")
            )
        elif cfg["type"] == "openrouter":
            key = os.getenv("OPENROUTER_API_KEY")
            if not key:
                raise ValueError("OPENROUTER_API_KEY is not set")
            openrouter_keys[m] = key
        else:
            raise ValueError(f"Invalid model type: {cfg['type']}")

    os.makedirs("eval_results", exist_ok=True)

    # Accumulate outputs per model
    all_outs: dict[ModelName, list[EvalSample]] = {m: [] for m in model_names}

    for batch_index, batch in enumerate(tqdm(loader)):
        # Per-model tracking structures for this batch
        batch_outs: dict[ModelName, list[EvalSample]] = {
            m: [
                {
                    "model": m,
                    "conversation": [],
                    "status": "running",
                    "game": "",
                }
                for _ in batch
            ]
            for m in model_names
        }

        conversation_batches: dict[ModelName, list[Conversation]] = {
            m: [] for m in model_names
        }

        # Initialize conversations per model and capture game names (e.g., for zork)
        for local_idx, sample in enumerate(batch):
            for m in model_names:
                ds = datasets[m]
                conv = ds.initial_conversation(sample, local_idx)
                conversation_batches[m].append(conv)
                if task == "zork":
                    ds = cast(ZorkDataset, ds)
                    game_name = ds.sample_games.get(local_idx)
                    if game_name:
                        batch_outs[m][local_idx]["game"] = game_name

        # Iterate steps until all models finish their rows
        step = 0
        while True:
            model_done_flags = {
                m: all(row["status"] == "done" for row in batch_outs[m])
                for m in model_names
            }
            if all(model_done_flags.values()):
                break

            # Build and run inference tasks for all not-done models concurrently
            async def _infer_for_model(m: ModelName):
                cfg = model_cfgs[m]
                model_type = model_types[m]
                ds = datasets[m]
                # Only run for rows that are still running
                run_indices = [
                    i
                    for i, row in enumerate(batch_outs[m])
                    if row["status"] == "running"
                ]
                if not run_indices:
                    return m, [], []
                convs = [
                    _sanitize_conversation(conversation_batches[m][i])
                    for i in run_indices
                ]
                try:
                    if model_type in ("finetuned", "huggingface"):
                        sampling_params = SamplingParams(max_tokens=ds.max_tokens)
                        vllm = vllm_models[m]
                        # Run sync vLLM call in a thread to avoid blocking
                        responses = await asyncio.to_thread(
                            lambda: vllm.chat(convs, sampling_params=sampling_params)  # type: ignore
                        )
                    elif model_type == "openai":
                        client = openai_clients[m]
                        responses = await asyncio.gather(
                            *[
                                client.chat.completions.create(
                                    model=cfg["model_id"],
                                    messages=conv,  # type: ignore
                                )
                                for conv in convs
                            ]
                        )
                    elif model_type == "openrouter":
                        key = openrouter_keys[m]
                        responses = await asyncio.gather(
                            *[
                                _openrouter_request(
                                    cfg["model_id"],
                                    conv,  # pyright: ignore[reportArgumentType]
                                    key,
                                    "medium",  # pyright: ignore[reportArgumentType]
                                )  # type: ignore
                                for conv in convs
                            ]
                        )
                    else:
                        raise ValueError(f"Invalid model type: {model_type}")
                except Exception as e:
                    logger.error(f"Inference failed for model {m}: {e}")
                    # Return empty responses to mark as failed
                    responses = []
                return m, responses, run_indices

            inference_tasks = [
                _infer_for_model(m) for m in model_names if not model_done_flags[m]
            ]
            results = await asyncio.gather(*inference_tasks)

            # Process responses for each model
            for m, responses, run_indices in results:
                if not responses:
                    continue
                mtype = model_types[m]
                ds = datasets[m]
                for resp, idx in zip(responses, run_indices):
                    row = batch_outs[m][idx]
                    if row["status"] == "done":
                        continue
                    reasoning_content = None
                    response_str = ""
                    try:
                        if mtype in ["finetuned", "huggingface"]:
                            assert isinstance(resp, RequestOutput)
                            response_str = resp.outputs[0].text
                        elif mtype == "openai":
                            assert isinstance(resp, ChatCompletion)
                            response_str = resp.choices[0].message.content
                        elif mtype == "openrouter":
                            assert isinstance(resp, dict)
                            response_str = resp["choices"][0]["message"]["content"]
                            if "reasoning" in resp["choices"][0]["message"]:
                                reasoning_content = resp["choices"][0]["message"][
                                    "reasoning"
                                ]
                        assert response_str is not None, "No response content"
                    except Exception as e:
                        logger.error(
                            f"Failed to parse response for model {m}, sample {idx}: {e}"
                        )
                        # Mark this sample as failed and continue
                        row["status"] = "error"
                        continue

                    conversation_batches[m][idx].append(
                        {
                            "role": "assistant",
                            "content": response_str,
                            "reasoning": reasoning_content,
                        }
                    )

                    try:
                        message, done = ds.step(idx, conversation_batches[m][idx])

                        if done and row["status"] == "running":
                            batch_outs[m][idx]["status"] = "done"
                            batch_outs[m][idx]["conversation"] = conversation_batches[
                                m
                            ][idx]
                            all_outs[m].append(batch_outs[m][idx])
                            _save_all_results(all_outs, task)

                        conversation_batches[m][idx].append(message)
                    except Exception as e:
                        logger.error(
                            f"Step processing failed for model {m}, sample {idx}: {e}"
                        )
                        # Mark this sample as failed
                        row["status"] = "error"

            # Log compact status per model
            status_str = {}
            for m in model_names:
                flags = [
                    "R"
                    if row["status"] == "running"
                    else "D"
                    if row["status"] == "done"
                    else "E"
                    for row in batch_outs[m]
                ]
                status_str[m] = "".join(flags)
            logger.info(f"Step: {step}, statuses per model: {status_str}")
            step += 1
        _save_all_results(all_outs, task)

    # Save all results from all models in a single file at the end
    _save_all_results(all_outs, task)


if __name__ == "__main__":
    fire.Fire(main)

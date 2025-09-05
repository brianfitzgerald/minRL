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
from minrl.tasks.zork import ZorkDataset
from minrl.modal_utils import download_checkpoint_from_modal

"""
Evaluate against any task in the minrl.tasks module.
"""

load_dotenv(".env")


def _save_results(out_rows: list[EvalSample], task: TaskChoice, model_name: ModelName):
    out_rows = [row for row in out_rows if row["status"] == "done"]
    if not out_rows:
        logger.warning(f"No completed results to save for {model_name}")
        return
    df = pd.DataFrame(out_rows)
    timestamp_short_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    # Include number of samples in filename
    num_samples = len(df.groupby("sample_run")) if "sample_run" in df.columns else 1
    file_path = f"eval_results/{task}/eval_{model_name}_{num_samples}samples_{timestamp_short_str}.parquet"
    if not Path(file_path).parent.exists():
        Path(file_path).parent.mkdir(parents=True, exist_ok=True)
    logger.info(f"Saving {len(out_rows)} results to {file_path}")
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
    model_names: str = "gemini_2.5_flash",
    samples_per_game: int = 1,
    batch_size: int = 8,
):
    # Parse model names from comma-separated string
    model_name_list: list[ModelName] = [
        cast(ModelName, name.strip()) for name in model_names.split(",")
    ]

    dataset_cls = TASK_DATASETS[task]["dataset"]
    dataset = dataset_cls(split="eval", host="local")
    loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=False, collate_fn=lambda x: x
    )
    random.seed(42)

    # Validate model names
    for model_name in model_name_list:
        if model_name not in INFERENCE_MODELS:
            raise ValueError(f"Invalid model name: {model_name}")

    # Initialize models
    model_configs = {}
    for model_name in model_name_list:
        model = INFERENCE_MODELS[model_name]
        model_config = {
            "model": model,
            "openai_client": None,
            "api_key": None,
            "vllm_model": None,
        }

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
                    download_checkpoint_from_modal(model["model_id"])
            else:
                model_path = model["model_id"]
            model_config["vllm_model"] = LLM(
                model=model_path,
                tokenizer=tokenizer_model_id,
                device="cuda",
                gpu_memory_utilization=0.2 / len(model_name_list),  # Split GPU memory
                max_model_len=1024,
                enforce_eager=True,
            )
        elif model_type == "openai":
            model_config["openai_client"] = AsyncOpenAI(
                api_key=os.getenv("OPENAI_API_KEY")
            )
        elif model_type == "openrouter":
            api_key = os.getenv("OPENROUTER_API_KEY")
            if not api_key:
                raise ValueError("OPENROUTER_API_KEY is not set")
            model_config["api_key"] = api_key
        else:
            raise ValueError(f"Invalid model type: {model_type}")

        model_configs[model_name] = model_config

    os.makedirs("eval_results", exist_ok=True)

    # Store all results by model and sample run
    all_results: dict[ModelName, list[EvalSample]] = {
        model_name: [] for model_name in model_name_list
    }

    # Evaluate all models on each game batch simultaneously, samples_per_game times
    for sample_run in range(samples_per_game):
        logger.info(f"Starting sample run {sample_run + 1}/{samples_per_game}")
        
        for batch in tqdm(loader, desc=f"Sample run {sample_run + 1}"):
            # Create batch_out for each model
            model_batch_outs: dict[ModelName, list[EvalSample]] = {}
            model_conversation_batches: dict[ModelName, list[Conversation]] = {}
            
            for model_name in model_name_list:
                batch_out: list[EvalSample] = [
                    {
                        "model": model_name,
                        "conversation": [],
                        "status": "running",
                        "game": "",
                        "sample_run": sample_run,
                    }
                    for _ in batch
                ]
                # Create batch of conversations for this model
                conversation_batch: list[Conversation] = []
                for i, sample in enumerate(batch):
                    conversation_batch.append(dataset.initial_conversation(sample, i))
                    # Capture game information if available (for zork task)
                    if task == "zork":
                        dataset_zork = cast(ZorkDataset, dataset)
                        game_name = dataset_zork.sample_games.get(i)
                        if game_name:
                            batch_out[i]["game"] = game_name
                
                model_batch_outs[model_name] = batch_out
                model_conversation_batches[model_name] = conversation_batch

            # iterate through conversation steps for all models simultaneously
            step = 0
            while not all(
                all(row["status"] == "done" for row in model_batch_out)
                for model_batch_out in model_batch_outs.values()
            ):
                # Batch requests across all models
                all_requests = []
                request_metadata = []  # Track which request belongs to which model/sample
                
                for model_name in model_name_list:
                    model_config = model_configs[model_name]
                    batch_out = model_batch_outs[model_name]
                    conversation_batch = model_conversation_batches[model_name]
                    
                    # Skip if all conversations for this model are done
                    if all(row["status"] == "done" for row in batch_out):
                        continue
                        
                    conversation_batch_for_inference = [
                        _sanitize_conversation(conv) for conv in conversation_batch
                    ]
                    
                    model = model_config["model"]
                    model_type = model["type"]
                    
                    # Collect requests for batching
                    for i, (conv, row) in enumerate(zip(conversation_batch_for_inference, batch_out)):
                        if row["status"] != "done":
                            request_metadata.append((model_name, model_type, model_config, i))
                            all_requests.append(conv)
                
                if not all_requests:
                    break
                    
                # Execute all requests in parallel by model type
                
                # Group by model type for efficient batching
                openai_requests = []
                openrouter_requests = []
                vllm_requests_by_model = {}
                
                for i, (model_name, model_type, model_config, _) in enumerate(request_metadata):
                    conv = all_requests[i]
                    if model_type == "openai":
                        openai_requests.append((i, model_config, conv))
                    elif model_type == "openrouter":
                        openrouter_requests.append((i, model_config, conv))
                    elif model_type in ["finetuned", "huggingface"]:
                        if model_name not in vllm_requests_by_model:
                            vllm_requests_by_model[model_name] = []
                        vllm_requests_by_model[model_name].append((i, model_config, conv))
                
                # Initialize responses list
                responses = [None] * len(all_requests)
                
                # Execute OpenAI requests in batch
                if openai_requests:
                    openai_responses = await asyncio.gather(*[
                        model_config["openai_client"].chat.completions.create(
                            model=model_config["model"]["model_id"],
                            messages=conv,  # type: ignore
                        )
                        for _, model_config, conv in openai_requests
                    ])
                    for (idx, _, _), response in zip(openai_requests, openai_responses):
                        responses[idx] = response
                
                # Execute OpenRouter requests in batch
                if openrouter_requests:
                    openrouter_responses = await asyncio.gather(*[
                        _openrouter_request(
                            model_config["model"]["model_id"], 
                            conv, 
                            model_config["api_key"], 
                            "medium"
                        )
                        for _, model_config, conv in openrouter_requests
                    ])
                    for (idx, _, _), response in zip(openrouter_requests, openrouter_responses):
                        responses[idx] = response
                
                # Execute VLLM requests by model
                for model_name, model_requests in vllm_requests_by_model.items():
                    if model_requests:
                        sampling_params = SamplingParams(max_tokens=dataset.max_tokens)
                        vllm_model = model_requests[0][1]["vllm_model"]
                        assert vllm_model is not None, f"VLLM model for {model_name} is not initialized"
                        
                        convs_for_vllm = [conv for _, _, conv in model_requests]
                        vllm_responses = vllm_model.chat(
                            convs_for_vllm,  # type: ignore
                            sampling_params=sampling_params,
                        )
                        for (idx, _, _), response in zip(model_requests, vllm_responses):
                            responses[idx] = response

                # Process responses for all models
                for response_idx, (model_name, model_type, model_config, sample_idx) in enumerate(request_metadata):
                    response = responses[response_idx]
                    batch_out = model_batch_outs[model_name]
                    conversation_batch = model_conversation_batches[model_name]
                    row = batch_out[sample_idx]
                    
                    if row["status"] == "done":
                        continue
                        
                    reasoning_content = None
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
                        if "reasoning" in response["choices"][0]["message"]:
                            reasoning_content = response["choices"][0]["message"]["reasoning"]
                    
                    assert response_str is not None, "No response content"

                    conversation_batch[sample_idx].append({
                        "role": "assistant",
                        "content": response_str,
                        "reasoning": reasoning_content,
                    })

                    message, done = dataset.step(sample_idx, conversation_batch[sample_idx])

                    # if sample is done, save result
                    if done and row["status"] == "running":
                        row["status"] = "done"
                        row["conversation"] = conversation_batch[sample_idx]
                        all_results[model_name].append(row)

                    conversation_batch[sample_idx].append(message)

                # Log progress for all models
                model_statuses = {}
                for model_name in model_name_list:
                    batch_out = model_batch_outs[model_name]
                    statuses = [
                        "R" if row["status"] == "running"
                        else "D" if row["status"] == "done"
                        else "E"
                        for row in batch_out
                    ]
                    model_statuses[model_name] = "".join(statuses)
                
                logger.info(f"Step {step}: {model_statuses}")
                step += 1

    # Save results for all models
    for model_name in model_name_list:
        logger.info(f"Saving results for {model_name}: {len(all_results[model_name])} samples")
        _save_results(all_results[model_name], task, model_name)


if __name__ == "__main__":
    fire.Fire(main)

import gc
import time
from pathlib import Path
from typing import Any, cast

import psutil
import os
import numpy as np
import torch
import torch.nn as nn
from loguru import logger
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers.models.auto.modeling_auto import AutoModelForCausalLM
from transformers.models.auto.tokenization_auto import AutoTokenizer
from transformers.tokenization_utils_base import PreTrainedTokenizerBase
from vllm import LLM
from vllm.envs import set_vllm_use_v1

from minrl.algorithms import rollout, update_policy
from minrl.constants import Episode, HostType, LoggerChoice, TrainerConfig
from minrl.metrics import MetricsWrapper
from minrl.tasks import TASK_DATASETS
from minrl.tasks.dataset import MinRLDataset
from minrl.utils import compute_metrics


def get_available_device() -> str:
    return (
        "cuda:0"
        if torch.cuda.is_available()
        else "mps"
        if torch.mps.is_available()
        else "cpu"
    )


def simple_timestamp() -> str:
    return time.strftime("%m%d_%H%M%S")


USING_MPS = torch.backends.mps.is_available() and torch.backends.mps.is_built()
if not USING_MPS:
    from bitsandbytes.optim import Adam8bit  # type: ignore


def get_memory_usage():
    """Get current memory usage in MB and percentage of total VRAM available."""
    process = psutil.Process(os.getpid())
    cpu_memory_mb = process.memory_info().rss / 1024 / 1024

    if torch.cuda.is_available():
        # Get GPU memory usage
        gpu_memory_allocated = torch.cuda.memory_allocated() / 1024 / 1024  # MB
        gpu_memory_reserved = torch.cuda.memory_reserved() / 1024 / 1024  # MB
        gpu_memory_total = (
            torch.cuda.get_device_properties(0).total_memory / 1024 / 1024
        )  # MB
        gpu_memory_percentage = (gpu_memory_allocated / gpu_memory_total) * 100

        return {
            "cpu_memory_mb": cpu_memory_mb,
            "gpu_memory_allocated_mb": gpu_memory_allocated,
            "gpu_memory_reserved_mb": gpu_memory_reserved,
            "gpu_memory_total_mb": gpu_memory_total,
            "gpu_memory_percentage": gpu_memory_percentage,
        }
    else:
        return {
            "cpu_memory_mb": cpu_memory_mb,
            "gpu_memory_allocated_mb": 0,
            "gpu_memory_reserved_mb": 0,
            "gpu_memory_total_mb": 0,
            "gpu_memory_percentage": 0,
        }


class Trainer:
    tokenizer: PreTrainedTokenizerBase | None = None
    model: AutoModelForCausalLM | None = None

    def __init__(self, host_type: HostType) -> None:
        """Initialize the trainer with configuration."""
        self.config = TrainerConfig()
        self.device = torch.device(get_available_device())
        self.host_type: HostType = host_type
        self.dtype = torch.bfloat16

    def init_model(self):
        """Initialize the model and tokenizer."""
        torch.set_default_device(self.device)
        if self.device.type == "mps":
            logger.warning("vLLM does not support MPS backend, falling back to CPU.")

        # Reduce vLLM memory usage significantly
        memory_info = get_memory_usage()
        logger.info(
            f"Memory usage - CPU: {memory_info['cpu_memory_mb']:.1f}MB, GPU: {memory_info['gpu_memory_allocated_mb']:.1f}MB ({memory_info['gpu_memory_percentage']:.1f}%)"
        )
        self.vllm_model = LLM(
            model=self.config.model_id,
            gpu_memory_utilization=0.2,
            max_model_len=2048,
            max_seq_len_to_capture=2048,
            enforce_eager=True,
            dtype="float16" if USING_MPS else "bfloat16",
        )
        tokenizer = AutoTokenizer.from_pretrained(self.config.model_id)
        attn_impl = "sdpa"  # Use SDPA instead of flash_attention_2 to avoid compatibility issues

        # Use more memory-efficient model loading
        model: AutoModelForCausalLM = AutoModelForCausalLM.from_pretrained(  # pyright: ignore[reportAssignmentType]
            self.config.model_id,
            device_map="auto",
            dtype=self.dtype,
            attn_implementation=attn_impl,
            low_cpu_mem_usage=True,  # Enable low memory usage
            use_cache=False,  # Disable KV cache to save memory
        )

        logger.info("Model loaded.")
        logger.info(f"Using device {self.device}, attn impl {attn_impl}")
        torch.random.manual_seed(42)
        self.tokenizer = tokenizer
        self.model = model

    def init_training(self) -> None:
        """Initialize training components including dataloader, optimizer, and logging."""
        assert self.tokenizer is not None, "Tokenizer not initialized"
        dataset_cls: type[MinRLDataset] = TASK_DATASETS[self.config.task]["dataset"]
        self.reward_function = TASK_DATASETS[self.config.task]["reward_function"]
        self.train_dataset = dataset_cls(
            split="train", host=self.host_type, tokenizer=self.tokenizer
        )
        self.eval_dataset = dataset_cls(
            split="eval", host=self.host_type, tokenizer=self.tokenizer
        )
        generator = torch.Generator(device=self.device)
        # Reduce batch size for memory efficiency
        effective_batch_size = max(1, self.config.train_batch_size // 2)
        self.train_dataloader = DataLoader(
            self.train_dataset,
            shuffle=True,
            generator=generator,
            batch_size=effective_batch_size,
            collate_fn=lambda x: x,
            pin_memory=False,  # Disable pin_memory to save memory
            num_workers=0,  # Use single process to avoid memory overhead
        )

        if self.config.optimizer == "adamw":
            self.optimizer = torch.optim.AdamW(
                cast(nn.Module, self.model).parameters(), lr=self.config.lr
            )
        elif self.config.optimizer == "adamw_8bit":
            self.optimizer = Adam8bit(
                cast(nn.Module, self.model).parameters(),
                lr=self.config.lr,
                betas=(0.9, 0.999),
                eps=1e-8,
            )
        self.start_time = time.time()
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.run_name = f"{self.config.model_display_name}-{self.config.algorithm}-{self.config.task}-{simple_timestamp()}"
        logger_choice: LoggerChoice = (
            "wandb" if self.host_type == "modal" else "tensorboard"
        )
        logger.info(f"Logging to: {logger_choice}")
        self.metrics_wrapper = MetricsWrapper(
            logger_choice, self.config.task, self.config.model_id, self.run_name
        )

    def train(self) -> None:
        """Run the main training loop.

        For each step:
            1. Performs rollout to generate episodes
            2. Updates policy using GRPO
            3. Evaluates model periodically
            4. Saves checkpoints periodically
            5. Logs metrics to TensorBoard or Weights & Biases
        """

        prev_reward_std: float | None = None

        for step, batch in enumerate(self.train_dataloader, start=1):
            logger.info(f"Starting rollout for step {step}")

            assert self.model is not None
            assert self.tokenizer is not None

            conversations = [
                self.train_dataset.initial_conversation(sample, i)
                for i, sample in enumerate(batch)
            ]

            episodes = rollout(
                self.config,
                self.tokenizer,
                self.config.group_size,
                self.train_dataset.max_steps,
                conversations,
                samples=batch,
                reward_function=self.reward_function,
                vllm_model=self.vllm_model,
                prev_reward_std=prev_reward_std,
            )

            logger.info(f"Updating policy for step {step}")

            # Use smaller micro batch size for memory efficiency
            micro_batch_size = max(1, self.config.train_batch_size // 4)
            results = update_policy(
                model=cast(nn.Module, self.model),
                optimizer=self.optimizer,
                episodes=episodes,
                tokenizer=self.tokenizer,
                micro_batch_size=micro_batch_size,
                pad_token_id=int(cast(Any, self.tokenizer.pad_token_id)),
                max_grad_norm=self.config.max_grad_norm,
                device=self.device,
                vllm_model=self.vllm_model,
                algorithm=self.config.algorithm,
            )

            # Compute current reward std for next iteration before clearing memory
            current_rewards = [episode.reward for episode in episodes]
            current_reward_std = float(np.std(current_rewards))

            # Clear memory after each step
            del episodes
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            if torch.cuda.is_available():
                torch.cuda.synchronize()
            if step % self.config.eval_interval == 0:
                self.evaluate(step)

            # Get temperature used for logging
            from minrl.algorithms import compute_scaled_temperature

            temperature_used = compute_scaled_temperature(self.config, prev_reward_std)

            # Note: episodes are already deleted for memory optimization
            # We'll pass empty list to compute_metrics to avoid errors
            compute_metrics(
                [],  # Empty list since episodes were deleted for memory optimization
                results,
                self.metrics_wrapper,
                step,
                self.optimizer,
                temperature_used,
            )

            # Update prev_reward_std for next iteration
            prev_reward_std = current_reward_std
            # save checkpoint
            if step % self.config.ckpt_save_interval == 0:
                logger.info(f"Saving checkpoint for step {step}")
                output_file = self.checkpoint_dir / f"{self.run_name}_step_{step:06d}"
                self.model.save_pretrained(output_file)  # type: ignore
                logger.info(f"Saved checkpoint to {output_file}")

        self.metrics_wrapper.close()

    def evaluate(self, step: int) -> None:
        """Evaluate the current model.

        Returns:
            float: The evaluation success rate (currently returns 0.0 as placeholder)
        """
        eval_loader = DataLoader(
            self.eval_dataset,
            shuffle=False,
            batch_size=self.config.eval_batch_size,
            collate_fn=lambda x: x,
        )
        assert self.model is not None

        assert self.tokenizer is not None
        mean_reward = 0
        episodes: list[Episode] = []
        for batch in tqdm(eval_loader):
            conversations = [
                self.eval_dataset.initial_conversation(sample, i)
                for i, sample in enumerate(batch)
            ]
            episodes = rollout(
                self.config,
                self.tokenizer,
                1,
                self.eval_dataset.max_steps,
                conversations,
                samples=batch,
                reward_function=self.reward_function,
                vllm_model=self.vllm_model,
            )

            episodes.extend(episodes)

        reward = [episode.reward for episode in episodes]
        mean_reward = sum(reward) / len(reward)
        self.metrics_wrapper.add_scalar("eval/mean_reward", mean_reward, step)

    @property
    def checkpoint_dir(self) -> Path:
        if self.host_type == "modal":
            return Path("/minrl-models/checkpoints")
        return Path("checkpoints")

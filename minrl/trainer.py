import time
from pathlib import Path
from typing import Any, cast

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
from minrl.algorithms import compute_scaled_temperature

from minrl.algorithms import rollout, sync_weights_to_vllm, update_policy
from minrl.constants import DeviceType, Episode, HostType, LoggerChoice, TrainerConfig
from minrl.metrics import MetricsWrapper
from minrl.tasks import TASK_DATASETS
from minrl.tasks.dataset import MinRLDataset
from minrl.utils import clear_memory, compute_metrics, USING_MPS, get_memory_usage

if not USING_MPS:
    from bitsandbytes.optim import Adam8bit  # pyright: ignore[reportMissingImports]
else:
    Adam8bit = torch.optim.AdamW


def get_available_device() -> DeviceType:
    return (
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.mps.is_available()
        else "cpu"
    )


def simple_timestamp() -> str:
    return time.strftime("%m%d_%H%M%S")


class Trainer:
    tokenizer: PreTrainedTokenizerBase
    model: nn.Module

    def __init__(self, host_type: HostType) -> None:
        """Initialize the trainer with configuration."""
        self.config = TrainerConfig()
        device_type = get_available_device()
        self.device_type: DeviceType = device_type
        self.device = torch.device(device_type)
        self.host_type: HostType = host_type
        self.dtype = torch.bfloat16

    def init_model(self):
        """Initialize the model and tokenizer."""
        torch.set_default_device(self.device)
        if self.device_type == "mps":
            logger.warning("vLLM does not support MPS backend, falling back to CPU.")
        set_vllm_use_v1(False)
        torch.random.manual_seed(42)

        # Reduce vLLM memory usage significantly
        get_memory_usage()
        logger.info("Initializing vLLM model")

        self.vllm_model = LLM(
            max_num_seqs=self.config.max_num_seqs,
            model=self.config.model_id,
            gpu_memory_utilization=0.5,
            enforce_eager=True,
            dtype="float16" if USING_MPS else "bfloat16",
            # Prefix caching requires CUDA for some model families (Gemma)
            enable_prefix_caching=self.device_type == "cuda",
        )
        get_memory_usage()
        tokenizer = AutoTokenizer.from_pretrained(self.config.model_id)
        # fallback to eager for mps
        attn_impl = "sdpa" if self.device_type == "mps" else "flash_attention_2"

        # Use more memory-efficient model loading
        logger.info(f"Initializing HF model, attn impl: {attn_impl}")
        model: nn.Module = AutoModelForCausalLM.from_pretrained(  # pyright: ignore[reportAssignmentType]
            self.config.model_id,
            device_map="auto",
            dtype=self.dtype,
            attn_implementation=attn_impl,
            low_cpu_mem_usage=True,
        )
        get_memory_usage()

        # Enable gradient checkpointing to save memory during backward pass
        if self.config.use_gradient_checkpointing:
            if hasattr(model, "gradient_checkpointing_enable"):
                model.gradient_checkpointing_enable()
                logger.info("Gradient checkpointing enabled")
            else:
                logger.warning("Gradient checkpointing not available for this model")

        logger.info("Model loaded.")
        logger.info(f"Using device {self.device}, attn impl {attn_impl}")
        self.tokenizer = tokenizer
        self.model = model

        if self.config.optimizer == "adamw":
            if (
                self.config.use_low_precision_optimizer_if_available
                and self.device_type == "cuda"
            ):
                self.optimizer = Adam8bit(
                    cast(nn.Module, self.model).parameters(),
                    lr=self.config.lr,
                    betas=(0.9, 0.999),
                    eps=1e-8,
                )
            else:
                self.optimizer = torch.optim.AdamW(
                    cast(nn.Module, self.model).parameters(), lr=self.config.lr
                )
        else:
            raise ValueError(f"Invalid optimizer choice: {self.config.optimizer}")

        logger.info(f"Using optimizer: {self.optimizer}")

    def init_training(self, logger_choice: LoggerChoice) -> None:
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
        self.train_dataloader = DataLoader(
            self.train_dataset,
            shuffle=True,
            generator=generator,
            batch_size=self.config.groups_per_batch,
            collate_fn=lambda x: x,
            pin_memory=False,  # Disable pin_memory to save memory
            num_workers=0,  # Use single process to avoid memory overhead
        )
        self.start_time = time.time()
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.run_name = f"{self.config.model_display_name}-{self.config.algorithm}-{self.config.task}-{simple_timestamp()}"
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

            episodes, rollout_duration = rollout(
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
            self.metrics_wrapper.add_scalar(
                "timing/train_rollout_duration_sec", rollout_duration, step
            )

            logger.info(f"Updating policy for step {step}")

            results = update_policy(
                model=cast(nn.Module, self.model),
                optimizer=self.optimizer,
                episodes=episodes,
                tokenizer=self.tokenizer,
                micro_batch_size=self.config.micro_batch_size,
                pad_token_id=int(cast(Any, self.tokenizer.pad_token_id)),
                max_grad_norm=self.config.max_grad_norm,
                device=self.device,
                algorithm=self.config.algorithm,
                entropy_coef=self.config.entropy_coef,
            )
            self.metrics_wrapper.add_scalar(
                "timing/update_policy_duration_sec", results["duration"], step
            )
            sync_weights_to_vllm(cast(nn.Module, self.model), self.vllm_model)

            # Compute current reward std for next iteration before clearing memory
            current_rewards = [episode.reward for episode in episodes]
            current_reward_std = float(np.std(current_rewards))

            # Get temperature used for logging
            temperature_used = compute_scaled_temperature(self.config, prev_reward_std)

            compute_metrics(
                episodes,
                results,  # pyright: ignore[reportArgumentType]
                self.metrics_wrapper,
                step,
                self.optimizer,
                temperature_used,
            )

            # Clear memory after each step more aggressively
            del episodes
            clear_memory()

            # Log memory usage after clearing
            get_memory_usage()
            if step % self.config.eval_interval == 0:
                self.evaluate(step)

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
            batch_episodes, rollout_duration = rollout(
                self.config,
                self.tokenizer,
                1,
                self.eval_dataset.max_steps,
                conversations,
                samples=batch,
                reward_function=self.reward_function,
                vllm_model=self.vllm_model,
            )

            self.metrics_wrapper.add_scalar(
                "timing/eval_rollout_duration_sec", rollout_duration, step
            )

            episodes.extend(batch_episodes)

        reward = [episode.reward for episode in episodes]
        mean_reward = sum(reward) / len(reward)
        self.metrics_wrapper.add_scalar("eval/mean_reward", mean_reward, step)

    @property
    def checkpoint_dir(self) -> Path:
        if self.host_type == "modal":
            return Path("/minrl-models/checkpoints")
        return Path("checkpoints")

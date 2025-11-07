import time
from pathlib import Path
from typing import Any, cast, Literal

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
from minrl.lora import apply_lora_to_model
from minrl.algorithms import rollout, sync_weights_to_vllm, update_policy
from minrl.constants import DeviceType, Episode, HostType, LoggerChoice, TrainerConfig
from minrl.metrics import MetricsWrapper
from minrl.tasks import TASK_DATASETS
from minrl.tasks.dataset import MinRLDataset
from minrl.utils import (
    clear_memory,
    compute_metrics,
    USING_MPS,
    log_memory_usage,
)

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

    def __init__(self, host_type: HostType, logger_choice: LoggerChoice) -> None:
        """Initialize the trainer with configuration."""
        self.config = TrainerConfig()
        device_type = get_available_device()
        self.device_type: DeviceType = device_type
        self.device = torch.device(device_type)
        self.host_type: HostType = host_type
        self.dtype = torch.bfloat16
        self.run_name = f"{self.config.model_display_name}-{self.config.algorithm}-{self.config.task}-{simple_timestamp()}"
        self.metrics_wrapper = MetricsWrapper(
            logger_choice, self.config.task, self.config, self.run_name
        )
        logger.info(f"Logging to: {logger_choice}")

    def _setup_hf_model(self):
        torch.set_default_device(self.device)
        if self.device_type == "mps":
            logger.warning("vLLM does not support MPS backend, falling back to CPU.")
        set_vllm_use_v1(False)
        torch.random.manual_seed(42)

        # Reduce vLLM memory usage significantly
        log_memory_usage("pre_init_model", metrics_wrapper=self.metrics_wrapper, step=0)
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
        log_memory_usage("init_model", metrics_wrapper=self.metrics_wrapper, step=0)

        # Enable gradient checkpointing to save memory during backward pass
        if self.config.use_gradient_checkpointing:
            if hasattr(model, "gradient_checkpointing_enable"):
                model.gradient_checkpointing_enable()
                logger.info("Gradient checkpointing enabled")
            else:
                logger.warning("Gradient checkpointing not available for this model")

        # Apply torch.compile for faster training
        if self.config.use_torch_compile:
            logger.info(f"Compiling model with mode: {self.config.torch_compile_mode}")
            model = torch.compile(model, mode=self.config.torch_compile_mode)  # type: ignore
            logger.info("torch.compile applied to model")

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
                    lr=self.config.learning_rate,
                    betas=(0.9, 0.999),
                    eps=1e-8,
                )
            else:
                # Use fused AdamW for faster optimizer step if available and enabled
                use_fused = (
                    self.config.use_fused_optimizer and self.device_type == "cuda"
                )
                self.optimizer = torch.optim.AdamW(
                    cast(nn.Module, self.model).parameters(),
                    lr=self.config.learning_rate,
                    fused=use_fused,
                )
        else:
            raise ValueError(f"Invalid optimizer choice: {self.config.optimizer}")

        logger.info(f"Using optimizer: {self.optimizer}")

        if self.config.lora_config is not None:
            logger.info("Applying LoRA to model")
            apply_lora_to_model(self.model, self.config.lora_config)
            logger.info("LoRA applied to model")

    def init_model(self):
        """Initialize the model and tokenizer."""
        logger.info("Initializing vLLM model")

        self._setup_hf_model()

        self.vllm_model = LLM(
            max_num_seqs=self.config.max_num_seqs,
            model=self.config.model_id,
            gpu_memory_utilization=self.config.vllm_gpu_memory_utilization,
            enforce_eager=True,
            dtype="float16" if USING_MPS else "bfloat16",
            # Prefix caching requires CUDA for some model families (Gemma)
            enable_prefix_caching=self.config.enable_prefix_caching
            and self.device_type == "cuda",
            # Enable sleep mode for memory management
            enable_sleep_mode=self.config.enable_sleep_mode
            and self.device_type == "cuda",
            max_model_len=self.config.max_seq_length,
            logprobs_mode="processed_logprobs",
        )
        log_memory_usage(
            "init_vllm_model", metrics_wrapper=self.metrics_wrapper, step=0
        )

    def init_training(self) -> None:
        """Initialize training components including dataloader, optimizer, and logging."""
        assert self.tokenizer is not None, "Tokenizer not initialized"
        dataset_cls: type[MinRLDataset] = TASK_DATASETS[self.config.task]
        self.reward_function = dataset_cls.reward_function
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
            drop_last=True,  # Ensure full batches to keep group inference consistent
        )
        self.start_time = time.time()
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

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
            step_start_time = time.time()
            logger.info(f"Starting rollout for step {step}")

            self.metrics_wrapper.add_scalar(
                "train/n_samples_in_batch", len(batch), step
            )

            self._wake_sleep_vllm("wake")

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

            self._wake_sleep_vllm("sleep")

            # Log GPU utilization after rollout
            log_memory_usage("rollout", metrics_wrapper=self.metrics_wrapper, step=step)

            # Clean up conversations to free memory
            del conversations
            clear_memory()

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
                metrics_wrapper=self.metrics_wrapper,
                step=step,
                gradient_accumulation_steps=self.config.gradient_accumulation_steps,
            )
            self.metrics_wrapper.add_scalar(
                "timing/update_policy_duration_sec", results["duration"], step
            )

            # Log GPU utilization after update_policy
            log_memory_usage(
                "update_policy", metrics_wrapper=self.metrics_wrapper, step=step
            )
            self._wake_sleep_vllm("wake")

            sync_weights_to_vllm(
                cast(nn.Module, self.model),
                self.vllm_model,
            )

            # Reset prefix cache after weight sync - cached KV states are invalid with new weights
            if self.config.enable_prefix_caching:
                self.vllm_model.llm_engine.reset_prefix_cache()

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
            del episodes, batch
            clear_memory()
            # Log memory usage after clearing
            log_memory_usage(
                "end_of_step", metrics_wrapper=self.metrics_wrapper, step=step
            )
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

            # Log total step duration
            total_step_duration = time.time() - step_start_time
            self.metrics_wrapper.add_scalar(
                "timing/total_step_duration_sec", total_step_duration, step
            )
            logger.info(f"Step {step} completed in {total_step_duration:.2f}s")

        self.metrics_wrapper.close()

    def _wake_sleep_vllm(self, action: Literal["wake", "sleep"]) -> None:
        if self.config.enable_sleep_mode and self.device_type == "cuda":
            if action == "wake":
                self.vllm_model.wake_up()
                if self.config.enable_prefix_caching:
                    self.vllm_model.llm_engine.reset_prefix_cache()
            elif action == "sleep":
                # https://github.com/vllm-project/vllm/issues/17103
                if self.config.enable_prefix_caching:
                    self.vllm_model.llm_engine.reset_prefix_cache()
                self.vllm_model.sleep(level=1)
        else:
            logger.info("Sleep mode is disabled, skipping sleep")

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

        self._wake_sleep_vllm("wake")

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
                self.config.group_size,
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

            # Clean up batch episodes after extending
            del batch_episodes
            clear_memory()

        reward = [episode.reward for episode in episodes]
        if len(reward) == 0:
            logger.warning("No episodes found in evaluation")
            mean_reward = 0.0
        else:
            mean_reward = sum(reward) / len(reward)
        self.metrics_wrapper.add_scalar("eval/num_episodes", len(episodes), step)
        self.metrics_wrapper.add_scalar("eval/mean_reward", mean_reward, step)

        # Clean up episodes after evaluation
        del episodes, reward, conversations
        clear_memory()

    @property
    def checkpoint_dir(self) -> Path:
        if self.host_type == "modal":
            return Path("/minrl-models/checkpoints")
        return Path("checkpoints")

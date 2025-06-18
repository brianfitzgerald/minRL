import time
from pathlib import Path
from typing import Any, Optional, cast, TYPE_CHECKING
from vllm import LLM

import torch
import torch.nn as nn
from loguru import logger
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader
from transformers.models.auto.modeling_auto import AutoModelForCausalLM
from transformers.models.auto.tokenization_auto import AutoTokenizer
from transformers.tokenization_utils_base import PreTrainedTokenizerBase
from minrl.constants import LoggerChoice, HostType, TrainerConfig
from minrl.metrics import MetricsWrapper
from minrl.tasks import TASK_DEFINITIONS

from minrl.algorithms import compute_metrics, rollout, update_policy
from minrl.tasks.dataset import MinRLDataset
from vllm.envs import set_vllm_use_v1

import wandb


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
        vllm_device = self.device.type
        set_vllm_use_v1(False)
        if self.device.type == "mps":
            logger.warning("vLLM does not support MPS backend, falling back to CPU.")
            vllm_device = "cpu"
        self.vllm_model = LLM(
            model=self.config.model_id,
            device=vllm_device,
            gpu_memory_utilization=0.2,
            max_model_len=self.config.max_new_tokens,
            max_seq_len_to_capture=self.config.max_new_tokens,
            enforce_eager=True,
        )
        tokenizer = AutoTokenizer.from_pretrained(self.config.model_id)
        attn_impl = "flash_attention_2" if self.device.type == "cuda" else "sdpa"
        model = AutoModelForCausalLM.from_pretrained(
            self.config.model_id,
            device_map="auto",
            torch_dtype=self.dtype,
            attn_implementation=attn_impl,
        )

        logger.info("Model loaded.")
        model.to(self.device)
        torch.set_default_device(self.device)
        logger.info(f"Using device {self.device}, attn impl {attn_impl}")
        torch.random.manual_seed(42)
        self.tokenizer = tokenizer
        self.model = model

    def init_training(self) -> None:
        """Initialize training components including dataloader, optimizer, and logging."""
        assert self.tokenizer is not None, "Tokenizer not initialized"
        dataset_cls: type[MinRLDataset] = TASK_DEFINITIONS[self.config.task]["dataset"]
        self.train_dataset = dataset_cls(
            split="train", host=self.host_type, tokenizer=self.tokenizer
        )
        self.eval_dataset = dataset_cls(
            split="eval", host=self.host_type, tokenizer=self.tokenizer
        )
        generator = torch.Generator(device=self.device)
        self.train_dataloader = DataLoader(
            self.train_dataset,
            shuffle=True,
            collate_fn=self.train_dataset.collate_fn,
            generator=generator,
            batch_size=1,
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
        self.ckpt_dir = Path("checkpoints")
        self.ckpt_dir.mkdir(parents=True, exist_ok=True)
        self.run_name = f"{self.config.model_display_name}-{self.config.algorithm}-{self.config.task}-{simple_timestamp()}"
        logger_choice: LoggerChoice = (
            "wandb" if self.host_type == "modal" else self.config.logger_choice
        )
        logger.info(f"Logging to: {logger_choice}")
        self.metrics_wrapper = MetricsWrapper(logger_choice, self.run_name)

    def train(self) -> None:
        """Run the main training loop.

        For each step:
            1. Performs rollout to generate episodes
            2. Updates policy using GRPO
            3. Evaluates model periodically
            4. Saves checkpoints periodically
            5. Logs metrics to TensorBoard or Weights & Biases
        """

        for step, batch in enumerate(self.train_dataloader, start=1):
            logger.info(f"Starting rollout for step {step}")

            assert self.model is not None
            assert self.tokenizer is not None

            episodes = rollout(
                tokenizer=self.tokenizer,
                batch=batch,
                max_new_tokens=self.config.max_new_tokens,
                num_answer_per_question=self.config.num_answers_per_question,
                reward_function=TASK_DEFINITIONS[self.config.task]["reward_function"],
                vllm_model=self.vllm_model,
            )
            if self.config.skip_unfinished_episodes:
                episodes = [episode for episode in episodes if episode.is_finished]
            logger.info(f"Updating policy for step {step}")

            # Update policy - compute loss and perform backward pass
            results = update_policy(
                model=cast(nn.Module, self.model),
                optimizer=self.optimizer,
                episodes=episodes,
                micro_batch_size=self.config.micro_batch_size,
                pad_token_id=int(cast(Any, self.tokenizer.pad_token_id)),
                max_grad_norm=self.config.max_grad_norm,
                device=self.device,
                vllm_model=self.vllm_model,
                algorithm=self.config.algorithm,
            )
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            if step % self.config.eval_interval == 0:
                eval_success_rate = self.evaluate()
                logger.info(f"\rEval success rate: {eval_success_rate:.2f}" + " " * 100)
                self.metrics_wrapper.add_scalar(
                    "success_rate/eval", eval_success_rate, step
                )

            compute_metrics(
                episodes, results, self.metrics_wrapper, step, self.optimizer
            )
            # save checkpoint
            if step % self.config.ckpt_save_interval == 0:
                logger.info(f"Saving checkpoint for step {step}")
                output_file = self.ckpt_dir / f"{self.run_name}_step_{step:06d}"
                self.model.save_pretrained(output_file)  # type: ignore
                logger.info(f"Saved checkpoint to {output_file}")

        self.metrics_wrapper.close()

    def evaluate(self) -> float:
        """Evaluate the current model.

        Returns:
            float: The evaluation success rate (currently returns 0.0 as placeholder)
        """
        # TODO: Implement evaluation logic
        return 0.0

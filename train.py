import time
from pathlib import Path
from typing import Optional, cast, Any

import fire
import torch
import torch.nn as nn
from loguru import logger
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader
from transformers.models.auto.modeling_auto import AutoModelForCausalLM
from transformers.models.auto.tokenization_auto import AutoTokenizer
from transformers.tokenization_utils import PreTrainedTokenizer
from tasks.connections import ConnectionsDataset, connections_reward_func, create_connections_datasets, Tokenizer
from minrl.grpo import compute_metrics, rollout, update_policy
from vllm_inference.client import VLLMClient
from pydantic import BaseModel
from typing import Literal

USING_MPS = torch.backends.mps.is_available() and torch.backends.mps.is_built()
if not USING_MPS:
    from bitsandbytes.optim import Adam8bit


def get_available_device() -> str:
    """Get the best available device for training.

    Returns:
        str: Device string ('cuda:0', 'mps', or 'cpu')
    """
    return (
        "cuda:0"
        if torch.cuda.is_available()
        else "mps" if torch.mps.is_available() else "cpu"
    )


OptimizerChoice = Literal["adamw", "adamw_8bit"]


class TrainerConfig(BaseModel):
    model_id: str = "Qwen/Qwen3-0.6B"
    eval_interval: int = 100
    num_answer_per_question: int = 2
    max_new_tokens: int = 256
    micro_batch_size: int = 2
    max_grad_norm: float = 1.0
    ckpt_save_interval: int = 100
    skip_unfinished_episodes: bool = False
    optimizer: OptimizerChoice = "adamw"


class Trainer:

    tokenizer: PreTrainedTokenizer | None = None
    model: AutoModelForCausalLM | None = None

    def __init__(
        self, config: Optional[TrainerConfig] = None
    ) -> None:
        """Initialize the trainer with configuration."""
        self.config = config or TrainerConfig()
        self.device = torch.device(get_available_device())
        self.dtype = torch.bfloat16

    def init_model(self):
        """Initialize the model and tokenizer."""
        tokenizer = AutoTokenizer.from_pretrained(self.config.model_id)
        attn_impl = "flash_attention_2" if self.device.type == "cuda" else "flex_attention"
        model = AutoModelForCausalLM.from_pretrained(
            self.config.model_id,
            device_map="auto",
            torch_dtype=self.dtype,
            attn_implementation=attn_impl,
        )

        logger.info("Model loaded.")
        model.to(self.device)
        logger.info(f"Using device {self.device}")
        torch.set_default_device(self.device)
        torch.random.manual_seed(42)
        self.tokenizer = tokenizer
        self.model = model

    def init_training(self) -> None:
        """Initialize training components including dataloader, optimizer, and logging."""
        self.train_dataset, _ = create_connections_datasets(
            cast(Tokenizer, self.tokenizer),
            jsonl_path="data/train_prompts.jsonl",
            num_samples=1000
        )
        generator = torch.Generator(device=self.device)
        self.train_dataloader = DataLoader(
            self.train_dataset,
            shuffle=True,
            collate_fn=ConnectionsDataset.collate_fn,
            generator=generator,
            batch_size=1,
        )

        if self.config.optimizer == "adamw":
            self.optimizer = torch.optim.AdamW(
                cast(nn.Module, self.model).parameters(), lr=1e-5
            )
        elif self.config.optimizer == "adamw_8bit":
            self.optimizer = Adam8bit(
                cast(nn.Module, self.model).parameters(),
                lr=1e-5,
                betas=(0.9, 0.999),
                eps=1e-8,
            )
        self.start_time = time.time()
        self.ckpt_dir = Path("ckpts")
        self.ckpt_dir.mkdir(parents=True, exist_ok=True)
        self.tb_writer = SummaryWriter()
        self.client = VLLMClient()
        self.client.reset_prefix_cache()

    def train(self) -> None:
        """Run the main training loop.

        For each step:
            1. Performs rollout to generate episodes
            2. Updates policy using GRPO
            3. Evaluates model periodically
            4. Saves checkpoints periodically
            5. Logs metrics to TensorBoard
        """
        for step, batch in enumerate(self.train_dataloader, start=1):
            logger.info(f"Starting rollout for step {step}")

            assert self.model is not None
            assert self.tokenizer is not None

            episodes = rollout(
                model=self.model,
                tokenizer=self.tokenizer,
                batch=batch,
                max_new_tokens=self.config.max_new_tokens,
                num_answer_per_question=self.config.num_answer_per_question,
                reward_function=connections_reward_func,
                device=self.device,
                client=self.client,
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
                dtype=self.dtype,
            )
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            if step % self.config.eval_interval == 0:
                eval_success_rate = self.evaluate()
                print(f"\rEval success rate: {eval_success_rate:.2f}" + " " * 100)
                self.tb_writer.add_scalar("success_rate/eval", eval_success_rate, step)

            compute_metrics(episodes, results, self.tb_writer, step, self.optimizer)
            # save checkpoint
            if step % self.config.ckpt_save_interval == 0:
                output_file = self.ckpt_dir / f"ckpt_{step:06d}.pt"
                torch.save(cast(nn.Module, self.model).state_dict(), output_file)
                print(f"Saved checkpoint to {output_file}")

    def evaluate(self) -> float:
        """Evaluate the current model.

        Returns:
            float: The evaluation success rate (currently returns 0.0 as placeholder)
        """
        # TODO: Implement evaluation logic
        return 0.0


def main(model_id: str = "Qwen/Qwen3-0.6B") -> None:
    """Main entry point for training.

    Args:
        model_id: The HuggingFace model ID to use for training
    """
    config = TrainerConfig(model_id=model_id)
    trainer = Trainer(config)
    trainer.init_model()
    trainer.init_training()
    trainer.train()


if __name__ == "__main__":
    fire.Fire(main)

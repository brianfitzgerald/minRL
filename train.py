import time
from dataclasses import dataclass
from pathlib import Path

import fire
import torch
import torch.nn as nn
from loguru import logger
from sympy import evaluate
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader, Dataset
from transformers.models.auto.modeling_auto import AutoModelForCausalLM
from transformers.models.auto.tokenization_auto import AutoTokenizer
from transformers.models.qwen3 import Qwen3ForCausalLM

from dataset import ConnectionsDataset, create_connections_datasets
from grpo import compute_metrics, rollout, update_policy
from tasks.countdown import reward_function


def get_available_device() -> str:
    return (
        "cuda:0"
        if torch.cuda.is_available()
        else "mps"
        if torch.mps.is_available()
        else "cpu"
    )


def init_model(
    model_id: str = "Qwen/Qwen3-0.6B",
) -> tuple[AutoTokenizer, AutoModelForCausalLM]:
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        ignore_mismatched_sizes=True,
    )

    logger.info("Model loaded.")
    device = get_available_device()
    model.to(device)
    logger.info(f"Using device {device}")
    torch.set_default_device(device)
    torch.random.manual_seed(42)
    return tokenizer, model


@dataclass
class Config:
    model_id: str = "Qwen2.5-3B-Instruct"
    eval_interval: int = 100
    num_answer_per_question: int = 2
    max_new_tokens: int = 512
    micro_batch_size: int = 2
    max_grad_norm: float = 1.0
    eval_interval: int = 100
    ckpt_save_interval: int = 100
    skip_unfinished_episodes: bool = False


def training_loop(
    model: Qwen3ForCausalLM,
    tokenizer: AutoTokenizer,
    train_dataset: ConnectionsDataset,
):
    to = init_training(model, train_dataset)

    for step, batch in enumerate(to.train_dataloader, start=1):
        logger.info(f"Starting rollout for step {step}")

        episodes = rollout(
            model=model,
            tokenizer=tokenizer,
            batch=batch,
            max_new_tokens=to.config.max_new_tokens,
            num_answer_per_question=to.config.num_answer_per_question,
            reward_function=reward_function,
            device=to.device,
        )
        if to.config.skip_unfinished_episodes:
            episodes = [episode for episode in episodes if episode.is_finished]
        logger.info(f"Updating policy for step {step}")

        # Update policy - compute loss and perform backward pass
        results = update_policy(
            model=model,
            optimizer=to.optimizer,
            episodes=episodes,
            micro_batch_size=to.config.micro_batch_size,
            pad_token_id=tokenizer.pad_token_id,
            max_grad_norm=to.config.max_grad_norm,
            device=to.device,
            dtype=to.dtype,
        )
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        if step % to.config.eval_interval == 0:
            eval_success_rate = evaluate(model, tokenizer, to.device, to.dtype, to.config)
            print(f"\rEval success rate: {eval_success_rate:.2f}" + " " * 100)
            to.tb_writer.add_scalar("success_rate/eval", eval_success_rate, step)

        compute_metrics(episodes, results, to.tb_writer, step, to.optimizer)
        # save checkpoint
        if step % to.config.ckpt_save_interval == 0:
            output_file = to.ckpt_dir / f"ckpt_{step:06d}.pt"
            torch.save(model.state_dict(), output_file)
            print(f"Saved checkpoint to {output_file}")


@dataclass
class TrainingObjects:
    train_dataloader: DataLoader
    optimizer: torch.optim.Optimizer
    start_time: float
    ckpt_dir: Path
    dtype: torch.dtype
    tb_writer: SummaryWriter
    config: Config
    device: torch.device

def init_training(model: nn.Module, train_dataset: Dataset) -> TrainingObjects:
    device = torch.device(get_available_device())
    generator = torch.Generator(device=device)
    train_dataloader = DataLoader(
        train_dataset,
        shuffle=True,
        collate_fn=ConnectionsDataset.collate_fn,
        generator=generator,
        batch_size=1,
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)
    start_time = time.time()
    ckpt_dir = Path("ckpts")
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    dtype = torch.bfloat16
    tb_writer = SummaryWriter()
    config = Config()
    return TrainingObjects(
        train_dataloader=train_dataloader,
        optimizer=optimizer,
        start_time=start_time,
        ckpt_dir=ckpt_dir,
        dtype=dtype,
        tb_writer=tb_writer,
        config=config,
        device=device,
    )


def main(model_id: str = "Qwen/Qwen3-0.6B"):
    tokenizer, model = init_model(model_id)
    train_dataset, _ = create_connections_datasets(tokenizer)
    training_loop(model, tokenizer, train_dataset)


if __name__ == "__main__":
    fire.Fire(main)

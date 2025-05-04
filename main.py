import html
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import fire
import numpy as np
import torch
from loguru import logger
from sympy import evaluate
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedTokenizer

from dataset import ConnectionsDataset, create_connections_datasets
from grpo import rollout, update_policy
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
    model_id: str = "Qwen/Qwen2.5-0.5B-Instruct",
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
    num_answer_per_question: int = 1
    max_gen_len: int = 100
    micro_batch_size: int = 1
    max_grad_norm: float = 1.0
    eval_interval: int = 100
    ckpt_save_interval: int = 100
    skip_unfinished_episodes: bool = False


def training_loop(
    model: AutoModelForCausalLM,
    tokenizer: PreTrainedTokenizer,
    train_dataset: ConnectionsDataset,
    device: torch.device,
):
    training_objects = init_training(model, train_dataset, device)

    for step, batch in enumerate(training_objects.train_dataloader, start=1):
        logger.info(f"Starting rollout for step {step}")

        episodes = rollout(
            model=model,
            tokenizer=tokenizer,
            batch=batch,
            max_gen_len=training_objects.config.max_gen_len,
            num_answer_per_question=training_objects.config.num_answer_per_question,
            reward_function=reward_function,
            device=device,
        )
        if training_objects.config.skip_unfinished_episodes:
            episodes = [episode for episode in episodes if episode.is_finished]
        logger.info(f"Updating policy for step {step}")

        # Update policy - compute loss and perform backward pass
        results = update_policy(
            model=model,
            optimizer=training_objects.optimizer,
            episodes=episodes,
            micro_batch_size=training_objects.config.micro_batch_size,
            pad_token_id=tokenizer.pad_token_id,
            max_grad_norm=training_objects.config.max_grad_norm,
            device=device,
            dtype=training_objects.dtype,
        )
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        end_time = time.time()
        duration = end_time - training_objects.start_time

        # compute and log important metrics
        reward = [episode.reward for episode in episodes]
        formatted_reward = [
            episode.reward_info["format_reward"] for episode in episodes
        ]
        answer_reward = [episode.reward_info["answer_reward"] for episode in episodes]
        num_finished_episodes = sum(episode.is_finished for episode in episodes)
        mean_reward = np.mean(reward)
        std_reward = np.std(reward)
        success_rate = np.mean(answer_reward)
        format_reward = np.mean(formatted_reward)
        grad_norm = results["grad_norm"]
        entropy = results["entropy"]
        lr = training_objects.optimizer.param_groups[0]["lr"]
        loss = results["loss"]
        mean_response_len = np.mean(
            [len(episode.generated_token_ids) for episode in episodes]
        )

        print(
            f"\rStep {step}, mean_reward: {mean_reward:.2f}, "
            f"train success_rate: {success_rate:.2f}, "
            f"grad_norm: {grad_norm:.2f}, duration: {duration:.2f}, "
            f"num_finished_episodes: {num_finished_episodes}, "
            f"mean_response_len: {mean_response_len:.2f}, "
            f"entropy: {entropy:.2f}"
        )
        if step % training_objects.config.eval_interval == 0:
            eval_success_rate = evaluate(model, tokenizer, device, dtype, training_objects.config)
            print(f"\rEval success rate: {eval_success_rate:.2f}" + " " * 100)
            training_objects.tb_writer.add_scalar("success_rate/eval", eval_success_rate, step)

        tb_writer = training_objects.tb_writer

        tb_writer.add_scalar("loss", loss, step)
        tb_writer.add_scalar("mean_reward", mean_reward, step)
        tb_writer.add_scalar("std_reward", std_reward, step)
        tb_writer.add_scalar("success_rate/train", success_rate, step)
        tb_writer.add_scalar("format_reward", format_reward, step)
        tb_writer.add_scalar("grad_norm", grad_norm, step)
        tb_writer.add_scalar("duration", duration, step)
        tb_writer.add_scalar("num_finished_episodes", num_finished_episodes, step)
        tb_writer.add_scalar("learning_rate", lr, step)
        tb_writer.add_scalar("mean_response_len", mean_response_len, step)
        tb_writer.add_scalar("entropy", entropy, step)
        for i, episode in enumerate(episodes):
            # TensorBoard treats text as markdown.
            text = html.escape(episode.text)
            tb_writer.add_text(f"text_{i}", f"<pre>{text}</pre>", step)

        # save checkpoint
        if step % config.ckpt_save_interval == 0:
            output_file = ckpt_dir / f"ckpt_{step:06d}.pt"
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

def init_training(model, train_dataset, device):
    logger.info("Starting training loop")
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
    logger.info("Training loop initialized")
    return TrainingObjects(
        train_dataloader=train_dataloader,
        optimizer=optimizer,
        start_time=start_time,
        ckpt_dir=ckpt_dir,
        dtype=dtype,
        tb_writer=tb_writer,
        config=config
    )


def main(model_id):
    tokenizer, model = init_model(model_id)
    train_dataset, _ = create_connections_datasets(tokenizer)
    device = get_available_device()
    training_loop(model, tokenizer, train_dataset, device)


if __name__ == "__main__":
    fire.Fire(main)

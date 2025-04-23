import html
from pathlib import Path
from time import time

import fire
import numpy as np
import torch
from loguru import logger
from sympy import evaluate
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer

from dataset import ConnectionsDataset, create_connections_datasets
from grpo import rollout
from dataclasses import dataclass

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
    model_id: str = "Qwen2.5-3B-Instruct",
) -> tuple[AutoTokenizer, AutoModelForCausalLM]:
    logger.info(f"Loading model {model_id}")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id)
    logger.info("Model loaded.")
    device = get_available_device()
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


def main(model_id: str = "Qwen2.5-3B-Instruct", eval_interval: int = 100):
    tokenizer, model = init_model(model_id)
    train_dataset, val_dataset = create_connections_datasets()
    device = get_available_device()
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

    for step, batch in enumerate(train_dataloader, start=1):
        episodes = rollout(
            model=model,
            tokenizer=tokenizer,
            batch=batch,
            max_gen_len=config.max_gen_len,
            num_answer_per_question=config.num_answer_per_question,
            reward_function=reward_function,
            device=device,
            dtype=dtype,
        )
        if config["training"]["skip_unfinished_episodes"]:
            episodes = [episode for episode in episodes if episode.is_finished]
        results = update_policy(
            model=model,
            optimizer=optimizer,
            episodes=episodes,
            micro_batch_size=config["training"]["micro_batch_size"],
            pad_token_id=tokenizer.pad_token_id,
            max_grad_norm=config["training"]["max_grad_norm"],
            device=device,
            dtype=dtype,
        )
        torch.cuda.synchronize()
        end_time = time.time()
        duration = end_time - start_time
        start_time = end_time

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
        lr = optimizer.param_groups[0]["lr"]
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
        if step % eval_interval == 0:
            eval_success_rate = evaluate(model, tokenizer, device, dtype, config)
            print(f"\rEval success rate: {eval_success_rate:.2f}" + " " * 100)
            tb_writer.add_scalar("success_rate/eval", eval_success_rate, step)

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
        if step % config["training"]["ckpt_save_interval"] == 0:
            output_file = ckpt_dir / f"ckpt_{step:06d}.pt"
            torch.save(model.state_dict(), output_file)
            print(f"Saved checkpoint to {output_file}")




if __name__ == "__main__":
    fire.Fire(main)

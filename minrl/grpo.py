import dataclasses
import gc
from collections import defaultdict
from pydoc import html
from typing import Dict, List
from tensorboardX import SummaryWriter
from vllm.sampling_params import (
    SamplingParams,
)

import numpy as np
import torch
import torch.nn as nn
from loguru import logger
from transformers.models.auto.tokenization_auto import AutoTokenizer
from transformers.tokenization_utils_base import PreTrainedTokenizerBase
from vllm.worker.model_runner_base import ModelRunnerBase

from minrl.tasks import RewardFunction
from minrl.tasks.dataset import Episode, MiniBatch
from minrl.constants import TrainerConfig
from vllm import LLM

debug_tokenizer = AutoTokenizer.from_pretrained(
    TrainerConfig().model_id, use_fast=False
)


def logprob_dict_to_logprobs(
    logprobs: list[list[dict[int, float]]], vocab_size: int
) -> torch.Tensor:
    """Convert from vLLM format to logprobs.
    vLLM returns a list of dicts, each containing a token and its logprob.
    We convert this to a tensor of shape (batch_size, seq_len, vocab_size)
    """
    all_probs = []
    for seq in logprobs:
        seq_probs = []
        for token in seq:
            tokens_sorted = sorted(token.items(), key=lambda x: int(x[1]), reverse=True)
            out_tensor = torch.zeros(vocab_size)
            for token_idx, prob in tokens_sorted:
                out_tensor[int(token_idx)] = prob
            seq_probs.append(out_tensor)
        all_probs.append(torch.stack(seq_probs))
    return torch.stack(all_probs)


@torch.no_grad()
def rollout(
    tokenizer: PreTrainedTokenizerBase,
    batch: MiniBatch,
    max_new_tokens: int,
    num_answer_per_question: int,
    reward_function: RewardFunction,
    vllm_model: LLM,
) -> List[Episode]:
    """Generate multiple responses for each prompt in the batch."""
    end_token_id = tokenizer.eos_token_id
    pad_token_id = tokenizer.pad_token_id

    # Convert to tensor and move to device

    logger.info(
        f"Generating responses for {len(batch.prefixes)} prompts, max_tokens={max_new_tokens}"
    )
    prefixes_batch: list[str] = [
        batch.prefixes[i]
        for i in range(len(batch.prefixes))
        for _ in range(num_answer_per_question)
    ]

    outputs = vllm_model.generate(
        prefixes_batch, sampling_params=SamplingParams(max_tokens=max_new_tokens)
    )
    # Clear CUDA cache
    gc.collect()
    torch.cuda.empty_cache()

    # Process outputs and create episodes
    episodes: List[Episode] = []
    for i in range(len(batch.prefixes)):
        for j in range(num_answer_per_question):
            # idx of the j-th answer for the i-th prompt
            idx = i * num_answer_per_question + j

            generated_token_ids: list[int] = list(outputs[idx].outputs[0].token_ids)

            # Remove padding tokens
            if pad_token_id in generated_token_ids:
                generated_token_ids = generated_token_ids[
                    : generated_token_ids.index(
                        pad_token_id  # type: ignore
                    )  # type: ignore
                ]

            generated_text = tokenizer.decode(
                generated_token_ids, skip_special_tokens=True
            )

            logger.info(generated_text)

            # Calculate rewards
            rewards = reward_function(
                response=generated_text,
                sample=batch.samples[i],
            )

            # Create episode
            episode = Episode(
                prefix=batch.prefixes[i],
                text=batch.prefixes[i] + generated_text,
                prefix_token_ids=batch.prefix_token_ids[i],
                generated_token_ids=generated_token_ids,
                is_finished=end_token_id in generated_token_ids,
                reward=rewards,
                reward_info={},
            )
            episodes.append(episode)

    return episodes


def compute_entropy(logits: torch.Tensor) -> torch.Tensor:
    """
    Compute entropy of logits.
    This is defined as -sum(p(x) * log(p(x))) for all x in the logits.
    """
    probs = nn.functional.softmax(logits, dim=-1)
    entropy = -torch.sum(probs * torch.log(probs), dim=-1)
    return entropy


def normalize_rewards_per_group(episodes: List[Episode]) -> List[Episode]:
    """
    Normalize rewards per group.
    This is done by subtracting the mean and dividing by the standard deviation of the rewards in each group.
    If std is 0, returns the original rewards.
    """
    # Group episodes by prefix and calculate stats
    groups: Dict[str, List[Episode]] = defaultdict(list)
    for episode in episodes:
        groups[episode.prefix].append(episode)

    # Normalize rewards within each group
    normalized_episodes: List[Episode] = []
    for group in groups.values():
        rewards = torch.tensor([e.reward for e in group])
        mean = rewards.mean()
        std = rewards.std()
        if torch.isnan(std):
            std = 0

        # Handle case where std is 0 to avoid division by zero
        if std == 0:
            normalized_rewards = rewards - mean
        else:
            normalized_rewards = (rewards - mean) / std

        for episode, norm_reward in zip(group, normalized_rewards):
            normalized_episodes.append(
                dataclasses.replace(episode, reward=float(norm_reward))
            )

    return normalized_episodes


def update_policy(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    episodes: List[Episode],
    micro_batch_size: int,
    pad_token_id: int,
    max_grad_norm: float,
    device: torch.device,
    vllm_model: LLM,
    apply_loss: bool = True
) -> dict[str, float]:
    episodes = normalize_rewards_per_group(episodes)
    # sort episodes by length, for more efficient batching
    episodes.sort(
        key=lambda x: len(x.prefix_token_ids) + len(x.generated_token_ids), reverse=True
    )
    n_micro_batches = len(episodes) // micro_batch_size
    n_target_tokens = sum(len(episode.generated_token_ids) for episode in episodes)
    entropy = torch.tensor(0.0, device=device)

    logger.info(
        f"Updating policy with {len(episodes)} episodes, {n_target_tokens} target tokens, {n_micro_batches} micro-batches"
    )

    loss, grad_norm, entropy = 0, 0, 0

    # Iterate over micro-batches
    for i in range(0, len(episodes), micro_batch_size):
        logger.info(f"* Computing policy gradient: {i:>2d}/{len(episodes):>2d}")

        # get a micro-batch of episodes
        j = min(i + micro_batch_size, len(episodes))
        batch_episodes = episodes[i:j]
        batch_lengths = [
            len(episode.prefix_token_ids) + len(episode.generated_token_ids)
            for episode in batch_episodes
        ]
        batch_max_length = max(batch_lengths)
        # get the token ids for the micro-batch
        batch_token_ids = [
            episode.prefix_token_ids
            + episode.generated_token_ids
            + [pad_token_id] * (batch_max_length - batch_lengths[i])
            for i, episode in enumerate(batch_episodes)
        ]
        batch_masks = [
            [0] * len(episode.prefix_token_ids)
            + [1] * len(episode.generated_token_ids)
            + [0] * (batch_max_length - batch_lengths[i])
            for i, episode in enumerate(batch_episodes)
        ]

        # advantage is just reward, once normalized
        batch_advantages = [episode.reward for episode in batch_episodes]
        batch_token_ids_tensor = torch.tensor(
            batch_token_ids, device=device, dtype=torch.long
        )
        batch_masks_tensor = torch.tensor(batch_masks, device=device, dtype=torch.bool)
        batch_advantages_tensor = torch.tensor(
            batch_advantages, device=device, dtype=torch.float32
        )

        input_token_ids = batch_token_ids_tensor[:, :-1]
        target_token_ids = batch_token_ids_tensor[:, 1:]
        target_masks = batch_masks_tensor[:, 1:]
        logger.info("Computing logits...")
        # TODO only compute logits for the target tokens, not the input tokens
        logits: torch.Tensor = model(input_token_ids).logits.float()
        logger.info("Logits computed")

        log_probs = -torch.nn.functional.cross_entropy(
            logits.reshape(-1, logits.size(-1)),
            target_token_ids.reshape(-1),
            ignore_index=pad_token_id,
            reduction="none",
        ).reshape(input_token_ids.shape[0], -1)

        with torch.no_grad():
            logits = logits.reshape(-1, logits.size(-1))
            token_entropy = compute_entropy(logits)
            # single entropy value for the sequence
            entropy = (
                entropy
                + (token_entropy * target_masks.reshape(-1)).sum() / n_target_tokens
            )

        # multiply the log probs by the advantages
        obj = log_probs * batch_advantages_tensor[:, None]
        # scale by the mask, and normalize by token count
        obj: torch.Tensor = (obj * target_masks).sum() / n_target_tokens
        if apply_loss:
            with torch.autograd.detect_anomaly():
                loss = -obj
                loss.backward()

    if apply_loss:
        # update the policy
        grad_norm = torch.nn.utils.clip_grad_norm_(
            model.parameters(), max_norm=max_grad_norm
        )
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)

    logger.info("Syncing params to vLLM...")

    state_dict = model.state_dict()
    state_dict = {k: v.to(dtype=torch.float16) for k, v in state_dict.items()}
    state_dict = {
        k.removeprefix("base_model.model.").replace(".base_layer", ""): v
        for k, v in state_dict.items()
    }

    model_runner: ModelRunnerBase = (
        vllm_model.llm_engine.model_executor.driver_worker.model_runner  # type: ignore
    )
    model_runner.model.load_weights(model.state_dict().items())  # type: ignore
    logger.info("Param update done")

    return {
        "loss": float(loss),
        "grad_norm": float(grad_norm),
        "entropy": float(entropy),
    }


def compute_metrics(
    episodes: List[Episode],
    results: Dict[str, float],
    tb_writer: SummaryWriter,
    step: int,
    optimizer: torch.optim.Optimizer,
) -> Dict[str, float]:
    reward = [episode.reward for episode in episodes]
    # formatted_reward = [episode.reward_info["format_reward"] for episode in episodes]
    # answer_reward = [episode.reward_info["answer_reward"] for episode in episodes]
    num_finished_episodes = sum(episode.is_finished for episode in episodes)
    mean_reward = float(np.mean(reward))
    std_reward = float(np.std(reward))
    # success_rate = float(np.mean(answer_reward))
    # format_reward = float(np.mean(formatted_reward))
    grad_norm = results["grad_norm"]
    entropy = results["entropy"]
    lr = optimizer.param_groups[0]["lr"]
    loss = results["loss"]
    mean_response_len = float(
        np.mean([len(episode.generated_token_ids) for episode in episodes])
    )

    tb_writer.add_scalar("loss", loss, step)
    tb_writer.add_scalar("mean_reward", mean_reward, step)
    tb_writer.add_scalar("std_reward", std_reward, step)
    tb_writer.add_scalar("grad_norm", grad_norm, step)
    tb_writer.add_scalar("num_finished_episodes", num_finished_episodes, step)
    tb_writer.add_scalar("learning_rate", lr, step)
    tb_writer.add_scalar("mean_response_len", mean_response_len, step)
    tb_writer.add_scalar("entropy", entropy, step)
    for i, episode in enumerate(episodes):
        # TensorBoard treats text as markdown.
        text = html.escape(episode.text)
        tb_writer.add_text(f"text_{i}", f"<pre>{text}</pre>", step)

    log_dict = {
        "mean_reward": mean_reward,
        "std_reward": std_reward,
        "grad_norm": grad_norm,
        "entropy": entropy,
        "learning_rate": lr,
        "loss": loss,
        "mean_response_len": mean_response_len,
        "num_finished_episodes": float(num_finished_episodes),
    }
    logger.info(log_dict)
    return log_dict

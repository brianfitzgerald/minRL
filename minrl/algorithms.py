import dataclasses
import gc
from collections import defaultdict
from pydoc import html
from typing import Dict, List

import numpy as np
import torch
import torch.nn as nn
from loguru import logger
from transformers.models.auto.tokenization_auto import AutoTokenizer
from transformers.tokenization_utils_base import PreTrainedTokenizerBase
from vllm import LLM
from vllm.sampling_params import (
    SamplingParams,
)
from vllm.worker.model_runner_base import ModelRunnerBase

from minrl.constants import AlgorithmChoice, TrainerConfig
from minrl.constants import RewardFunction
from minrl.tasks.dataset import Episode, MiniBatch
from minrl.metrics import MetricsWrapper

debug_tokenizer = AutoTokenizer.from_pretrained(
    TrainerConfig().model_id, use_fast=False
)


@torch.no_grad()
def rollout(
    config: TrainerConfig,
    tokenizer: PreTrainedTokenizerBase,
    batch: MiniBatch,
    max_new_tokens: int,
    num_answers_per_question: int,
    reward_function: RewardFunction,
    vllm_model: LLM,
) -> List[Episode]:
    """Generate multiple responses for each prompt in the batch."""
    end_token_id: int = tokenizer.eos_token_id  # type: ignore
    pad_token_id: int = tokenizer.pad_token_id  # type: ignore

    # Convert to tensor and move to device

    logger.info(
        f"Generating responses for {len(batch.prefixes)} prompts, max_tokens={max_new_tokens}, n={num_answers_per_question}"
    )
    prefixes_batch: list[str] = [batch.prefixes[i] for i in range(len(batch.prefixes))]

    outputs = vllm_model.generate(
        prefixes_batch,
        sampling_params=SamplingParams(
            max_tokens=max_new_tokens,
            temperature=config.temperature,
            n=num_answers_per_question,
        ),
    )
    # Clear CUDA cache
    gc.collect()
    torch.cuda.empty_cache()

    # Process outputs and create episodes
    episodes: List[Episode] = []
    for i in range(len(outputs)):
        for j in range(num_answers_per_question):
            # idx of the j-th answer for the i-th prompt

            generated_token_ids: list[int] = list(outputs[i].outputs[j].token_ids)

            # Remove padding tokens
            if pad_token_id in generated_token_ids:
                pad_token_idx = generated_token_ids.index(pad_token_id)
                generated_token_ids = generated_token_ids[:pad_token_idx]

            generated_text = outputs[i].outputs[j].text

            logger.info(f"\nText for response {i}.{j}: {generated_text}")

            # Calculate rewards
            # TODO store and send the whole conversation
            reward = reward_function(
                [{"role": "assistant", "content": generated_text}], batch.samples[i]
            )

            # Create episode
            episode = Episode(
                batch_index=i,
                answer_index=j,
                prefix=batch.prefixes[i],
                text=batch.prefixes[i] + generated_text,
                prefix_token_ids=batch.prefix_token_ids[i],
                generated_token_ids=generated_token_ids,
                is_finished=end_token_id in generated_token_ids,
                reward=reward,
                sample=batch.samples[i],
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
    algorithm: AlgorithmChoice,
    apply_loss: bool = True,
) -> dict[str, float]:
    if algorithm == "grpo":
        episodes = normalize_rewards_per_group(episodes)
    # sort episodes by length, for more efficient batching
    episodes.sort(
        key=lambda x: len(x.prefix_token_ids) + len(x.generated_token_ids), reverse=True
    )
    n_target_tokens = sum(len(episode.generated_token_ids) for episode in episodes)
    entropy = torch.tensor(0.0, device=device)

    logger.info(
        f"Updating policy with {len(episodes)} episodes, {n_target_tokens} target tokens"
    )

    loss, grad_norm, entropy = 0, 0, 0

    # Iterate over micro-batches
    for i in range(0, len(episodes), micro_batch_size):
        # get a micro-batch of episodes
        j = min(i + micro_batch_size, len(episodes))

        batch_episodes = episodes[i:j]
        batch_lengths = [
            len(episode.prefix_token_ids) + len(episode.generated_token_ids)
            for episode in batch_episodes
        ]
        batch_max_length = max(batch_lengths)

        # pad all token ids (prefix + generated) to the same length
        batch_token_ids = [
            episode.prefix_token_ids
            + episode.generated_token_ids
            + [pad_token_id] * (batch_max_length - batch_lengths[i])
            for i, episode in enumerate(batch_episodes)
        ]
        # Mask out the input tokens, and the padding tokens
        batch_masks = [
            [0] * len(episode.prefix_token_ids)
            + [1] * len(episode.generated_token_ids)
            + [0] * (batch_max_length - batch_lengths[i])
            for i, episode in enumerate(batch_episodes)
        ]

        # advantage is just normalized reward
        batch_rewards = [episode.reward for episode in batch_episodes]
        batch_token_ids_tensor = torch.tensor(
            batch_token_ids, device=device, dtype=torch.long
        )
        batch_masks_t = torch.tensor(batch_masks, device=device, dtype=torch.bool)
        batch_rewards_t = torch.tensor(
            batch_rewards, device=device, dtype=torch.float32
        )

        input_token_ids = batch_token_ids_tensor[:, :-1]
        target_token_ids = batch_token_ids_tensor[:, 1:]
        target_masks = batch_masks_t[:, 1:]
        # Often OOMs here, so clear cache
        gc.collect()
        torch.cuda.empty_cache()
        # TODO only compute logits for the target tokens, not the input tokens
        logits: torch.Tensor = model(input_token_ids).logits.float()

        logprobs = -torch.nn.functional.cross_entropy(
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
        if algorithm == "grpo":
            objective = logprobs * batch_rewards_t[:, None]
        elif algorithm == "gpg":
            # subtract baseline, which is the mean of the rewards
            advantages = batch_rewards_t - batch_rewards_t.mean()
            objective = logprobs * advantages[:, None]
        elif algorithm == "reinforce":
            objective = logprobs * batch_rewards_t[:, None]

        # scale by the mask, and normalize by token count
        objective = (objective * target_masks).sum() / n_target_tokens
        if apply_loss:
            loss = -objective
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
    metrics_wrapper: MetricsWrapper,
    step: int,
    optimizer: torch.optim.Optimizer,
) -> Dict[str, float]:
    reward = [episode.reward for episode in episodes]
    num_finished_episodes = sum(episode.is_finished for episode in episodes)
    mean_reward = float(np.mean(reward))
    std_reward = float(np.std(reward))
    grad_norm = results["grad_norm"]
    entropy = results["entropy"]
    lr = optimizer.param_groups[0]["lr"]
    loss = results["loss"]
    mean_response_len = float(
        np.mean([len(episode.generated_token_ids) for episode in episodes])
    )

    metrics_wrapper.add_scalar("train/loss", loss, step)
    metrics_wrapper.add_scalar("train/mean_reward", mean_reward, step)
    metrics_wrapper.add_scalar("train/std_reward", std_reward, step)
    metrics_wrapper.add_scalar("train/grad_norm", grad_norm, step)
    metrics_wrapper.add_scalar(
        "train/num_finished_episodes", num_finished_episodes, step
    )
    metrics_wrapper.add_scalar("train/learning_rate", lr, step)
    metrics_wrapper.add_scalar("train/mean_response_len", mean_response_len, step)
    metrics_wrapper.add_scalar("train/entropy", entropy, step)
    for i, episode in enumerate(episodes):
        text = html.escape(episode.text)
        metrics_wrapper.add_text(f"sample_{i}", text, step)

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
    logger.info(f"Metrics: {log_dict}")
    return log_dict

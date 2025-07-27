import dataclasses
import gc
from collections import defaultdict
from pydoc import html
from typing import Dict, List

import numpy as np
import torch
import torch.nn as nn
from loguru import logger
from tqdm import tqdm
from transformers.models.auto.tokenization_auto import AutoTokenizer
from transformers.tokenization_utils_base import PreTrainedTokenizerBase
from vllm import LLM, TokensPrompt
from vllm.sampling_params import (
    SamplingParams,
)
from vllm.worker.model_runner_base import ModelRunnerBase
import torch.nn.functional as F

from minrl.constants import AlgorithmChoice, Conversation, Sample, TrainerConfig
from minrl.constants import RewardFunction
from minrl.tasks.dataset import Episode, MinRLDataset
from minrl.metrics import MetricsWrapper

debug_tokenizer = AutoTokenizer.from_pretrained(
    TrainerConfig().model_id, use_fast=False
)


def compute_scaled_temperature(
    config: TrainerConfig,
    prev_reward_std: float | None = None,
    reference_std: float = 1.0,
) -> float:
    """
    Compute temperature scaled based on previous batch reward standard deviation.
    As reward std decreases, temperature increases.

    """
    if not config.temperature_scaling or prev_reward_std is None:
        return config.temperature

    if prev_reward_std <= 0:
        return config.temperature_max

    scale_factor = reference_std / prev_reward_std

    scaled_temp = config.temperature * scale_factor

    scaled_temp = max(config.temperature_min, min(config.temperature_max, scaled_temp))

    return scaled_temp


@torch.no_grad()
def rollout(
    config: TrainerConfig,
    tokenizer: PreTrainedTokenizerBase,
    group_size: int,
    dataset: MinRLDataset,
    conversations: list[Conversation],
    samples: list[Sample],
    reward_function: RewardFunction,
    vllm_model: LLM,
    prev_reward_std: float | None = None,
) -> List[Episode]:
    """
    Generate completions for each turn in a batch of conversations.
    Runs for max_turns turns, and generates num_answers_per_question completions for each turn.
    """
    # Compute scaled temperature based on previous reward std
    temperature = compute_scaled_temperature(config, prev_reward_std)

    logger.info(
        f"Generating responses for {len(conversations)} prompts, max_tokens={config.max_new_tokens}, n={config.group_size}, temp={temperature:.3f}"
    )

    # For each turn, generate responses, add to conversation
    for step_idx in tqdm(range(dataset.max_steps), desc="Steps"):
        # Tokenize the conversations
        prefixes_batch = tokenizer.apply_chat_template(
            conversations,  # type: ignore
            tokenize=True,
            enable_thinking=False,  # type: ignore
        )
        prefixes_prompts: list[TokensPrompt] = [
            {"prompt_token_ids": prefix}
            for prefix in prefixes_batch  # type: ignore
        ]
        # Generate list of n_conversations * group_size responses
        outputs = vllm_model.generate(
            prefixes_prompts,
            sampling_params=SamplingParams(
                max_tokens=config.max_new_tokens,
                temperature=temperature,
                # If past the first turn, only generate one response per conversation
                n=group_size if step_idx == 0 else 1,
            ),
        )

        # Parse out the responses, and add to conversation
        for i in range(len(outputs)):
            if step_idx == 0:
                # If first turn, add all responses to conversation
                for j in range(group_size):
                    generated_text = outputs[i].outputs[j].text
                    logger.info(f"\nText for response {i}.{j}: {generated_text}")
                    conversations[i].append(
                        {"role": "assistant", "content": generated_text}
                    )
            else:
                # Otherwise, add the first response to the conversation
                generated_text = outputs[i].outputs[0].text
                conversations[i].append(
                    {"role": "assistant", "content": generated_text}
                )

        # Clear CUDA cache
        gc.collect()
        torch.cuda.empty_cache()

    episodes: List[Episode] = []
    for i, (conversation, sample) in enumerate(zip(conversations, samples)):
        # Calculate rewards
        reward = reward_function(conversation, sample)
        # Create episode
        episode = Episode(
            group_index=i,
            answer_index=0,
            sample=sample,
            reward=reward,
            conversation=conversation,
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


def get_token_ids_and_role_mask(
    conversation: Conversation,
    tokenizer: PreTrainedTokenizerBase,
) -> tuple[list[int], list[str]]:
    if len(conversation) < 2:
        raise ValueError("Conversation must have at least 2 messages")

    # Split conversation into prefix (everything except last assistant message) and generated part
    prefix_conversation = conversation[:-1]
    last_message = conversation[-1]

    if last_message.get("role") != "assistant":
        raise ValueError("Last message in conversation must be from assistant")

    # Build the full conversation (prefix + last assistant message)
    full_conversation = prefix_conversation + [last_message]

    # Tokenize each message individually to build the role mask
    all_token_ids: list[int] = []
    role_mask: list[str] = []
    for msg in full_conversation:
        role = msg["role"]
        # Use the chat template for user/assistant, or fallback to encode
        msg_token_ids = tokenizer.apply_chat_template(
            [msg],  # type: ignore
            tokenize=True,
            enable_thinking=False,
        )
        all_token_ids.extend(msg_token_ids)  # type: ignore
        role_mask.extend([role] * len(msg_token_ids))

    return all_token_ids, role_mask


def normalize_rewards_per_group(episodes: List[Episode]) -> List[Episode]:
    """
    Normalize rewards per group.
    This is done by subtracting the mean and dividing by the standard deviation of the rewards in each group.
    If std is 0, returns the original rewards.
    """
    # Group episodes by prefix and calculate stats
    groups: Dict[int, List[Episode]] = defaultdict(list)
    for episode in episodes:
        groups[episode.group_index].append(episode)

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
    tokenizer: PreTrainedTokenizerBase,
    apply_loss: bool = True,
) -> dict[str, float]:
    """
    Once episodes are generated, use them to update the policy
    by computing the loss from the reward and generated logits.
    This implements a number of different algorithms.
    """
    if algorithm == "grpo":
        episodes = normalize_rewards_per_group(episodes)

    # Extract token IDs from conversations and sort episodes by length for more efficient batching
    episode_tokens = []
    for episode in episodes:
        token_ids, role_mask = get_token_ids_and_role_mask(
            episode.conversation, tokenizer
        )
        episode_tokens.append((episode, token_ids, role_mask))

    episodes = [x[0] for x in episode_tokens]
    n_target_tokens = sum(len(x[1]) for x in episode_tokens)
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
        batch_episode_tokens = episode_tokens[i:j]

        batch_lengths = [len(token_ids) for _, token_ids, _ in batch_episode_tokens]
        batch_max_length = max(batch_lengths)

        # pad all token ids (prefix + generated) to the same length
        batch_token_ids = [
            prefix_tokens
            + generated_tokens
            + [pad_token_id] * (batch_max_length - batch_lengths[idx])
            for idx, (_, prefix_tokens, generated_tokens) in enumerate(
                batch_episode_tokens
            )
        ]
        batch_token_ids_t = torch.tensor(
            batch_token_ids, device=device, dtype=torch.long
        )
        target_token_ids = batch_token_ids_t[:, 1:]
        target_masks = target_token_ids != pad_token_id

        # advantage is just normalized reward
        batch_rewards = [episode.reward for episode in batch_episodes]
        batch_rewards_t = torch.tensor(
            batch_rewards, device=device, dtype=torch.float32
        )
        # Often OOMs here, so clear cache
        gc.collect()
        torch.cuda.empty_cache()
        # TODO only compute logits for the target tokens, not the input tokens
        logits: torch.Tensor = model(batch_token_ids_t).logits.float()

        # Get the cross entropy loss of the label and generated tokens
        logprobs = -torch.nn.functional.cross_entropy(
            logits.reshape(-1, logits.size(-1)),
            target_token_ids.reshape(-1),
            ignore_index=pad_token_id,
            reduction="none",
        ).reshape(batch_token_ids_t.shape[0], -1)

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
    temperature: float | None = None,
) -> Dict[str, float]:
    reward = [episode.reward for episode in episodes]
    # Assume all episodes are finished since rollout completes them
    num_finished_episodes = len(episodes)
    mean_reward = float(np.mean(reward))
    std_reward = float(np.std(reward))
    grad_norm = results["grad_norm"]
    entropy = results["entropy"]
    lr = optimizer.param_groups[0]["lr"]
    loss = results["loss"]
    mean_response_len = float(
        np.mean(
            [
                len(
                    get_token_ids_and_role_mask(episode.conversation, debug_tokenizer)[
                        1
                    ]
                )
                for episode in episodes
            ]
        )
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
    if temperature is not None:
        metrics_wrapper.add_scalar("train/temperature", temperature, step)
    for i, episode in enumerate(episodes):
        # Convert conversation to text format for logging
        conversation_text = "\n".join(
            [f"{msg['role']}: {msg['content']}" for msg in episode.conversation]
        )
        text = html.escape(conversation_text)
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
    if temperature is not None:
        log_dict["temperature"] = temperature
    logger.info(f"Metrics: {log_dict}")
    return log_dict

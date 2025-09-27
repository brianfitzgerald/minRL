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

from minrl.constants import (
    AlgorithmChoice,
    Conversation,
    RewardFunction,
    Sample,
    TrainerConfig,
)
from minrl.metrics import MetricsWrapper
from minrl.constants import Episode
from minrl.utils import NEWLINE_TOKEN_ID, find_assistant_sections

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
    max_steps: int,
    conversations: list[Conversation],
    samples: list[Sample],
    reward_function: RewardFunction,
    vllm_model: LLM,
    prev_reward_std: float | None = None,
) -> List[Episode]:
    """
    Generate completions for each turn in a batch of conversations.
    Runs for max_turns turns, and generates num_answers_per_question completions
    for each turn.
    """
    # Compute scaled temperature based on previous reward std
    temperature = compute_scaled_temperature(config, prev_reward_std)

    logger.info(
        f"Generating responses for {len(conversations)} prompts, "
        f"max_tokens={config.max_new_tokens}, n={config.group_size}, "
        f"temp={temperature:.3f}"
    )

    # For each turn, generate responses, add to conversation
    for step_idx in tqdm(range(max_steps), desc="Steps"):
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


def get_token_ids_and_assistant_mask(
    conversation: Conversation,
    tokenizer: PreTrainedTokenizerBase,
) -> tuple[list[int], list[bool]]:
    if len(conversation) < 1:
        raise ValueError("Conversation must have at least 1 message")

    # Apply chat template to get the full formatted conversation
    all_token_ids = tokenizer.apply_chat_template(
        conversation,  # type: ignore
        tokenize=True,
        enable_thinking=False,
    )

    # Create assistant mask - initially all False
    assistant_mask = [False] * len(all_token_ids)

    # Get special tokens for pattern matching
    assistant_role_tokens = tokenizer.encode("assistant", add_special_tokens=False)
    im_start_token = tokenizer.encode("<|im_start|>", add_special_tokens=False)
    im_end_token = tokenizer.encode("<|im_end|>", add_special_tokens=False)

    # Find all assistant sections and mark them
    assistant_sections = find_assistant_sections(
        all_token_ids, im_start_token, assistant_role_tokens, im_end_token
    )

    for start, end in assistant_sections:
        for j in range(start, end):
            assistant_mask[j] = True

    return all_token_ids, assistant_mask


def normalize_rewards_per_group(episodes: List[Episode]) -> List[Episode]:
    """
    Normalize rewards per group.
    This is done by subtracting the mean and dividing by the standard deviation
    of the rewards in each group. If std is 0, returns the original rewards.
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

    # Extract token IDs from conversations and sort episodes by length
    # for more efficient batching
    episode_tokens: list[tuple[Episode, list[int], list[bool]]] = []
    for episode in episodes:
        token_ids, assistant_mask = get_token_ids_and_assistant_mask(
            episode.conversation, tokenizer
        )
        episode_tokens.append((episode, token_ids, assistant_mask))

    episodes = [episode_data[0] for episode_data in episode_tokens]
    n_target_tokens = sum(len(token_data[1]) for token_data in episode_tokens)
    entropy = torch.tensor(0.0, device=device)

    logger.info(
        f"Updating policy with {len(episodes)} episodes, "
        f"{n_target_tokens} target tokens"
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

        # Pad all token ids to the same length
        batch_token_ids = [
            token_ids + [pad_token_id] * (batch_max_length - batch_lengths[idx])
            for idx, (_, token_ids, _) in enumerate(batch_episode_tokens)
        ]
        batch_token_ids_t = torch.tensor(
            batch_token_ids, device=device, dtype=torch.long
        )

        # Create assistant masks for each sequence in the batch
        batch_assistant_masks = [
            assistant_mask + [False] * (batch_max_length - batch_lengths[idx])
            for idx, (_, _, assistant_mask) in enumerate(batch_episode_tokens)
        ]
        batch_assistant_masks_t = torch.tensor(
            batch_assistant_masks, device=device, dtype=torch.bool
        )

        # Shift tokens and masks for next-token prediction
        target_token_ids = batch_token_ids_t[:, 1:]
        target_assistant_masks = batch_assistant_masks_t[:, 1:]
        pad_masks = target_token_ids != pad_token_id

        # Combine assistant mask with padding mask
        target_masks = target_assistant_masks & pad_masks

        # advantage is just normalized reward
        batch_rewards = [episode.reward for episode in batch_episodes]
        batch_rewards_t = torch.tensor(
            batch_rewards, device=device, dtype=torch.float32
        )
        # Often OOMs here, so clear cache
        gc.collect()
        torch.cuda.empty_cache()

        # Use gradient checkpointing to save memory
        logits: torch.Tensor = model(batch_token_ids_t).logits.float()

        # Get the cross entropy loss of the label and generated tokens
        # Slice logits to match target tokens (exclude first position)
        next_token_logits = logits[:, :-1]

        # Clear logits from memory immediately after use
        del logits
        gc.collect()
        torch.cuda.empty_cache()

        logprobs = -torch.nn.functional.cross_entropy(
            next_token_logits.reshape(-1, next_token_logits.size(-1)),
            target_token_ids.reshape(-1),
            ignore_index=pad_token_id,
            reduction="none",
        ).reshape(batch_token_ids_t.shape[0], -1)

        with torch.no_grad():
            # Calculate entropy only for target positions
            next_token_logits_flat = next_token_logits.reshape(
                -1, next_token_logits.size(-1)
            )
            token_entropy = compute_entropy(next_token_logits_flat)
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

        # Clear intermediate tensors to save memory
        del batch_token_ids_t, target_token_ids, target_masks, batch_rewards_t
        if "logprobs" in locals():
            del logprobs
        if "next_token_logits" in locals():
            del next_token_logits
        gc.collect()
        torch.cuda.empty_cache()

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

    try:
        model_runner: ModelRunnerBase = (
            vllm_model.llm_engine.model_executor.driver_worker.model_runner  # type: ignore
        )
        model_runner.model.load_weights(model.state_dict().items())  # type: ignore
    except AttributeError:
        # vLLM API change: model_executor might not be available in newer versions
        logger.warning(
            "Cannot sync params to vLLM - model_executor not found. This is expected in newer vLLM versions."
        )
    logger.info("Param update done")

    return {
        "loss": float(loss),
        "grad_norm": float(grad_norm),
        "entropy": float(entropy),
    }

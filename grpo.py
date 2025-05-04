import dataclasses
import gc
from collections import defaultdict
from typing import Callable

import torch
import torch.nn as nn
from loguru import logger
from transformers import AutoModelForCausalLM, PreTrainedTokenizer

from data_types import Episode, MiniBatch


@torch.no_grad()
def rollout(
    model: AutoModelForCausalLM,
    batch: MiniBatch,
    tokenizer: PreTrainedTokenizer,
    max_gen_len: int,
    num_answer_per_question: int,
    reward_function: Callable,
    device: torch.device,
) -> list[Episode]:
    """Generate multiple responses for each prompt in the batch."""
    end_token = tokenizer.eos_token
    end_token_id = tokenizer.eos_token_id
    pad_token_id = tokenizer.pad_token_id

    # Prepare input_ids for generation
    input_ids = []
    for prefix_ids in batch.prefix_token_ids:
        for _ in range(num_answer_per_question):
            input_ids.append(prefix_ids)

    # Convert to tensor and move to device
    input_ids = torch.tensor(input_ids, dtype=torch.long, device=device)

    logger.info(f"Generating responses for {len(input_ids)} prompts, max_gen_len={max_gen_len}")
    # Generate responses
    outputs = model.generate(
        input_ids=input_ids,
        max_new_tokens=max_gen_len,
        pad_token_id=pad_token_id,
        eos_token_id=end_token_id,
        do_sample=True,
        use_cache=True,
        return_dict_in_generate=True,
        output_scores=True,
    )

    # Clear CUDA cache
    gc.collect()
    torch.cuda.empty_cache()


    # Process outputs and create episodes
    episodes = []
    for i in range(len(batch.prefix)):
        for j in range(num_answer_per_question):
            idx = i * num_answer_per_question + j

            # Get tokens generated
            generated_token_ids = outputs.sequences[idx][
                len(batch.prefix_token_ids[i]) :
            ].tolist()

            logger.info(f"Generated token ids: {len(generated_token_ids)}")

            # Remove padding tokens
            if pad_token_id in generated_token_ids:
                generated_token_ids = generated_token_ids[
                    : generated_token_ids.index(pad_token_id)
                ]
            logger.info(f"Generated token ids after removing padding: {len(generated_token_ids)}")

            generated_text = tokenizer.decode(
                generated_token_ids, skip_special_tokens=True
            )
            logger.info(f"Generated text: {generated_text}")

            # Calculate rewards
            rewards = reward_function(
                response=generated_text,
                answer=batch.answer[i],
                end_token=end_token,
            )

            # Create episode
            episode = Episode(
                prefix=batch.prefix[i],
                text=batch.prefix[i] + generated_text,
                prefix_token_ids=batch.prefix_token_ids[i],
                generated_token_ids=generated_token_ids,
                is_finished=end_token_id in generated_token_ids,
                reward=rewards["reward"],
                reward_info=rewards["reward_info"],
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


def normalize_rewards_per_group(episodes: list[Episode]) -> list[Episode]:
    """
    Normalize rewards per group.
    This is done by subtracting the mean and dividing by the standard deviation of the rewards in each group.
    """
    # Group episodes by prefix and calculate stats
    groups = defaultdict(list)
    for episode in episodes:
        groups[episode.prefix].append(episode)

    # Normalize rewards within each group
    normalized_episodes = []
    for group in groups.values():
        rewards = torch.tensor([e.reward for e in group])
        mean, std = rewards.mean(), rewards.std()
        for episode in group:
            normalized_episodes.append(
                dataclasses.replace(episode, reward=(episode.reward - mean) / std)
            )

    return normalized_episodes


def update_policy(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    episodes: list[Episode],
    micro_batch_size: int,
    pad_token_id: int,
    max_grad_norm: float,
    device: torch.device,
    dtype: torch.dtype,
):
    episodes = normalize_rewards_per_group(episodes)
    # sort episodes by length, for more efficient batching
    episodes.sort(
        key=lambda x: len(x.prefix_token_ids) + len(x.generated_token_ids), reverse=True
    )
    n_micro_batches = len(episodes) // micro_batch_size
    n_target_tokens = sum(len(episode.generated_token_ids) for episode in episodes)
    entropy = 0.0

    logger.info(
        f"Updating policy with {len(episodes)} episodes, {n_target_tokens} target tokens, {n_micro_batches} micro-batches"
    )

    for i in range(0, len(episodes), micro_batch_size):
        logger.info(
            f"\r* Computing policy gradient: {i:>2d}/{len(episodes):>2d}",
            flush=True,
            end="",
        )

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
        batch_masks: list[list[int]] = [
            [0] * len(episode.prefix_token_ids)
            + [1] * len(episode.generated_token_ids)
            + [0] * (batch_max_length - batch_lengths[i])
            for i, episode in enumerate(batch_episodes)
        ]

        # advantage is just reward, once normalized
        batch_advantages = [episode.reward for episode in batch_episodes]
        batch_token_ids = torch.tensor(batch_token_ids, device=device, dtype=torch.long)
        batch_masks = torch.tensor(batch_masks, device=device, dtype=torch.bool)
        batch_advantages = torch.tensor(
            batch_advantages, device=device, dtype=torch.float32
        )

        with torch.autocast(device_type=device.type, dtype=dtype):
            input_token_ids = batch_token_ids[:, :-1]
            target_token_ids = batch_token_ids[:, 1:]
            target_masks = batch_masks[:, 1:]
            # get the logits for the micro-batch
            # TODO replace with vllm
            out = model.forward(input_token_ids)
            logits: torch.Tensor = out.logits.float()

        # cross entropy, ignore padding tokens
        log_probs = -torch.nn.functional.cross_entropy(
            logits.reshape(-1, logits.size(-1)),
            target_token_ids.reshape(-1),
            ignore_index=pad_token_id,
            reduction="none",
        ).reshape(input_token_ids.shape[0], -1)

        with torch.no_grad():
            token_entropy = compute_entropy(logits)
            entropy = entropy + (token_entropy * target_masks).sum() / n_target_tokens

        # multiply the log probs by the advantages
        obj = log_probs * batch_advantages[:, None]
        # scale by the mask, and normalize by token count
        obj: torch.Tensor = (obj * target_masks).sum() / n_target_tokens
        loss = -obj
        loss.backward()

    # update the policy
    grad_norm = torch.nn.utils.clip_grad_norm_(
        model.parameters(), max_norm=max_grad_norm
    )
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    return {
        "loss": loss.item(),
        "grad_norm": grad_norm.item(),
        "entropy": entropy.item(),
    }

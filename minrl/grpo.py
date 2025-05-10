import dataclasses
import gc
from collections import defaultdict
from pydoc import html
from typing import Callable, Dict, List
from tensorboardX import SummaryWriter

import numpy as np
import torch
import torch.nn as nn
from loguru import logger
from transformers.generation.utils import GenerateOutput
from transformers.models.auto.modeling_auto import AutoModelForCausalLM
from transformers.tokenization_utils import PreTrainedTokenizer

from minrl.data_types import Episode, MiniBatch
from vllm_inference.client import GenerateResponse, VLLMClient


@torch.no_grad()
def rollout(
    model: AutoModelForCausalLM,
    tokenizer: PreTrainedTokenizer,
    batch: MiniBatch,
    max_new_tokens: int,
    num_answer_per_question: int,
    reward_function: Callable,
    device: torch.device,
    client: VLLMClient | None = None,
) -> List[Episode]:
    """Generate multiple responses for each prompt in the batch."""
    end_token = tokenizer.eos_token
    end_token_id = tokenizer.eos_token_id
    pad_token_id = tokenizer.pad_token_id
    using_vllm = client is not None

    # Convert to tensor and move to device

    logger.info(
        f"Generating responses for {len(batch.prefixes)} prompts, max_tokens={max_new_tokens}"
    )
    if using_vllm:
        # repeat each prompt num_answer_per_question times
        prefixes_batch = [
            batch.prefixes[i]
            for i in range(len(batch.prefixes))
            for _ in range(num_answer_per_question)
        ]
        print(len(prefixes_batch))
        outputs = client.generate(prompts=prefixes_batch)
    else:

        # Prepare input_ids for generation
        input_ids: list[list[int]] = []
        for prefix_ids in batch.prefix_token_ids:
            for _ in range(num_answer_per_question):
                input_ids.append(prefix_ids)
        input_ids_tensor = torch.tensor(input_ids, dtype=torch.long, device=device)
        # Generate responses
        outputs_transformers = model.generate(  # type: ignore
            input_ids=input_ids_tensor,
            max_new_tokens=max_new_tokens,
            pad_token_id=pad_token_id,
            eos_token_id=end_token_id,
            do_sample=True,
            use_cache=True,
            top_k=20,
            return_dict_in_generate=True,
            output_scores=True,
        )
        outputs = GenerateResponse(
            completion_ids=outputs_transformers.sequences,
            generated_logprobs=outputs_transformers.logits,
        )
    # Clear CUDA cache
    gc.collect()
    torch.cuda.empty_cache()

    # Process outputs and create episodes
    episodes: List[Episode] = []
    for i in range(len(batch.prefixes)):
        for j in range(num_answer_per_question):
            idx = i * num_answer_per_question + j

            if using_vllm:
                print(outputs)
                generated_token_ids = outputs.completion_ids[idx]
                assert outputs.generated_logprobs is not None
                generated_logprobs = outputs.generated_logprobs[idx]
            else:
                # Get tokens generated
                generated_token_ids = outputs.sequences[idx][  # type: ignore
                    len(batch.prefix_token_ids[i]) :
                ].tolist()

            logger.info(f"Generated token ids: {len(generated_token_ids)}")

            # Remove padding tokens
            if pad_token_id in generated_token_ids:
                generated_token_ids = generated_token_ids[
                    : generated_token_ids.index(pad_token_id)
                ]
            logger.info(
                f"Generated token ids after removing padding: {len(generated_token_ids)}"
            )

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
            print("rewards", rewards)

            # Create episode
            episode = Episode(
                prefix=batch.prefixes[i],
                text=batch.prefixes[i] + generated_text,
                prefix_token_ids=batch.prefix_token_ids[i],
                generated_token_ids=generated_token_ids,
                is_finished=end_token_id in generated_token_ids,
                reward=rewards["reward"],
                reward_info=rewards["reward_info"],
                generated_logprobs=generated_logprobs,
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
        print("rewards", rewards)
        print("mean", mean)
        print("std", std)

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
    dtype: torch.dtype,
    apply_loss: bool = True,
) -> dict[str, float]:
    print("episodes", [episode.reward for episode in episodes])
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

        with torch.autocast(device_type=device.type, dtype=dtype):
            input_token_ids = batch_token_ids_tensor[:, :-1]
            target_token_ids = batch_token_ids_tensor[:, 1:]
            target_masks = batch_masks_tensor[:, 1:]
            # get the logits for the micro-batch
            if episodes[0].generated_logprobs is not None:
                all_logprobs = []
                for episode in batch_episodes:
                    assert episode.generated_logprobs is not None
                    for logprobs in episode.generated_logprobs:
                        all_logprobs.append(logprobs)
                # HACK: vllm returns logprobs, not logits - this is likely unstable and needs to be fixed.
                # Need to at least prune to top N logprobs and remove low probability tokens.
                logits = torch.tensor(all_logprobs, device=device, dtype=dtype)
            else:
                out = model(input_ids=input_token_ids, attention_mask=batch_masks_tensor)
                logits: torch.Tensor = out.logits

        # cross entropy, ignore padding tokens
        log_probs = -torch.nn.functional.cross_entropy(
            logits.reshape(-1, logits.size(-1)),
            target_token_ids.reshape(-1),
            ignore_index=pad_token_id,
            reduction="none",
        ).reshape(input_token_ids.shape[0], -1)
        with torch.no_grad():
            token_entropy = compute_entropy(logits)
            # single entropy value for the sequence
            entropy = entropy + (token_entropy * target_masks).sum() / n_target_tokens

        # multiply the log probs by the advantages
        obj = log_probs * batch_advantages_tensor[:, None]
        # scale by the mask, and normalize by token count
        obj: torch.Tensor = (obj * target_masks).sum() / n_target_tokens
        if apply_loss:
            loss = -obj
            loss.backward()

    if apply_loss:
        # update the policy
        grad_norm = torch.nn.utils.clip_grad_norm_(
            model.parameters(), max_norm=max_grad_norm
        )
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)

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
    """
    Compute and return important metrics from episodes and training results.

    Args:
        episodes: List of episodes from the current batch
        results: Dictionary containing training results (loss, grad_norm, entropy)
        optimizer: The optimizer used for training

    Returns:
        Dictionary containing all computed metrics
    """
    reward = [episode.reward for episode in episodes]
    formatted_reward = [episode.reward_info["format_reward"] for episode in episodes]
    answer_reward = [episode.reward_info["answer_reward"] for episode in episodes]
    num_finished_episodes = sum(episode.is_finished for episode in episodes)
    mean_reward = float(np.mean(reward))
    std_reward = float(np.std(reward))
    success_rate = float(np.mean(answer_reward))
    format_reward = float(np.mean(formatted_reward))
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
    tb_writer.add_scalar("success_rate/train", success_rate, step)
    tb_writer.add_scalar("format_reward", format_reward, step)
    tb_writer.add_scalar("grad_norm", grad_norm, step)
    tb_writer.add_scalar("num_finished_episodes", num_finished_episodes, step)
    tb_writer.add_scalar("learning_rate", lr, step)
    tb_writer.add_scalar("mean_response_len", mean_response_len, step)
    tb_writer.add_scalar("entropy", entropy, step)
    for i, episode in enumerate(episodes):
        # TensorBoard treats text as markdown.
        text = html.escape(episode.text)
        tb_writer.add_text(f"text_{i}", f"<pre>{text}</pre>", step)
    print(
        f"\rStep {step}, mean_reward: {mean_reward:.2f}, "
        f"train success_rate: {success_rate:.2f}, "
        f"grad_norm: {grad_norm:.2f}, "
        f"num_finished_episodes: {num_finished_episodes}, "
        f"mean_response_len: {mean_response_len:.2f}, "
        f"entropy: {entropy:.2f}"
    )

    return {
        "mean_reward": mean_reward,
        "std_reward": std_reward,
        "success_rate": success_rate,
        "format_reward": format_reward,
        "grad_norm": grad_norm,
        "entropy": entropy,
        "learning_rate": lr,
        "loss": loss,
        "mean_response_len": mean_response_len,
        "num_finished_episodes": float(num_finished_episodes),
    }

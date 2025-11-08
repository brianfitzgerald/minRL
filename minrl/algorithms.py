import html
import time
from collections import defaultdict
from typing import List, TypedDict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from loguru import logger
from torch import Tensor as T
from tqdm import tqdm
from transformers.tokenization_utils_base import PreTrainedTokenizerBase
from vllm import LLM, TokensPrompt
from vllm.sampling_params import (
    SamplingParams,
)
from vllm.worker.model_runner_base import ModelRunnerBase

from minrl.constants import (
    AlgorithmChoice,
    Conversation,
    Episode,
    RewardFunction,
    Sample,
    TrainerConfig,
)
from minrl.metrics import MetricsWrapper
from minrl.utils import (
    clear_memory,
    find_assistant_sections,
    log_conversation,
    log_memory_usage,
)


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
) -> tuple[List[Episode], float]:
    """
    Generate completions for each turn in a batch of conversations.
    Runs for max_steps turns, and generates group_size completions
    for the first turn, then 1 completion per turn for subsequent turns.
    """
    rollout_start_time = time.perf_counter()

    # Get stop token IDs to stop generation on
    stop_token_ids: list[int] | None = None
    eos_token_id = tokenizer.eos_token_id
    if isinstance(eos_token_id, int):
        stop_token_ids = [eos_token_id]

    num_prompts = len(conversations)
    total_conversations = num_prompts * group_size

    logger.info(
        f"Generating responses for {num_prompts} prompts x {group_size} group_size = {total_conversations} total conversations, "  # noqa: E501
        f"max_tokens={config.max_new_tokens}, temp={config.temperature:.3f}"
    )

    # Flatten structure: instead of nested lists, maintain flat list of all conversations
    # Index: [group0_resp0, group0_resp1, ..., group0_respN, group1_resp0, ...]
    # This makes batching simpler and more efficient
    flat_conversations: list[Conversation] = []
    group_indices: list[
        int
    ] = []  # Track which original prompt each conversation belongs to

    # Initialize: create group_size copies of each initial conversation
    for i, conv in enumerate(conversations):
        for j in range(group_size):
            flat_conversations.append(list(conv))
            group_indices.append(i)

    # Generate responses for all conversations across all steps
    for step_idx in tqdm(range(max_steps), desc="Rollout Step", disable=max_steps == 1):
        # Prepare batch: use all flat_conversations for step 0, or vLLM's n parameter
        if step_idx == 0:
            # First step optimization: batch initial prompts and use vLLM's n parameter
            batch_conversations = [conversations[i] for i in range(num_prompts)]
            n_responses = group_size
        else:
            # Subsequent steps: batch all existing conversations
            batch_conversations = flat_conversations
            n_responses = 1

        # Tokenize all conversations in one batch
        templated_conversations = tokenizer.apply_chat_template(
            batch_conversations,  # pyright: ignore[reportArgumentType]
            tokenize=False,
            add_generation_prompt=True,
        )
        tokenized_conversations = tokenizer(templated_conversations)["input_ids"]  # pyright: ignore[reportArgumentType]

        # Prepare vLLM input as a single batch
        vllm_input: list[TokensPrompt] = [
            {"prompt_token_ids": token_ids}
            for token_ids in tokenized_conversations  # type: ignore
        ]

        logger.info(
            f"N responses: {n_responses} inference batch size: {len(vllm_input)}"
        )

        # Single batched generation call
        outputs = vllm_model.generate(
            vllm_input,
            sampling_params=SamplingParams(
                max_tokens=config.max_new_tokens,
                temperature=config.temperature,
                n=n_responses,
                stop_token_ids=stop_token_ids,
            ),
        )

        # Parse and append responses
        if step_idx == 0:
            # First step: vLLM returned group_size responses per prompt
            # Update our flat_conversations structure
            for prompt_idx, output in enumerate(outputs):
                for resp_idx, completion in enumerate(output.outputs):
                    flat_idx = prompt_idx * group_size + resp_idx
                    flat_conversations[flat_idx].append(
                        {"role": "assistant", "content": completion.text}
                    )
        else:
            # Subsequent steps: vLLM returned 1 response per conversation
            for conv_idx, output in enumerate(outputs):
                generated_text = output.outputs[0].text
                flat_conversations[conv_idx].append(
                    {"role": "assistant", "content": generated_text}
                )

        # Free memory
        del outputs, vllm_input, tokenized_conversations, templated_conversations
        clear_memory()

    # Create episodes from all generated conversations
    episodes: List[Episode] = []
    for flat_idx, (conversation, group_idx) in enumerate(
        zip(flat_conversations, group_indices)
    ):
        answer_idx = flat_idx % group_size
        sample = samples[group_idx]
        reward = reward_function(conversation, sample)

        episode = Episode(
            group_index=group_idx,
            answer_index=answer_idx,
            sample=sample,
            reward=reward,
            conversation=conversation,
        )
        episodes.append(episode)
        logger.info(f"Episode {group_idx}:{answer_idx}: reward: {reward:.4f}")
        log_conversation(conversation, only_roles=["assistant"])

    rollout_duration = time.perf_counter() - rollout_start_time

    return episodes, rollout_duration


def compute_entropy(logits: T) -> T:
    """Compute entropy efficiently in-place to minimize memory allocation.

    Using the identity: entropy = -sum(p * log(p))
    Where p = softmax(logits) and log(p) = log_softmax(logits)
    """
    # Use log_softmax and softmax - PyTorch optimizes these internally
    logp = F.log_softmax(logits, dim=-1)
    p = F.softmax(logits, dim=-1)
    # Compute entropy: -sum(p * log(p))
    # Use in-place operations where possible to reduce memory
    entropy = -(p * logp).sum(dim=-1)
    return entropy


def get_token_ids_and_assistant_mask(
    conversation: Conversation,
    tokenizer: PreTrainedTokenizerBase,
) -> tuple[list[int], list[bool]]:
    """Get token IDs and assistant mask, using cache if available."""
    # Check cache first

    if len(conversation) < 1:
        raise ValueError("Conversation must have at least 1 message")

    # Apply chat template to get the full formatted conversation
    all_token_ids: list[int] = tokenizer.apply_chat_template(
        conversation,  # type: ignore
        tokenize=True,
        enable_thinking=False,
    )

    # Create assistant mask - initially all False
    assistant_mask = [False] * len(all_token_ids)

    # Get newline token ID for this tokenizer
    newline_token_ids = tokenizer.encode("\n", add_special_tokens=False)
    newline_token_id = newline_token_ids[0] if newline_token_ids else None

    # Try ChatML format first (Qwen, SmolLM, etc.)
    assistant_role_tokens = tokenizer.encode("assistant", add_special_tokens=False)
    im_start_token = tokenizer.encode("<|im_start|>", add_special_tokens=False)
    im_end_token = tokenizer.encode("<|im_end|>", add_special_tokens=False)

    # Find all assistant sections and mark them
    assistant_sections = find_assistant_sections(
        all_token_ids,
        im_start_token,
        assistant_role_tokens,
        im_end_token,
        newline_token_id,
    )

    # If no sections found, try Gemma format
    if not assistant_sections:
        model_role_tokens = tokenizer.encode("model", add_special_tokens=False)
        start_of_turn_token = tokenizer.encode(
            "<start_of_turn>", add_special_tokens=False
        )
        end_of_turn_token = tokenizer.encode("<end_of_turn>", add_special_tokens=False)

        assistant_sections = find_assistant_sections(
            all_token_ids,
            start_of_turn_token,
            model_role_tokens,
            end_of_turn_token,
            newline_token_id,
        )

    # Assert we found assistant spans; if not, this is likely a chat template mismatch
    if not assistant_sections:
        raise ValueError(
            "No assistant spans found in conversation tokens. "
            "Ensure the tokenizer's chat template matches the conversation format."
        )

    for start, end in assistant_sections:
        for j in range(start, end):
            assistant_mask[j] = True

    return all_token_ids, assistant_mask


def compute_algorithm_loss(
    logprobs: T,
    target_masks: T,
    batch_advantages: T,
    algorithm: AlgorithmChoice,
    n_target_tokens: int,
    entropy: T | None = None,
    entropy_coef: float = 0.0,
) -> T:
    """
    Compute the loss for the given algorithm and other required inputs.
    """
    # multiply the log probs by the advantages
    if algorithm == "grpo":
        obj = logprobs * batch_advantages[:, None]
    elif algorithm == "gpg":
        # subtract baseline, which is the mean of the rewards
        advantages = batch_advantages - batch_advantages.mean()
        obj = logprobs * advantages[:, None]
    elif algorithm == "reinforce":
        obj = logprobs * batch_advantages[:, None]

    # scale by the mask, and normalize by token count
    # this sets the advantage to 0 for padding tokens
    # Sum over all tokens and divide by count for mean loss per token
    loss = (obj * target_masks).sum() / n_target_tokens

    # Add entropy regularization (negative because we want to maximize entropy)
    if entropy is not None and entropy_coef > 0:
        entropy_loss = -entropy_coef * entropy
        loss += entropy_loss

    loss = -loss

    return loss


def sync_weights_to_vllm(
    model: nn.Module,
    vllm_model: LLM,
) -> None:
    logger.info("Syncing params to vLLM...")

    state_dict = model.state_dict()

    try:
        model_runner: ModelRunnerBase = (
            vllm_model.llm_engine.model_executor.driver_worker.model_runner  # type: ignore
        )
        model_runner.model.load_weights(state_dict.items())  # type: ignore
        logger.info("Param update succesful.")
    except AttributeError:
        # vLLM API change: model_executor might not be available in newer versions
        logger.warning(
            "Cannot sync params to vLLM - model_executor not found. Hint: disable v1 API."
        )


class UpdatePolicyResults(TypedDict):
    loss: float
    grad_norm: float
    mean_policy_ratio: float
    mean_kl: float
    duration: float


class EpisodeWithTokens(TypedDict):
    episode: Episode
    token_ids: list[int]
    assistant_mask: list[bool]


class PreprocessedBatch(TypedDict):
    # shape: (batch_size, seq_len) - int
    batch_token_ids_t: T
    # shape: (batch_size, seq_len) - binary
    batch_assistant_masks_t: T
    # shape: (batch_size, seq_len) - int
    target_token_ids: T
    # shape: (batch_size, seq_len) - binary
    target_masks: T
    # shape: (batch_size,) - float
    advantages: T


def preprocess_batch(
    episodes: List[Episode],
    tokenizer: PreTrainedTokenizerBase,
    pad_token_id: int,
) -> PreprocessedBatch:
    """
    Preprocess a single batch of episodes to get masks and tokens for
    forward pass and loss computation.
    """
    # Extract token IDs and assistant masks for each episode
    token_ids_list: list[list[int]] = []
    assistant_mask_list: list[list[bool]] = []
    for episode in episodes:
        token_ids, assistant_mask = get_token_ids_and_assistant_mask(
            episode.conversation, tokenizer
        )
        token_ids_list.append(token_ids)
        assistant_mask_list.append(assistant_mask)

    # Pad token IDs on CPU and optionally pin memory for faster H2D copies
    batch_token_ids_t = [
        torch.tensor(token_ids, dtype=torch.long) for token_ids in token_ids_list
    ]
    batch_token_ids_t = torch.nn.utils.rnn.pad_sequence(
        batch_token_ids_t, batch_first=True, padding_value=pad_token_id
    )

    # Pad assistant masks on CPU and optionally pin memory
    batch_assistant_masks_t = [
        torch.tensor(assistant_mask, dtype=torch.bool)
        for assistant_mask in assistant_mask_list
    ]
    batch_assistant_masks_t = torch.nn.utils.rnn.pad_sequence(
        batch_assistant_masks_t, batch_first=True, padding_value=False
    )

    target_token_ids = batch_token_ids_t[:, 1:]
    target_assistant_masks = batch_assistant_masks_t[:, 1:]
    # Create mask with padding tokens
    pad_masks = target_token_ids != pad_token_id

    # Combine assistant mask with padding mask
    target_masks = target_assistant_masks & pad_masks

    # get n unique group indices
    group_indices = list(set([e.group_index for e in episodes]))
    n_prompts_per_step = len(group_indices)
    n_groups = len(episodes) // n_prompts_per_step

    raw_reward_tensor = torch.tensor(
        [e.reward for e in episodes], dtype=torch.float
    ).reshape((n_prompts_per_step, n_groups))
    means = raw_reward_tensor.mean(dim=-1).unsqueeze(1)
    advantages = (raw_reward_tensor - means).reshape((n_prompts_per_step * n_groups,))

    return {
        "batch_token_ids_t": batch_token_ids_t,
        "batch_assistant_masks_t": batch_assistant_masks_t,
        "target_token_ids": target_token_ids,
        "target_masks": target_masks,
        "advantages": advantages,
    }


def get_response_log_probs(
    model: torch.nn.Module,
    batch: PreprocessedBatch,
    device: torch.device,
) -> torch.Tensor:
    """
    Compute log probabilities for given labels efficiently.
    Uses F.log_softmax which is more memory efficient than manual computation.
    """
    input_ids = batch["batch_token_ids_t"].to(device)
    labels = batch["target_token_ids"].to(device)
    logits = model(input_ids).logits
    # Use F.log_softmax which is more memory efficient than manual computation
    log_probs = F.log_softmax(logits, dim=-1)
    # Free logits immediately
    del logits
    # Gather only the logprobs for the labels
    logprobs_for_label = torch.gather(
        log_probs, dim=-1, index=labels.unsqueeze(-1)
    ).squeeze(-1)
    del log_probs
    return logprobs_for_label


def update_policy(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    episodes: List[Episode],
    micro_batch_size: int,
    pad_token_id: int,
    max_grad_norm: float,
    device: torch.device,
    tokenizer: PreTrainedTokenizerBase,
    apply_loss: bool = True,
    metrics_wrapper: MetricsWrapper | None = None,
    step: int | None = None,
    gradient_accumulation_steps: int | None = None,
) -> UpdatePolicyResults:
    """
    Once episodes are generated, use them to update the policy
    by computing the loss from the reward and generated logits.
    This implements a number of different algorithms.
    """
    update_start_time = time.perf_counter()

    # Filter out groups where mean or std == 0
    grouped_eps = defaultdict(list)
    for ep in episodes:
        grouped_eps[ep.group_index].append(ep)

    filtered_episodes = []
    for group_index, group in grouped_eps.items():
        rewards = [ep.reward for ep in group]
        group_mean = np.mean(rewards)
        group_std = np.std(rewards)
        # Filter out groups where mean or std == 0
        if not (group_std == 0 or group_mean == 0):
            logger.info(
                f"Filtering group {group_index} with mean {group_mean} and std {group_std}"
            )
            filtered_episodes.extend(group)

    episodes = filtered_episodes

    # Debug: Log overall episode statistics
    all_rewards = [e.reward for e in episodes]
    logger.debug(
        f"update_policy: Total episodes={len(episodes)}, "
        f"rewards - mean={sum(all_rewards) / len(all_rewards):.4f}, "
        f"min={min(all_rewards):.4f}, max={max(all_rewards):.4f}, "
        f"std={torch.tensor(all_rewards).std().item():.4f}, "
        f"all_zero={all(r == 0 for r in all_rewards)}"
    )

    total_loss = 0.0
    grad_norm = 0.0
    num_micro_batches = 0

    # Compute number of micro-batches up-front for correct gradient accumulation scaling
    # This makes the accumulated gradient equivalent to averaging over the full batch
    total_micro_batches = (len(episodes) + micro_batch_size - 1) // micro_batch_size

    # If gradient_accumulation_steps is not provided, use total_micro_batches
    if gradient_accumulation_steps is None:
        gradient_accumulation_steps = total_micro_batches
        logger.info(
            f"Auto-setting gradient_accumulation_steps to {gradient_accumulation_steps}"
        )

    logger.info(
        f"Processing {total_micro_batches} micro-batches, {gradient_accumulation_steps} total accumulation steps"
    )

    all_ref_logprobs = []

    logger.info("Starting gradient updates with on-the-fly preprocessing...")
    epoch_kl_divs = []
    epoch_policy_ratios = []
    preprocessed_batches = []
    for micro_batch_start in tqdm(
        range(0, len(episodes), micro_batch_size), desc="Ref logprobs"
    ):
        micro_batch_end = min(micro_batch_start + micro_batch_size, len(episodes))
        batch_episodes = episodes[micro_batch_start:micro_batch_end]
        ppb = preprocess_batch(
            batch_episodes,
            tokenizer,
            pad_token_id,
        )
        preprocessed_batches.append(ppb)
        with torch.no_grad():
            ref_logprobs = get_response_log_probs(model, ppb, device).detach()
            all_ref_logprobs.append(ref_logprobs)

    # Iterate over micro-batches, preprocessing on-the-fly to save memory
    micro_batch_idx = 0
    for micro_batch_start in tqdm(
        range(0, len(episodes), micro_batch_size), desc="Policy training"
    ):
        micro_batch_end = min(micro_batch_start + micro_batch_size, len(episodes))
        batch_episodes = episodes[micro_batch_start:micro_batch_end]
        micro_batch_start_time = time.perf_counter()
        ppb = preprocessed_batches[micro_batch_idx]

        mask: T = ppb["target_masks"].to(device, non_blocking=True)
        advantages = ppb["advantages"].to(device, non_blocking=True)
        # Compute reference logprobs for this micro-batch only (on-the-fly)

        # Compute policy logprobs (single forward pass with gradients)
        policy_logprobs = get_response_log_probs(
            model,
            ppb,
            device,
        )

        # get the importance weights
        # use exp here to avoid numerical instability
        ratio = torch.exp(policy_logprobs - all_ref_logprobs[micro_batch_idx])
        # TODO normalize per group
        # advantages already contains only the current micro-batch, no need to slice
        micro_batch_adv = advantages.unsqueeze(-1)

        per_token_loss: T = -ratio * micro_batch_adv

        # Compute KL divergence
        with torch.no_grad():
            approx_kl = ((ratio - 1) - torch.log(ratio)) * mask
            approx_kl_mean = approx_kl.sum() / mask.sum()
            # Log the KL divergence for this micro-batch
            epoch_kl_divs.append(approx_kl_mean.item())

            # Policy ratio statistics for this micro-batch
            policy_ratio_mean = (ratio * mask).sum() / mask.sum()
            epoch_policy_ratios.append(policy_ratio_mean.item())

        # Get the loss per token, by multiplying by the mask
        masked_loss: T = per_token_loss * mask
        # Compute loss as mean over all masked tokens (not per-prompt average)
        total_masked_tokens = mask.sum().clamp_min(1)
        loss = masked_loss.sum() / total_masked_tokens / gradient_accumulation_steps
        if apply_loss:
            loss.backward()
            logger.info(
                f"Applied loss for micro-batch {micro_batch_idx}, loss: {loss:.4f}"
            )

        total_loss += float(loss.detach().cpu())
        num_micro_batches += 1
        micro_batch_idx += 1

        # Update weights after accumulating gradient_accumulation_steps batches
        # or if this is the last batch
        should_update = (micro_batch_idx % gradient_accumulation_steps == 0) or (
            micro_batch_end >= len(episodes)
        )

        if apply_loss and should_update:
            grad_norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(), max_norm=max_grad_norm
            )
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            logger.info(
                f"Updated weights after micro-batch {micro_batch_idx}, grad_norm: {grad_norm:.4f}"  # noqa: E501
            )

        micro_batch_duration = time.perf_counter() - micro_batch_start_time
        logger.info(
            f"Micro-batch {micro_batch_idx} completed in {micro_batch_duration:.2f}s"
        )

        del policy_logprobs
        del ratio
        del mask
        del micro_batch_adv
        del masked_loss
        del loss
        del advantages
        del batch_episodes
        del ppb
        clear_memory()

    # Log GPU utilization after compute_loss
    log_memory_usage("update_policy", metrics_wrapper=metrics_wrapper, step=step)

    # Return average loss and entropy across all micro-batches
    avg_loss = total_loss / num_micro_batches if num_micro_batches > 0 else 0.0

    mean_policy_ratio = (
        sum(epoch_policy_ratios) / len(epoch_policy_ratios)
        if epoch_policy_ratios
        else 0
    )
    mean_kl = sum(epoch_kl_divs) / len(epoch_kl_divs) if epoch_kl_divs else 0

    update_duration = time.perf_counter() - update_start_time
    logger.info(f"Policy update completed in {update_duration:.2f}s")

    clear_memory()

    return {
        "loss": float(avg_loss),
        "grad_norm": float(grad_norm),
        "duration": update_duration,
        "mean_policy_ratio": mean_policy_ratio,
        "mean_kl": mean_kl,
    }


def compute_metrics(
    episodes: List[Episode],
    results: UpdatePolicyResults,
    metrics_wrapper: MetricsWrapper,
    step: int,
    optimizer: torch.optim.Optimizer,
    log_text: bool = False,
) -> dict[str, float]:
    reward = [episode.reward for episode in episodes]
    # Assume all episodes are finished since rollout completes them
    num_finished_episodes = len(episodes)
    mean_reward = float(np.mean(reward))
    std_reward = float(np.std(reward))
    grad_norm = results["grad_norm"]
    mean_policy_ratio = results["mean_policy_ratio"]
    mean_kl = results["mean_kl"]
    lr = optimizer.param_groups[0]["lr"]
    loss = results["loss"]
    metrics_wrapper.add_scalar("train/loss", loss, step)
    metrics_wrapper.add_scalar("train/mean_reward", mean_reward, step)
    metrics_wrapper.add_scalar("train/std_reward", std_reward, step)
    metrics_wrapper.add_scalar("train/grad_norm", grad_norm, step)
    metrics_wrapper.add_scalar(
        "train/num_finished_episodes", num_finished_episodes, step
    )
    metrics_wrapper.add_scalar("train/learning_rate", lr, step)
    metrics_wrapper.add_scalar("train/mean_policy_ratio", mean_policy_ratio, step)
    metrics_wrapper.add_scalar("train/mean_kl", mean_kl, step)
    if log_text:
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
        "mean_policy_ratio": mean_policy_ratio,
        "mean_kl": mean_kl,
        "learning_rate": lr,
        "loss": loss,
        "num_finished_episodes": float(num_finished_episodes),
    }
    logger.info(f"Metrics: {log_dict}")
    return log_dict

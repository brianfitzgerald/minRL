import time
from contextlib import nullcontext
from typing import List, TypedDict

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
) -> tuple[List[Episode], float]:
    """
    Generate completions for each turn in a batch of conversations.
    Runs for max_steps turns, and generates group_size completions
    for the first turn, then 1 completion per turn for subsequent turns.
    """
    rollout_start_time = time.perf_counter()

    # Compute scaled temperature based on previous reward std
    temperature = compute_scaled_temperature(config, prev_reward_std)

    # Get stop token IDs to stop generation on
    stop_token_ids: list[int] | None = None
    eos_token_id = tokenizer.eos_token_id
    if isinstance(eos_token_id, int):
        stop_token_ids = [eos_token_id]
    if eos_token_id is not None:
        logger.info(
            f"Stop token IDs: {eos_token_id} decoded: {tokenizer.decode(eos_token_id)}"
        )
    else:
        logger.info("No EOS token ID found")

    num_prompts = len(conversations)
    total_conversations = num_prompts * group_size

    logger.info(
        f"Generating responses for {num_prompts} prompts × {group_size} group_size = {total_conversations} total conversations, "
        f"max_tokens={config.max_new_tokens}, temp={temperature:.3f}"
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
            logger.info(f"BF DEBUG first turn {n_responses} {len(batch_conversations)}")
        else:
            # Subsequent steps: batch all existing conversations
            batch_conversations = flat_conversations
            n_responses = 1
            logger.info(
                f"BF DEBUG second turn {n_responses} {len(batch_conversations)}"
            )

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
                temperature=temperature,
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
    entropy: float
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
    device: torch.device,
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

    # Pad token IDs
    batch_token_ids_t = [
        torch.tensor(token_ids, dtype=torch.long, device=device)
        for token_ids in token_ids_list
    ]
    batch_token_ids_t = torch.nn.utils.rnn.pad_sequence(
        batch_token_ids_t, batch_first=True, padding_value=pad_token_id
    )

    # Pad assistant masks
    batch_assistant_masks_t = [
        torch.tensor(assistant_mask, dtype=torch.bool, device=device)
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

    # Debug: Log advantage computation details
    logger.debug(
        f"preprocess_batch: n_prompts_per_step={n_prompts_per_step}, n_groups={n_groups}, "
        f"raw_rewards - mean={raw_reward_tensor.mean().item():.4f}, "
        f"std={raw_reward_tensor.std().item():.4f}, "
        f"min={raw_reward_tensor.min().item():.4f}, max={raw_reward_tensor.max().item():.4f}, "
        f"means_per_prompt={means.squeeze().tolist()}, "
        f"advantages - mean={advantages.mean().item():.6f}, "
        f"std={advantages.std().item():.6f}, "
        f"min={advantages.min().item():.6f}, max={advantages.max().item():.6f}, "
        f"zeros={torch.sum(advantages == 0).item()}/{len(advantages)}, "
        f"all_zero={torch.all(advantages == 0).item()}"
    )

    return {
        "batch_token_ids_t": batch_token_ids_t,
        "batch_assistant_masks_t": batch_assistant_masks_t,
        "target_token_ids": target_token_ids,
        "target_masks": target_masks,
        "advantages": advantages,
    }


def get_response_log_probs(
    model: torch.nn.Module,
    input_ids: torch.Tensor,
    labels: torch.Tensor,
) -> torch.Tensor:
    """
    Compute log probabilities for given labels efficiently.
    Uses F.log_softmax which is more memory efficient than manual computation.
    """
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


def _get_logprobs(
    model: nn.Module,
    batch_token_ids_t: T,
    dtype: torch.dtype,
    device: torch.device,
) -> T:
    """
    DEPRECATED: This function is unused and kept for reference only.
    Use get_response_log_probs instead which is more memory efficient.
    """
    # Use mixed precision for forward pass to reduce memory usage
    # Use autocast even without scaler for BFloat16 models to save memory
    fwd_ctx = (
        torch.autocast(device_type="cuda", enabled=True, dtype=dtype)
        if device.type == "cuda"
        else nullcontext()
    )
    with fwd_ctx:
        logits: T = model(batch_token_ids_t).logits

    # Get the cross entropy loss of the label and generated tokens
    # Slice logits to match target tokens (exclude first position)
    next_token_logits = logits[:, :-1]

    # Clear logits from memory immediately after use
    del logits
    clear_memory()

    return next_token_logits


def update_policy(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    episodes: List[Episode],
    micro_batch_size: int,
    pad_token_id: int,
    max_grad_norm: float,
    device: torch.device,
    algorithm: AlgorithmChoice,
    tokenizer: PreTrainedTokenizerBase,
    apply_loss: bool = True,
    entropy_coef: float = 0.0,
    metrics_wrapper: MetricsWrapper | None = None,
    step: int | None = None,
    gradient_accumulation_steps: int | None = None,
    dtype: torch.dtype = torch.bfloat16,
) -> UpdatePolicyResults:
    """
    Once episodes are generated, use them to update the policy
    by computing the loss from the reward and generated logits.
    This implements a number of different algorithms.
    """
    update_start_time = time.perf_counter()

    # Debug: Log overall episode statistics
    all_rewards = [e.reward for e in episodes]
    logger.debug(
        f"update_policy: Total episodes={len(episodes)}, "
        f"rewards - mean={sum(all_rewards) / len(all_rewards):.4f}, "
        f"min={min(all_rewards):.4f}, max={max(all_rewards):.4f}, "
        f"std={torch.tensor(all_rewards).std().item():.4f}, "
        f"all_zero={all(r == 0 for r in all_rewards)}"
    )

    # Pre-tokenize all episodes once
    for episode in episodes:
        get_token_ids_and_assistant_mask(episode.conversation, tokenizer)

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

    # Preprocess episodes
    logger.info("Preprocessing all episodes...")
    all_preprocessed: list[PreprocessedBatch] = []
    for micro_batch_start in range(0, len(episodes), micro_batch_size):
        micro_batch_end = min(micro_batch_start + micro_batch_size, len(episodes))
        batch_episodes = episodes[micro_batch_start:micro_batch_end]
        ppb = preprocess_batch(batch_episodes, tokenizer, pad_token_id, device)
        all_preprocessed.append(ppb)

    # Compute reference logprobs
    logger.info("Computing reference logprobs...")
    ref_logprobs_all: list[T] = []
    with torch.no_grad():
        for ppb in all_preprocessed:
            ref_logprobs = (
                get_response_log_probs(
                    model, ppb["batch_token_ids_t"], ppb["target_token_ids"]
                )
                .detach()
                .cpu()
            )  # Move to CPU to save GPU memory
            ref_logprobs_all.append(ref_logprobs)
            del ref_logprobs
        clear_memory()

    logger.info("Starting gradient updates...")
    epoch_kl_divs = []
    epoch_policy_ratios = []

    # Iterate over micro-batches using pre-computed data
    micro_batch_idx = 0
    for micro_batch_start in tqdm(
        range(0, len(episodes), micro_batch_size), desc="Micro-batches"
    ):
        micro_batch_end = min(micro_batch_start + micro_batch_size, len(episodes))
        batch_episodes = episodes[micro_batch_start:micro_batch_end]
        micro_batch_start_time = time.perf_counter()

        # Debug: Log episode rewards
        batch_rewards = [e.reward for e in batch_episodes]
        logger.debug(
            f"Micro-batch {micro_batch_idx}: Episode rewards - "
            f"mean={sum(batch_rewards) / len(batch_rewards):.4f}, "
            f"min={min(batch_rewards):.4f}, max={max(batch_rewards):.4f}, "
            f"std={torch.tensor(batch_rewards).std().item():.4f}, "
            f"all_zero={all(r == 0 for r in batch_rewards)}"
        )

        # Use pre-computed preprocessed batch
        ppb = all_preprocessed[micro_batch_idx]

        # Debug: Log advantage statistics
        advantages = ppb["advantages"]
        mask: T = ppb["target_masks"]
        logger.debug(
            f"Micro-batch {micro_batch_idx}: Advantages - "
            f"mean={advantages.mean().item():.6f}, "
            f"std={advantages.std().item():.6f}, "
            f"min={advantages.min().item():.6f}, "
            f"max={advantages.max().item():.6f}, "
            f"zeros={torch.sum(advantages == 0).item()}/{len(advantages)}, "
            f"all_zero={torch.all(advantages == 0).item()}"
        )

        # Debug: Log mask statistics
        logger.debug(
            f"Micro-batch {micro_batch_idx}: Target mask - "
            f"shape={mask.shape}, "
            f"total_tokens={mask.numel()}, "
            f"masked_tokens={mask.sum().item()}, "
            f"unmasked_tokens={(~mask).sum().item()}, "
            f"mask_ratio={mask.sum().item() / mask.numel():.4f}, "
            f"tokens_per_sample={mask.sum(dim=-1).float().mean().item():.2f}"
        )

        # Get pre-computed reference logprobs
        ref_logprobs = ref_logprobs_all[micro_batch_idx].to(device)

        # Compute policy logprobs (single forward pass with gradients)
        policy_logprobs = get_response_log_probs(
            model,
            ppb["batch_token_ids_t"],
            ppb["target_token_ids"],
        )

        # Debug: Log logprobs statistics
        with torch.no_grad():
            policy_logprobs_masked = policy_logprobs[mask]
            ref_logprobs_masked = ref_logprobs[mask]
            logger.debug(
                f"Micro-batch {micro_batch_idx}: Policy logprobs (masked) - "
                f"mean={policy_logprobs_masked.mean().item():.6f}, "
                f"std={policy_logprobs_masked.std().item():.6f}, "
                f"min={policy_logprobs_masked.min().item():.6f}, "
                f"max={policy_logprobs_masked.max().item():.6f}"
            )
            logger.debug(
                f"Micro-batch {micro_batch_idx}: Reference logprobs (masked) - "
                f"mean={ref_logprobs_masked.mean().item():.6f}, "
                f"std={ref_logprobs_masked.std().item():.6f}, "
                f"min={ref_logprobs_masked.min().item():.6f}, "
                f"max={ref_logprobs_masked.max().item():.6f}"
            )
            logprob_diff = policy_logprobs_masked - ref_logprobs_masked
            logger.debug(
                f"Micro-batch {micro_batch_idx}: Logprob diff (policy - ref, masked) - "
                f"mean={logprob_diff.mean().item():.6f}, "
                f"std={logprob_diff.std().item():.6f}, "
                f"min={logprob_diff.min().item():.6f}, "
                f"max={logprob_diff.max().item():.6f}"
            )

        # get the importance weights
        # use exp here to avoid numerical instability
        ratio = torch.exp(policy_logprobs - ref_logprobs)
        # TODO normalize per group
        # advantages already contains only the current micro-batch, no need to slice
        micro_batch_adv = advantages.unsqueeze(-1).to(device)

        # Debug: Log ratio statistics
        with torch.no_grad():
            ratio_masked = ratio[mask]
            logger.debug(
                f"Micro-batch {micro_batch_idx}: Ratio (masked) - "
                f"mean={ratio_masked.mean().item():.6f}, "
                f"std={ratio_masked.std().item():.6f}, "
                f"min={ratio_masked.min().item():.6f}, "
                f"max={ratio_masked.max().item():.6f}, "
            )
            logger.debug(
                f"Micro-batch {micro_batch_idx}: Advantages (expanded) - "
                f"shape={micro_batch_adv.shape}, "
                f"mean={micro_batch_adv.mean().item():.6f}, "
            )

        per_token_loss: T = -ratio * micro_batch_adv

        # Debug: Log per-token loss before masking
        with torch.no_grad():
            per_token_loss_masked = per_token_loss[mask]
            logger.debug(
                f"Micro-batch {micro_batch_idx}: Per-token loss (before masking) - "
                f"mean={per_token_loss_masked.mean().item():.6f}, "
                f"std={per_token_loss_masked.std().item():.6f}, "
                f"min={per_token_loss_masked.min().item():.6f}, "
                f"max={per_token_loss_masked.max().item():.6f}, "
            )

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
        # This avoids the issue where averaging per-prompt losses zeros out due to normalized advantages
        total_masked_tokens = mask.sum().clamp_min(1)
        loss = masked_loss.sum() / total_masked_tokens / gradient_accumulation_steps

        # Debug: Log masked loss and loss per prompt
        with torch.no_grad():
            masked_mean = masked_loss[mask].mean().item() if mask.sum() > 0 else 0.0
            denom = mask.sum(dim=-1).clamp_min(1)
            loss_per_prompt = masked_loss.sum(dim=-1) / denom
            logger.debug(
                f"Micro-batch {micro_batch_idx}: Masked loss - "
                f"sum={masked_loss.sum().item():.6f}, "
                f"mean={masked_mean:.6f}"
            )
            logger.debug(
                f"Micro-batch {micro_batch_idx}: Loss per prompt - "
                f"mean={loss_per_prompt.mean().item():.6f}, "
                f"std={loss_per_prompt.std().item():.6f}, "
                f"min={loss_per_prompt.min().item():.6f}, "
                f"max={loss_per_prompt.max().item():.6f}, "
                f"denom_min={denom.min().item():.0f}, "
                f"denom_mean={denom.float().mean().item():.2f}"
            )

        logger.debug(
            f"Micro-batch {micro_batch_idx}: Final loss - "
            f"loss={loss.item():.8f}, "
            f"gradient_accumulation_steps={gradient_accumulation_steps}, "
            f"unscaled_mean={masked_loss.sum().item() / total_masked_tokens.item():.8f}, "
            f"total_masked_tokens={total_masked_tokens.item()}"
        )

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
                f"Updated weights after micro-batch {micro_batch_idx}, grad_norm: {grad_norm:.4f}"
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
        del loss_per_prompt
        del loss
        del ref_logprobs
        del batch_episodes
        # Note: ppb is from all_preprocessed list, don't delete it here
        clear_memory()

    # Clean up stored data
    del ref_logprobs_all, all_preprocessed
    clear_memory()

    # Log GPU utilization after compute_loss
    log_memory_usage("update_policy", metrics_wrapper=metrics_wrapper, step=step)

    # Return average loss and entropy across all micro-batches
    avg_loss = total_loss / num_micro_batches if num_micro_batches > 0 else 0.0
    avg_entropy = 0.0

    # Debug: Log final statistics
    logger.debug(
        f"update_policy: Final statistics - "
        f"avg_loss={avg_loss:.8f}, "
        f"total_loss={total_loss:.8f}, "
        f"num_micro_batches={num_micro_batches}, "
        f"grad_norm={grad_norm:.6f}"
    )

    update_duration = time.perf_counter() - update_start_time
    logger.info(f"Policy update completed in {update_duration:.2f}s")

    clear_memory()

    return {
        "loss": float(avg_loss),
        "grad_norm": float(grad_norm),
        "entropy": float(avg_entropy),
        "duration": update_duration,
    }

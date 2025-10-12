import dataclasses
from contextlib import nullcontext
from collections import defaultdict
from typing import Dict, List, TypedDict
import torch.nn.functional as F

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
    Episode,
    RewardFunction,
    Sample,
    TrainerConfig,
)
from minrl.utils import clear_memory, find_assistant_sections, log_conversation

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
    Runs for max_steps turns, and generates group_size completions
    for the first turn, then 1 completion per turn for subsequent turns.
    """

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

    logger.info(
        f"Generating responses for {len(conversations)} prompts, "
        f"max_tokens={config.max_new_tokens}, n={group_size}, "
        f"temp={temperature:.3f}"
    )

    # Store all generated responses for all steps
    # size is len(conversations) * group_size
    # indexing is [prompt_idx][group_idx]
    conversations_out: list[list[Conversation]] = [
        [list(c) for _ in range(group_size)] for c in conversations
    ]

    # For turn 1, generate group_size responses, and add them to separate conversations.
    # For subsequent turns, generate 1 response per conversation so the group size is maintained.
    for step_idx in tqdm(range(max_steps), desc="Steps"):
        # Tokenize the conversations and convert to input for vLLM
        if step_idx == 0:
            # First step: use the first conversation from each group for tokenization
            current_conversations = [
                conversations_out[i][0] for i in range(len(conversations_out))
            ]
        else:
            # Subsequent steps: generate a new completion for each conversation in each group
            # Flatten the conversations in the order expected by vLLM: [group0_conv0, group0_conv1, ..., group1_conv0, group1_conv1, ...]
            current_conversations = [
                conversations_out[i][j]
                for i in range(len(conversations_out))
                for j in range(group_size)
            ]
        templated_conversations = tokenizer.apply_chat_template(
            current_conversations,  # pyright: ignore[reportArgumentType]
            tokenize=False,
            add_generation_prompt=True,
        )
        tokenized_conversations = tokenizer(templated_conversations)["input_ids"]  # pyright: ignore[reportArgumentType]
        vllm_input: list[TokensPrompt] = [
            {
                "prompt_token_ids": tokenized_conversation,
            }
            for tokenized_conversation in tokenized_conversations  # type: ignore
        ]

        # Generate list of n_conversations * group_size responses
        # Convert to proper type for SamplingParams
        outputs_for_step = vllm_model.generate(
            vllm_input,
            sampling_params=SamplingParams(
                max_tokens=config.max_new_tokens,
                # temperature=temperature,
                # If past the first turn, only generate one response per conversation
                n=group_size if step_idx == 0 else 1,
                stop_token_ids=stop_token_ids,
            ),
        )

        # Parse out the responses, and add to conversation
        if step_idx == 0:
            # First step: add each response to its corresponding conversation in the group
            for i in range(len(outputs_for_step)):
                for j in range(len(outputs_for_step[i].outputs)):
                    generated_text = outputs_for_step[i].outputs[j].text
                    # Only add this response to the corresponding conversation
                    conversations_out[i][j].append(
                        {"role": "assistant", "content": generated_text}
                    )
        else:
            # Subsequent steps: add each response to its corresponding conversation
            # The flattened order is [group0_conv0, group0_conv1, ..., group1_conv0, group1_conv1, ...]
            flattened_idx = 0
            for i in range(len(outputs_for_step)):
                for j in range(len(outputs_for_step[i].outputs)):
                    generated_text = outputs_for_step[i].outputs[j].text
                    # Calculate the group and conversation indices from the flattened index
                    group_idx = flattened_idx // group_size
                    conv_idx = flattened_idx % group_size
                    conversations_out[group_idx][conv_idx].append(
                        {"role": "assistant", "content": generated_text}
                    )
                    flattened_idx += 1

        # Clear CUDA cache
        clear_memory()

    # Create episodes from all generated responses
    episodes: List[Episode] = []
    for i, (conversations, sample) in enumerate(zip(conversations_out, samples)):
        for j in range(group_size):
            # For the first step, create multiple episodes (one per response in group)
            reward = reward_function(conversations[j], sample)
            logger.info(f"Episode {j}: reward: {reward}")
            conv_to_log = [msg for msg in conversations[j] if msg["role"] != "system"]
            log_conversation(conv_to_log)
            episode = Episode(
                group_index=i,
                answer_index=j,
                sample=sample,
                reward=reward,
                conversation=conversations[j],
            )
            episodes.append(episode)

    return episodes


def compute_entropy(logits: torch.Tensor) -> torch.Tensor:
    # Use log_softmax to avoid numerical instability
    logp = F.log_softmax(logits, dim=-1)
    # Compute entropy: -sum(p * log(p)) = -sum(exp(logp) * logp)
    entropy = -(logp.exp() * logp).sum(dim=-1)
    return entropy


def get_token_ids_and_assistant_mask(
    conversation: Conversation,
    tokenizer: PreTrainedTokenizerBase,
    episode: Episode | None = None,
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
        std = rewards.std(unbiased=False)  # Use biased std for small groups

        # Handle edge cases
        if torch.isnan(std) or std == 0:
            # If all rewards are identical, keep them as-is (will be zero after mean subtraction)
            # This happens when all responses get same reward
            # Small epsilon prevents completely killing gradient signal
            normalized_rewards = torch.zeros_like(rewards)
        else:
            normalized_rewards = (rewards - mean) / (
                std + 1e-8
            )  # Add epsilon for numerical stability

        for episode, norm_reward in zip(group, normalized_rewards):
            normalized_episodes.append(
                dataclasses.replace(episode, reward=float(norm_reward))
            )

    return normalized_episodes


def compute_algorithm_loss(
    logprobs: torch.Tensor,
    target_masks: torch.Tensor,
    batch_advantages: torch.Tensor,
    algorithm: AlgorithmChoice,
    n_target_tokens: int,
    entropy: torch.Tensor | None = None,
    entropy_coef: float = 0.0,
) -> torch.Tensor:
    """
    Math: Policy gradient theorem says gradient should be: grad = E[grad log π(a|s) * A]
    where A is advantage/reward. Since we minimize loss, we set:
    loss = -log π(a|s) * A

    Here logprobs = -cross_entropy = log π(a|s) (negative values)
    So: advantage_t = logprobs * reward = (negative) * (pos/neg)
        loss = -advantage_t.sum()
    This gives correct gradients via SGD.

    Entropy regularization encourages exploration by penalizing low entropy (peaked distributions).
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


class EpisodeWithTokens(TypedDict):
    episode: Episode
    token_ids: list[int]
    assistant_mask: list[bool]


def process_batch(
    model: nn.Module,
    episodes: List[Episode],
    tokenizer: PreTrainedTokenizerBase,
    pad_token_id: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, int]:
    """
    Preprocess a single batch of episodes to compute logprobs, masks, rewards, and entropy,
    which are all needed for computing the loss.
    """
    # Extract token IDs and assistant masks for each episode
    token_ids_list: list[list[int]] = []
    assistant_mask_list: list[list[bool]] = []
    n_target_tokens = 0
    for episode in episodes:
        token_ids, assistant_mask = get_token_ids_and_assistant_mask(
            episode.conversation, tokenizer, episode
        )
        token_ids_list.append(token_ids)
        assistant_mask_list.append(assistant_mask)
        n_target_tokens += len(token_ids)
    # Pad token IDs to the same length using PyTorch's pad_sequence
    batch_token_ids_t = [
        torch.tensor(token_ids, dtype=torch.long, device=device)
        for token_ids in token_ids_list
    ]
    batch_token_ids_t = torch.nn.utils.rnn.pad_sequence(
        batch_token_ids_t, batch_first=True, padding_value=pad_token_id
    )

    # Pad assistant masks to the same length using PyTorch's pad_sequence
    batch_assistant_masks_t = [
        torch.tensor(assistant_mask, dtype=torch.bool, device=device)
        for assistant_mask in assistant_mask_list
    ]
    batch_assistant_masks_t = torch.nn.utils.rnn.pad_sequence(
        batch_assistant_masks_t, batch_first=True, padding_value=False
    )

    # Shift tokens and masks for next-token prediction
    target_token_ids = batch_token_ids_t[:, 1:]
    target_assistant_masks = batch_assistant_masks_t[:, 1:]
    pad_masks = target_token_ids != pad_token_id

    # Combine assistant mask with padding mask
    target_masks = target_assistant_masks & pad_masks

    # advantage is just normalized reward
    batch_rewards = [episode.reward for episode in episodes]
    batch_rewards_t = torch.tensor(batch_rewards, device=device, dtype=torch.float32)

    # Use mixed precision for forward pass to reduce memory usage
    # Use autocast even without scaler for BFloat16 models to save memory
    fwd_ctx = (
        torch.autocast(device_type="cuda", enabled=True)
        if device.type == "cuda"
        else nullcontext()
    )
    with fwd_ctx:
        logits: torch.Tensor = model(batch_token_ids_t).logits

    # Get the cross entropy loss of the label and generated tokens
    # Slice logits to match target tokens (exclude first position)
    next_token_logits = logits[:, :-1].float()

    # Clear logits from memory immediately after use
    del logits
    clear_memory()

    bs = batch_token_ids_t.shape[0]
    logprobs = -F.cross_entropy(
        next_token_logits.reshape(-1, next_token_logits.size(-1)),
        target_token_ids.reshape(-1),
        ignore_index=pad_token_id,
        reduction="none",
    ).reshape(bs, -1)

    # Calculate entropy only for target positions - memory efficient version
    # Compute entropy in chunks to avoid creating large intermediate tensors
    bs, seq_len, vocab_size = next_token_logits.shape
    target_masks_flat = target_masks.reshape(-1)

    # Only compute entropy for positions that are actually targets
    target_positions = target_masks_flat.nonzero(as_tuple=True)[0]
    if len(target_positions) > 0:
        next_token_logits_flat = next_token_logits.reshape(-1, vocab_size)
        target_logits = next_token_logits_flat[target_positions]

        # Use mixed precision for entropy computation
        with fwd_ctx:
            token_entropy = compute_entropy(target_logits)

        # single entropy value for the sequence
        entropy = token_entropy.sum() / n_target_tokens
    else:
        entropy = torch.tensor(0.0, device=device)

    del batch_token_ids_t, target_token_ids, next_token_logits
    clear_memory()

    return logprobs, target_masks, batch_rewards_t, entropy, n_target_tokens


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
) -> UpdatePolicyResults:
    """
    Once episodes are generated, use them to update the policy
    by computing the loss from the reward and generated logits.
    This implements a number of different algorithms.
    """
    if algorithm == "grpo":
        episodes = normalize_rewards_per_group(episodes)

    # Pre-tokenize all episodes once
    for episode in episodes:
        get_token_ids_and_assistant_mask(episode.conversation, tokenizer)

    total_entropy = torch.tensor(0.0, device=device)
    total_loss = torch.tensor(0.0, device=device)
    grad_norm = 0.0
    num_micro_batches = 0

    # Compute number of micro-batches up-front for correct gradient accumulation scaling
    # This makes the accumulated gradient equivalent to averaging over the full batch
    total_micro_batches = (len(episodes) + micro_batch_size - 1) // micro_batch_size

    logger.info(
        f"Computing backward pass for {total_micro_batches} micro-batches of size {micro_batch_size}"
    )

    # Iterate over micro-batches
    for i in range(0, len(episodes), micro_batch_size):
        j = min(i + micro_batch_size, len(episodes))
        batch_episodes = episodes[i:j]

        # Process the batch
        logprobs, target_masks, batch_rewards_t, batch_entropy, n_target_tokens = (
            process_batch(
                model=model,
                episodes=batch_episodes,
                tokenizer=tokenizer,
                pad_token_id=pad_token_id,
                device=device,
            )
        )

        reward_mean, reward_std = batch_rewards_t.mean(), batch_rewards_t.std()
        logger.info(
            f"Micro-batch {i}: reward mean: {reward_mean}, reward std: {reward_std}"
        )
        if reward_std == 0:
            logger.warning("Reward std is 0, skipping micro-batch")
            continue

        # Compute algorithm-specific loss
        batch_loss = compute_algorithm_loss(
            logprobs,
            target_masks,
            batch_rewards_t,
            algorithm,
            n_target_tokens,
            entropy=batch_entropy,
            entropy_coef=entropy_coef,
        )

        # Track total loss for logging (unscaled). We scale only for backward
        total_loss += batch_loss
        total_entropy += batch_entropy
        num_micro_batches += 1

        # Backward pass - gradients accumulate naturally
        if apply_loss:
            # Scale per micro-batch loss by number of accumulation steps so that
            # the accumulated gradient matches the average gradient over the full batch
            scaled_loss = batch_loss / max(total_micro_batches, 1)
            scaled_loss.backward()

        # Clear intermediate tensors to save memory
        del logprobs, target_masks, batch_rewards_t, batch_loss
        clear_memory()

    # Update parameters once after all gradients accumulated
    if apply_loss:
        grad_norm = torch.nn.utils.clip_grad_norm_(
            model.parameters(), max_norm=max_grad_norm
        )
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        clear_memory()

    # Return average loss and entropy across all micro-batches
    avg_loss = total_loss / num_micro_batches if num_micro_batches > 0 else 0.0
    avg_entropy = total_entropy / num_micro_batches if num_micro_batches > 0 else 0.0

    return {
        "loss": float(avg_loss),
        "grad_norm": float(grad_norm),
        "entropy": float(avg_entropy),
    }

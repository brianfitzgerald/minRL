import re
from pydoc import html
from typing import Dict, List

import numpy as np
import torch
from loguru import logger

from minrl.constants import Episode
from minrl.metrics import MetricsWrapper


def clean_observation(obs: str) -> str:
    """Clean the observation to remove the score and move information."""
    obs = obs.strip()
    lines = obs.split("\n")
    processed_lines = []
    for line in lines:
        # Rule 1: If a line starts with '^', discard this line and all subsequent lines
        if line.lstrip().startswith("^") or line.lstrip().startswith(">"):
            break

        lstripped_line = line.lstrip()

        # Filter for any line starting with '>' or containing "Purpose:"
        if lstripped_line.startswith(">") or "Purpose:" in lstripped_line:
            continue

        cleaned_line = re.sub(r"Score: \d+ Moves: \d+", "", line).strip()
        processed_lines.append(cleaned_line)

    return "\n".join(processed_lines).strip()


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
    metrics_wrapper.add_scalar("train/loss", loss, step)
    metrics_wrapper.add_scalar("train/mean_reward", mean_reward, step)
    metrics_wrapper.add_scalar("train/std_reward", std_reward, step)
    metrics_wrapper.add_scalar("train/grad_norm", grad_norm, step)
    metrics_wrapper.add_scalar(
        "train/num_finished_episodes", num_finished_episodes, step
    )
    metrics_wrapper.add_scalar("train/learning_rate", lr, step)
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
        "num_finished_episodes": float(num_finished_episodes),
    }
    if temperature is not None:
        log_dict["temperature"] = temperature
    logger.info(f"Metrics: {log_dict}")
    return log_dict


NEWLINE_TOKEN_ID = 198  # Token ID for newline character


def find_assistant_sections(
    all_token_ids: list[int],
    im_start_token: list[int],
    assistant_role_tokens: list[int],
    im_end_token: list[int],
) -> list[tuple[int, int]]:
    """Find all assistant sections in tokenized conversation."""
    sections = []
    i = 0

    while i < len(all_token_ids):
        # Look for <|im_start|>assistant pattern
        start_pattern_len = len(im_start_token) + len(assistant_role_tokens)
        if (
            i + start_pattern_len < len(all_token_ids)
            and all_token_ids[i : i + len(im_start_token)] == im_start_token
        ):
            next_pos = i + len(im_start_token)
            role_end = next_pos + len(assistant_role_tokens)
            if all_token_ids[next_pos:role_end] == assistant_role_tokens:
                # Found assistant section start
                mark_start = i
                search_pos = role_end

                # Find the end of this assistant section
                end_pos = len(all_token_ids)  # Default to end of sequence
                while search_pos + len(im_end_token) <= len(all_token_ids):
                    end_token_end = search_pos + len(im_end_token)
                    if all_token_ids[search_pos:end_token_end] == im_end_token:
                        end_pos = end_token_end

                        # Include trailing newline if it's the last token
                        is_last_token = end_pos == len(all_token_ids) - 1
                        is_newline = (
                            end_pos < len(all_token_ids)
                            and all_token_ids[end_pos] == NEWLINE_TOKEN_ID
                        )
                        if is_last_token and is_newline:
                            end_pos += 1
                        break
                    search_pos += 1

                sections.append((mark_start, end_pos))
                i = end_pos
            else:
                i += 1
        else:
            i += 1

    return sections

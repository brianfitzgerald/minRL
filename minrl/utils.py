import gc
import os
import re
from pydoc import html
from typing import TypedDict

import numpy as np
import psutil
import torch
from loguru import logger

from minrl.constants import Conversation, ConversationMessage, Episode, Role
from minrl.metrics import MetricsWrapper

USING_MPS = torch.backends.mps.is_available() and torch.backends.mps.is_built()
if not USING_MPS:
    from pynvml import (  # pyright: ignore[reportMissingImports]
        nvmlDeviceGetHandleByIndex,
        nvmlDeviceGetMemoryInfo,
        nvmlDeviceGetUtilizationRates,
    )  # pyright: ignore[reportMissingImports]


def clean_observation(obs: str) -> str:
    """Clean the observation to remove the score and move information."""
    obs = obs.strip()
    lines = obs.split("\n")
    processed_lines = []
    for line in lines:
        lstripped_line = line.lstrip()

        # Rule 1: If a line starts with '^', discard this line and all subsequent lines
        if lstripped_line.startswith("^"):
            break

        # Filter for any line starting with '>' or containing "Purpose:"
        if lstripped_line.startswith(">") or "Purpose:" in lstripped_line:
            continue

        cleaned_line = re.sub(r"Score: \d+ Moves: \d+", "", line).strip()
        processed_lines.append(cleaned_line)

    return "\n".join(processed_lines).strip()


NEWLINE_TOKEN_ID = 198  # Token ID for newline character


def find_assistant_sections(
    all_token_ids: list[int],
    im_start_token: list[int],
    assistant_role_tokens: list[int],
    im_end_token: list[int],
    newline_token_id: int | None = None,
) -> list[tuple[int, int]]:
    """Find all assistant sections in tokenized conversation."""
    sections = []
    i = 0

    # Use provided newline token ID or fall back to default (Qwen/GPT-style)
    if newline_token_id is None:
        newline_token_id = NEWLINE_TOKEN_ID

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
                            and all_token_ids[end_pos] == newline_token_id
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


def clear_memory():
    """Clear memory more aggressively to prevent lockups during training."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        # Force garbage collection of CUDA tensors
        torch.cuda.synchronize()

        # Reset peak memory stats to get accurate measurements
        torch.cuda.reset_peak_memory_stats()
        # Force garbage collection again after CUDA operations
        gc.collect()


class GPUStats(TypedDict):
    cpu_memory_mb: float
    gpu_memory_allocated: float
    gpu_memory_reserved: float
    gpu_memory_total: float
    gpu_memory_percentage: float
    gpu_utilization: float


def log_memory_usage(
    label: str = "Memory usage",
    metrics_wrapper: MetricsWrapper | None = None,
    step: int | None = None,
) -> GPUStats:
    """
    Get current memory usage in MB and percentage of total VRAM available, plus GPU utilization.
    If metrics_wrapper is provided, add the memory usage to the metrics wrapper.
    """
    process = psutil.Process(os.getpid())
    cpu_memory_mb = process.memory_info().rss / 1024 / 1024
    (
        gpu_memory_allocated,
        gpu_memory_percentage,
        gpu_memory_reserved,
        gpu_memory_total,
        gpu_utilization,
    ) = 0.0, 0.0, 0.0, 0.0, 0.0

    if torch.cuda.is_available():
        # Try NVML first
        try:
            if not USING_MPS:
                handle = nvmlDeviceGetHandleByIndex(0)
                info = nvmlDeviceGetMemoryInfo(handle)
                gpu_memory_allocated = info.used
                gpu_memory_reserved = info.reserved
                gpu_memory_total = info.total

                # Get GPU utilization
                utilization = nvmlDeviceGetUtilizationRates(handle)
                gpu_utilization = float(utilization.gpu)

                # Convert to MB
                gpu_memory_allocated = gpu_memory_allocated / 1024 / 1024  # pyright: ignore[reportOperatorIssue]
                gpu_memory_reserved = gpu_memory_reserved / 1024 / 1024  # pyright: ignore[reportOperatorIssue]
                gpu_memory_total = gpu_memory_total / 1024 / 1024  # pyright: ignore[reportOperatorIssue]
                gpu_memory_percentage = (gpu_memory_allocated / gpu_memory_total) * 100
        except Exception as e:
            logger.debug(f"NVML failed, trying PyTorch CUDA: {e}")
            # Fallback to PyTorch CUDA
            try:
                if torch.cuda.is_initialized():
                    gpu_memory_allocated = torch.cuda.memory_allocated()
                    gpu_memory_reserved = torch.cuda.memory_reserved()
                    gpu_memory_total = torch.cuda.get_device_properties(0).total_memory

                    # Convert to MB
                    gpu_memory_allocated = gpu_memory_allocated / 1024 / 1024  # pyright: ignore[reportOperatorIssue]
                    gpu_memory_reserved = gpu_memory_reserved / 1024 / 1024  # pyright: ignore[reportOperatorIssue]
                    gpu_memory_total = gpu_memory_total / 1024 / 1024  # pyright: ignore[reportOperatorIssue]
                    gpu_memory_percentage = (
                        gpu_memory_allocated / gpu_memory_total
                    ) * 100
                    # Note: PyTorch doesn't provide GPU utilization, so it remains 0
                else:
                    logger.debug(
                        "CUDA is available but not initialized, skipping GPU stats"
                    )
            except Exception as e2:
                logger.debug(f"PyTorch CUDA also failed: {e2}, skipping GPU stats")

    if label:
        logger.info(
            f"{label} - CPU: {cpu_memory_mb:.1f}MB, GPU: {gpu_memory_allocated:.1f}MB ({gpu_memory_percentage:.1f}%), GPU Util: {gpu_utilization:.1f}%"
        )

    out_dict: GPUStats = {
        "cpu_memory_mb": float(cpu_memory_mb),
        "gpu_memory_allocated": float(gpu_memory_allocated),
        "gpu_memory_reserved": float(gpu_memory_reserved),
        "gpu_memory_total": float(gpu_memory_total),
        "gpu_memory_percentage": float(gpu_memory_percentage),
        "gpu_utilization": float(gpu_utilization),
    }

    if metrics_wrapper is not None:
        assert step is not None
        for k, v in out_dict.items():
            metrics_wrapper.add_scalar(f"resources/{label}_{k}", v, step)  # pyright: ignore[reportArgumentType]

    return out_dict


def log_conversation(
    conversation: Conversation, only_roles: list[Role] | None = None
) -> None:
    """
    Log a Conversation to the console using loguru with colored output based on roles.

    Args:
        conversation: The conversation to log
    """
    # Color mapping for different roles using ANSI color codes
    role_colors = {
        "system": "\033[36m",  # Cyan
        "user": "\033[32m",  # Green
        "assistant": "\033[34m",  # Blue
        "function": "\033[33m",  # Yellow
        "tool": "\033[35m",  # Magenta
        "unknown": "\033[31m",  # Red
    }
    reset_color = "\033[0m"

    # Build the complete log message
    log_lines = []

    for i, message in enumerate[ConversationMessage](conversation, 1):
        role = message.get("role", "unknown")
        if only_roles is not None and role not in only_roles:
            continue
        content = message.get("content", "")

        if isinstance(content, str):
            content_str = content
        elif content is None:
            content_str = ""
        else:
            content_str = str(content)

        # Get color for role, default to red for unknown roles
        color = role_colors.get(role, role_colors["unknown"])

        # Add the message with color and clear separation
        log_lines.append(f"\n{color}[{i}] {role.upper()}{reset_color}\n")
        log_lines.append(f"{color}{content_str}{reset_color}\n")

    logger.info("".join(log_lines))

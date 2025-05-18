import re
from typing import Any, Dict, List, TypedDict


def format_reward_function(response: str) -> float:
    """
    Checks if the response follows the format <think>...</think><answer>...</answer>
    """

    think_regex = r"<think>.*?<\/think>"
    answer_regex = r"<answer>.*?<\/answer>"
    full_format_regex = r"^<think>.*?<\/think>\n<answer>.*?<\/answer>$"

    think_match = re.search(think_regex, response, re.DOTALL)
    answer_match = re.search(answer_regex, response, re.DOTALL)
    full_format_match = re.match(full_format_regex, response, re.DOTALL)

    if full_format_match:
        return 1.0

    reward = 0.0

    if think_match:
        reward += 0.1

    if answer_match:
        reward += 0.5

    return reward


def answer_reward_function(
    response: str, numbers: List[int], target: int
) -> float:
    """
    Checks if the answer uses all numbers exactly once and evaluates to the target
    """
    answer_regex = r"<answer>(.*?)<\/answer>"
    answer_match = re.search(answer_regex, response, re.DOTALL)
    if not answer_match:
        return 0.0

    answer_content = answer_match.group(1)
    if not answer_content:
        return 0.0

    allowed_chars = r"^[0-9+\-*/() ]+$"
    if not re.match(allowed_chars, answer_content):
        return 0.0

    # Check if the answer uses all numbers exactly once
    used_numbers = [int(n) for n in re.findall(r"\d+", answer_content)]
    if sorted(used_numbers) != sorted(numbers):
        return 0.0

    # Check if the answer evaluates to the target
    try:
        result = eval(answer_content, {"__builtins__": None}, {})
        if abs(float(result) - float(target)) < 1e-5:
            return 1.0
    except:  # noqa: E722
        pass

    return 0.0


class CountdownSample(TypedDict):
    numbers: list[int]
    answer: int

def countdown_reward_function(
    response: str,
    sample: list[str]
) -> Dict[str, float]:
    """Reward function for Countdown Tasks.

    Total reward = 0.1 * format_reward + answer_reward
    Args:
        response: The response from the model.
        sample: A list containing [numbers_str, target_str] where numbers_str is a comma-separated list of numbers
               and target_str is the target number as a string.
    Returns:
        A dictionary containing the reward and individual reward components.
    """
    numbers_str, target_str = sample
    numbers = [int(n.strip()) for n in numbers_str.split(",")]
    target = int(target_str)
    
    format_reward = format_reward_function("<think>" + response)
    answer_reward = answer_reward_function(response, numbers, target)
    total_reward = format_reward * 0.1 + answer_reward
    
    return {
        "reward": total_reward,
        "format_reward": format_reward,
        "answer_reward": answer_reward,
    }

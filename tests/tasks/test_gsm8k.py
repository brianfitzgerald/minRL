import pytest
from unittest.mock import MagicMock, patch

from minrl.tasks.gsm8k import (
    GSM8KDataset,
    extract_solution,
    compute_score,
    SYSTEM_PROMPT,
    TEMPLATE,
)


# Tests for extract_solution function
@pytest.mark.parametrize(
    "solution_str,method,expected",
    [
        # Strict mode - requires #### format
        ("The answer is #### 42", "strict", "42"),
        ("After calculation #### 123", "strict", "123"),
        ("Multiple #### 10 and #### 20", "strict", "20"),  # Takes last one
        ("Price is #### 1,234", "strict", "1234"),  # Removes commas
        ("Cost #### 500", "strict", "500"),  # Dollar sign would prevent regex match
        ("Negative #### -15", "strict", "-15"),
        ("Decimal #### 3.14", "strict", "3.14"),
        ("No marker 42", "strict", None),  # No #### marker
        ("Empty ####", "strict", None),  # No number after ####
        # Flexible mode - extracts last number
        ("The answer is 42", "flexible", "42"),
        ("Multiple numbers 10 and 20", "flexible", "20"),  # Takes last one
        (
            "With comma 1,234",
            "flexible",
            "1,234",
        ),  # Regex captures commas within numbers
        ("Negative value is -15", "flexible", "-15"),  # Regex captures minus sign
        ("Decimal 3.14", "flexible", "3.14"),  # Regex captures decimal point
        ("Ends with period 42.", "flexible", "42."),  # Regex captures trailing period
        ("Just a dot.", "flexible", "."),  # Only the period is matched
        ("No numbers here", "flexible", None),
        ("Empty string", "flexible", None),
    ],
)
def test_extract_solution_parametrized(solution_str, method, expected):
    """Test extract_solution with various inputs and methods."""
    result = extract_solution(solution_str, method=method)
    assert result == expected


@pytest.mark.parametrize(
    "solution_str,expected",
    [
        # Test long strings that get clipped
        ("x" * 400 + "#### 99", "99"),  # Should clip but still find answer
        ("#### 11 " + "x" * 400, None),  # Answer at start gets clipped off
    ],
)
def test_extract_solution_clipping(solution_str, expected):
    """Test that long solutions are properly clipped to last 300 chars."""
    result = extract_solution(solution_str, method="strict")
    assert result == expected


def test_extract_solution_invalid_method():
    """Test that invalid method raises assertion error."""
    with pytest.raises(AssertionError):
        extract_solution("#### 42", method="invalid")


# Tests for compute_score function
@pytest.mark.parametrize(
    "solution_str,ground_truth,method,format_score,score,expected_score",
    [
        # Correct answers
        ("#### 42", "42", "flexible", 0.1, 1.0, 1.0),
        ("The answer is #### 100", "100", "strict", 0.1, 1.0, 1.0),
        # Wrong answers but correct format
        ("#### 42", "43", "flexible", 0.1, 1.0, 0.1),
        ("#### 100", "99", "strict", 0.2, 1.0, 0.2),
        # No answer extracted
        ("No number here", "42", "flexible", 0.1, 1.0, 0.0),
        ("Missing marker 42", "42", "strict", 0.1, 1.0, 0.0),
    ],
)
def test_compute_score_parametrized(
    solution_str, ground_truth, method, format_score, score, expected_score
):
    """Test compute_score with various scenarios."""
    result = compute_score(
        solution_str=solution_str,
        ground_truth=ground_truth,
        method=method,
        format_score=format_score,
        score=score,
    )
    assert result == expected_score


# Tests for GSM8KDataset class
@patch("minrl.tasks.gsm8k.load_dataset")
def test_gsm8k_dataset_initialization_train(mock_load_dataset):
    """Test GSM8KDataset initialization for train split."""
    # Mock the dataset
    mock_dataset = MagicMock()
    mock_dataset.__len__ = MagicMock(return_value=100)
    mock_dataset.__iter__ = MagicMock(return_value=iter([]))
    mock_load_dataset.return_value = mock_dataset

    dataset = GSM8KDataset(split="train", host="local", tokenizer=None)

    # Verify dataset was loaded correctly
    mock_load_dataset.assert_called_once_with("openai/gsm8k", "main", split="train")
    assert dataset.split == "train"
    assert len(dataset) == 100


@patch("minrl.tasks.gsm8k.load_dataset")
def test_gsm8k_dataset_initialization_eval(mock_load_dataset):
    """Test GSM8KDataset initialization for eval split."""
    # Mock the dataset with select method
    mock_dataset = MagicMock()
    mock_selected = MagicMock()
    mock_selected.__len__ = MagicMock(return_value=128)
    mock_selected.__iter__ = MagicMock(return_value=iter([]))
    mock_dataset.select = MagicMock(return_value=mock_selected)
    mock_load_dataset.return_value = mock_dataset

    dataset = GSM8KDataset(split="eval", host="local", tokenizer=None)

    # Verify eval uses "test" split and selects first 128 samples
    mock_load_dataset.assert_called_once_with("openai/gsm8k", "main", split="test")
    mock_dataset.select.assert_called_once_with(range(128))
    assert len(dataset) == 128


@patch("minrl.tasks.gsm8k.load_dataset")
def test_gsm8k_initial_conversation(mock_load_dataset):
    """Test that initial_conversation creates correct format."""
    mock_dataset = MagicMock()
    mock_dataset.__len__ = MagicMock(return_value=100)
    mock_dataset.__iter__ = MagicMock(return_value=iter([]))
    mock_load_dataset.return_value = mock_dataset

    dataset = GSM8KDataset(split="train", host="local", tokenizer=None)

    sample = {"question": "What is 2+2?", "answer": "#### 4"}
    conversation = dataset.initial_conversation(sample, sample_index=0)

    assert len(conversation) == 2
    assert conversation[0]["role"] == "system"
    assert conversation[0]["content"] == SYSTEM_PROMPT
    assert conversation[1]["role"] == "user"
    assert conversation[1]["content"] == TEMPLATE.format(question="What is 2+2?")


# Tests for reward_function
@pytest.mark.parametrize(
    "model_answer,ground_truth_answer,expected_reward",
    [
        # Correct answer
        ("#### 42", "#### 42", 1.0),
        ("The final answer is #### 100", "#### 100", 1.0),
        # Wrong answer but correct format (gets format_score of 0.1 by default)
        ("#### 42", "#### 43", 0.1),
        ("#### 100", "#### 99", 0.1),
        # Flexible mode matches (reward_function uses flexible mode by default)
        ("The answer is 42", "#### 42", 1.0),  # Both extract to "42"
        ("No answer provided", "#### 100", 0.0),  # No number in model answer
        # Invalid ground truth
        ("#### 42", "Invalid ground truth", 0.0),
    ],
)
def test_reward_function_parametrized(
    model_answer, ground_truth_answer, expected_reward
):
    """Test reward_function with various answer scenarios."""
    conversation = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": "What is the answer?"},
        {"role": "assistant", "content": model_answer},
    ]
    sample = {"question": "What is the answer?", "answer": ground_truth_answer}

    reward = GSM8KDataset.reward_function(conversation, sample)
    assert reward == expected_reward

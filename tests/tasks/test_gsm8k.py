import pytest
from minrl.tasks.gsm8k import GSM8KDataset, extract_solution, compute_score
from minrl.constants import Conversation


class TestGSM8KIntegration:
    """Integration tests that load the actual GSM8K dataset"""

    @pytest.mark.parametrize("split", ["train", "test"])
    def test_load_dataset_and_complete_workflow(self, split):
        """
        Integration test: Load real GSM8K dataset, get a sample,
        create conversation, mock LLM response, and score it.
        """
        # Load the actual dataset
        dataset = GSM8KDataset(split=split, host="local")

        # Verify dataset loaded correctly
        assert len(dataset) > 0
        if split == "train":
            assert len(dataset) == 7473
        else:  # test split
            assert len(dataset) == 1319

        # Get a real sample from the dataset
        sample = dataset[0]
        assert "question" in sample
        assert "answer" in sample

        # Create initial conversation from sample
        conversation = dataset.initial_conversation(sample, 0)
        assert len(conversation) == 1
        assert conversation[0]["role"] == "user"
        assert conversation[0]["content"] == sample["question"]

        # Mock LLM responses - test both correct and incorrect scenarios
        ground_truth = extract_solution(sample["answer"], method="strict")
        assert ground_truth is not None, "Sample should have valid answer format"

        # Scenario 1: Correct LLM response with proper format
        correct_response = (
            f"Let me solve this step by step.\n\nFinal answer: #### {ground_truth}"
        )
        correct_conversation: Conversation = [
            {"role": "user", "content": sample["question"]},
            {"role": "assistant", "content": correct_response},
        ]

        reward_correct = dataset.reward_function(correct_conversation, sample)
        assert reward_correct == 1.0, "Should get full reward for correct answer"

        # Scenario 2: Incorrect LLM response with proper format
        wrong_answer = "999999"  # Unlikely to match any real answer
        incorrect_response = f"Let me solve this.\n\nAnswer: #### {wrong_answer}"
        incorrect_conversation: Conversation = [
            {"role": "user", "content": sample["question"]},
            {"role": "assistant", "content": incorrect_response},
        ]

        reward_incorrect = dataset.reward_function(incorrect_conversation, sample)
        assert reward_incorrect == 0.0, "Should get no reward for wrong answer"

        # Scenario 3: No proper format in response
        no_format_response = f"The answer is {ground_truth}"
        no_format_conversation: Conversation = [
            {"role": "user", "content": sample["question"]},
            {"role": "assistant", "content": no_format_response},
        ]

        reward_no_format = dataset.reward_function(no_format_conversation, sample)
        assert reward_no_format == 0.0, "Should get no reward without proper format"

    def test_real_gsm8k_examples(self):
        """
        Integration test with known GSM8K examples to verify end-to-end behavior.
        Tests the full workflow with realistic problem-solving scenarios.
        """
        dataset = GSM8KDataset(split="test", host="local")

        # Get first few samples and test them
        samples_to_test = 5
        for i in range(samples_to_test):
            sample = dataset[i]

            # Verify sample structure
            assert "question" in sample
            assert "answer" in sample
            assert isinstance(sample["question"], str)
            assert isinstance(sample["answer"], str)

            # Verify answer has correct format
            ground_truth = extract_solution(sample["answer"], method="strict")
            assert ground_truth is not None, f"Sample {i} should have valid answer"

            # Test conversation initialization
            conversation = dataset.initial_conversation(sample, i)
            assert conversation[0]["content"] == sample["question"]

            # Simulate correct LLM response
            # Extract the actual ground truth from the sample's full solution
            correct_llm_response = sample[
                "answer"
            ]  # Use the full answer with reasoning
            full_conversation: Conversation = [
                {"role": "user", "content": sample["question"]},
                {"role": "assistant", "content": correct_llm_response},
            ]

            # Should get perfect score since we're using the ground truth
            reward = dataset.reward_function(full_conversation, sample)
            assert reward == 1.0, (
                f"Sample {i}: Should get perfect score with ground truth answer"
            )


class TestGSM8KHelperFunctions:
    """Unit tests for helper functions"""

    def test_extract_solution_strict_basic(self):
        """Test basic strict mode extraction"""
        assert extract_solution("Answer: #### 42", method="strict") == "42"
        assert extract_solution("#### 100", method="strict") == "100"
        assert extract_solution("No format here", method="strict") is None

    def test_extract_solution_flexible_basic(self):
        """Test basic flexible mode extraction"""
        assert extract_solution("The answer is 42", method="flexible") == "42"
        assert extract_solution("Total: 100 dollars", method="flexible") == "100"

    def test_compute_score_basic(self):
        """Test basic scoring logic"""
        # Correct answer
        assert (
            compute_score(
                solution_str="Steps...\n#### 50", ground_truth="50", method="strict"
            )
            == 1.0
        )

        # Wrong answer
        assert (
            compute_score(
                solution_str="Steps...\n#### 40", ground_truth="50", method="strict"
            )
            == 0.0
        )

        # No format
        assert (
            compute_score(
                solution_str="I don't know", ground_truth="50", method="strict"
            )
            == 0
        )

    def test_extract_solution_handles_commas_and_negatives(self):
        """Test extraction handles number formatting"""
        assert extract_solution("#### 1,234", method="strict") == "1234"
        assert extract_solution("#### -50", method="strict") == "-50"
        assert extract_solution("#### 3.14", method="strict") == "3.14"

    def test_extract_solution_long_string_clipping(self):
        """Test that very long strings are handled efficiently"""
        # Answer at the end should be found
        long_string = "x" * 500 + " #### 999"
        assert extract_solution(long_string, method="strict") == "999"

        # Answer too far from end should not be found (due to 300 char clip)
        long_string = "#### 111 " + "x" * 500
        assert extract_solution(long_string, method="strict") is None

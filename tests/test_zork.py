import os
from unittest.mock import MagicMock, patch

import pytest

from minrl.tasks.zork import (
    ZorkDataset,
    parse_command,
)


# Parametrized tests
@pytest.mark.parametrize(
    "command_input,expected_output",
    [
        ("<command>north</command>", "north"),
        ("<command>take sword</command>", "take sword"),
        ("COMMAND: <command>look</command>", "look"),
        ("\n<command>inventory</command>", "inventory"),
        ("<command>  go east  </command>", "  go east  "),
    ],
)
def test_parse_command_parametrized(command_input, expected_output):
    """Test command parsing with various inputs."""
    assert parse_command(command_input) == expected_output


class TestZorkDatasetIntegration:
    """Integration tests for ZorkDataset"""

    @pytest.fixture
    def mock_games_directory(self, tmp_path):
        """Create a mock games directory with fake game files"""
        games_dir = tmp_path / "games"
        games_dir.mkdir()

        # Create mock game files
        (games_dir / "zork.z5").write_bytes(b"mock game data")
        (games_dir / "zork1.z5").write_bytes(b"mock game data")

        return games_dir

    @pytest.fixture
    def mock_textworld_env(self):
        """Mock textworld environment"""
        mock_env = MagicMock()
        mock_env.reset.return_value = ("You are in a room.", {"inventory": ""})
        mock_env.step.return_value = (
            "You moved north.",
            10,  # score
            False,  # done
            {"inventory": "You have nothing."},
        )
        return mock_env

    @patch("textworld.gym.register_game")
    @patch("textworld.gym.make")
    def test_dataset_initialization_success(
        self, mock_make, mock_register, mock_games_directory
    ):
        """Test successful dataset initialization with mock games"""
        # Setup mocks
        mock_register.return_value = "mock_env_id"
        mock_make.return_value = MagicMock()

        with patch.dict(
            os.environ, {"INFORM_GAMES_DIRECTORY": str(mock_games_directory)}
        ):
            dataset = ZorkDataset(
                split="train",
                host="local",
                tokenizer=None,
                game_names=["zork", "zork1"],
            )

        assert len(dataset.env_ids) == 2
        assert "zork" in dataset.env_ids
        assert "zork1" in dataset.env_ids
        assert dataset.max_steps == 10
        assert len(dataset) == 1000

    @patch("textworld.gym.register_game")
    def test_dataset_initialization_no_games_found(self, mock_register, tmp_path):
        """Test dataset initialization when no game files are found"""
        empty_games_dir = tmp_path / "empty_games"
        empty_games_dir.mkdir()

        with patch.dict(os.environ, {"INFORM_GAMES_DIRECTORY": str(empty_games_dir)}):
            with pytest.raises(ValueError, match="No valid game files found"):
                ZorkDataset(
                    split="train",
                    host="local",
                    tokenizer=None,
                    game_names=["nonexistent"],
                )

    @patch("textworld.gym.register_game")
    @patch("textworld.gym.make")
    def test_dataset_getitem(self, mock_make, mock_register, mock_games_directory):
        """Test __getitem__ returns correct sample format"""
        mock_register.return_value = "mock_env_id"
        mock_make.return_value = MagicMock()

        with patch.dict(
            os.environ, {"INFORM_GAMES_DIRECTORY": str(mock_games_directory)}
        ):
            dataset = ZorkDataset(split="train", host="local", tokenizer=None)

        sample = dataset[0]
        assert isinstance(sample, dict)
        assert "index" in sample
        assert sample["index"] == 0

    @patch("textworld.gym.register_game")
    @patch("textworld.gym.make")
    def test_conversation_generation(
        self, mock_make, mock_register, mock_games_directory
    ):
        """Test conversation generation for new and existing trajectories"""
        mock_register.return_value = "mock_env_id"
        mock_env = MagicMock()
        mock_env.reset.return_value = ("Welcome to Zork!", {"inventory": ""})
        mock_make.return_value = mock_env

        with patch.dict(
            os.environ, {"INFORM_GAMES_DIRECTORY": str(mock_games_directory)}
        ):
            dataset = ZorkDataset(split="train", host="local", tokenizer=None)

        sample = dataset[0]

        # Test first conversation (should create new environment)
        conv1 = dataset.conversation(sample, 0)
        assert isinstance(conv1, list)
        assert len(conv1) >= 1
        assert conv1[0]["role"] == "system"
        assert "Zork" in conv1[0]["content"]

        # Verify environment was created
        assert 0 in dataset.envs
        assert 0 in dataset.agents
        assert 0 in dataset.sample_games

        # Test second conversation (should reuse environment)
        conv2 = dataset.conversation(sample, 0)
        assert isinstance(conv2, list)

import pytest
from unittest.mock import MagicMock, patch
from minrl.tasks.zork import parse_command, ZorkDataset


def test_parse_command():
    assert parse_command("<command>go north</command>") == "go north"
    assert parse_command("  <command>take lantern</command>  ") == "take lantern"
    assert parse_command("COMMAND: <command>open mailbox</command>") == "open mailbox"

    with pytest.raises(IndexError):
        parse_command("no command here")


@patch("minrl.tasks.zork.textworld.gym.register_game")
@patch("minrl.tasks.zork.textworld.gym.make")
@patch("minrl.tasks.zork.os.getenv")
@patch("minrl.tasks.zork.os.listdir")
@patch("minrl.tasks.zork.Path.exists")
def test_zork_dataset(
    mock_exists, mock_listdir, mock_getenv, mock_make, mock_register_game
):
    # Mocking the environment setup
    mock_listdir.return_value = ["zork1.z5"]
    mock_getenv.return_value = "/games"
    mock_register_game.return_value = "zork-zork1-z5"
    mock_exists.return_value = True

    # Mocking the environment interactions
    mock_env = MagicMock()
    mock_env.reset.return_value = ("Initial observation", {})
    mock_env.step.return_value = (
        "Next observation",
        1,
        False,
        {"inventory": "a sword"},
    )
    mock_make.return_value = mock_env

    dataset = ZorkDataset(split="train", host="local")

    # Test initial conversation
    initial_conv = dataset.initial_conversation({}, 0)
    assert isinstance(initial_conv, list)
    assert initial_conv[1]["role"] == "user"
    assert "Initial observation" in initial_conv[1]["content"]

    # Test get_next_state
    obs, done, step_metadata = dataset.step(
        0, [{"role": "assistant", "content": "<command>go east</command>"}]
    )
    assert "Next observation" in obs
    assert "Inventory: a sword" in obs
    assert done is False
    assert isinstance(step_metadata, dict)
    assert "observation" in step_metadata
    assert step_metadata["inventory"] == ["a sword"]

    # Test __len__
    assert len(dataset) == 43  # Number of known good games

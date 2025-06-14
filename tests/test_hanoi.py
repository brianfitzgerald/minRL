import pytest

from minrl.tasks.hanoi import TowerOfHanoi


class TestTowerOfHanoi:

    def setup_method(self):
        """Set up test fixtures."""
        self.game = TowerOfHanoi(3)

    def test_initialization(self):
        """Test game initialization."""
        assert self.game.num_disks == 3
        assert self.game.moves_count == 0
        assert self.game.stacks[0] == [3, 2, 1]
        assert self.game.stacks[1] == []
        assert self.game.stacks[2] == []

    def test_initialization_with_different_sizes(self):
        """Test initialization with different disk counts."""
        game1 = TowerOfHanoi(1)
        assert game1.stacks[0] == [1]

        game5 = TowerOfHanoi(5)
        assert game5.stacks[0] == [5, 4, 3, 2, 1]

    def test_initialization_invalid_size(self):
        """Test initialization with invalid disk count."""
        with pytest.raises(ValueError):
            TowerOfHanoi(0)

        with pytest.raises(ValueError):
            TowerOfHanoi(-1)

    def test_reset(self):
        """Test game reset functionality."""
        self.game.make_move(0, 1)
        self.game.make_move(0, 2)
        assert self.game.moves_count == 2

        self.game.reset()
        assert self.game.moves_count == 0
        assert self.game.stacks[0] == [3, 2, 1]
        assert self.game.stacks[1] == []
        assert self.game.stacks[2] == []

    def test_get_state(self):
        """Test getting game state."""
        state = self.game.get_state()
        expected = {"stacks": [[3, 2, 1], [], []], "moves_count": 0}
        assert state == expected

    def test_valid_moves(self):
        """Test valid move scenarios."""
        # Move smallest disk from stack 0 to stack 1
        is_valid, msg = self.game.is_valid_move(0, 1)
        assert is_valid
        assert msg == ""

        # Make the move and verify
        assert self.game.make_move(0, 1)
        assert self.game.stacks[0] == [3, 2]
        assert self.game.stacks[1] == [1]
        assert self.game.moves_count == 1

    def test_invalid_move_same_stack(self):
        """Test invalid move to same stack."""
        is_valid, msg = self.game.is_valid_move(0, 0)
        assert not is_valid
        assert "same stack" in msg

    def test_invalid_move_empty_stack(self):
        """Test invalid move from empty stack."""
        is_valid, msg = self.game.is_valid_move(1, 0)
        assert not is_valid
        assert "empty" in msg

    def test_invalid_move_larger_on_smaller(self):
        """Test invalid move placing larger disk on smaller."""
        # Move small disk to stack 1
        self.game.make_move(0, 1)

        # Try to move medium disk on top of small disk
        is_valid, msg = self.game.is_valid_move(0, 1)
        assert not is_valid
        assert "Cannot place disk" in msg

    def test_invalid_stack_indices(self):
        """Test invalid stack indices."""
        is_valid, msg = self.game.is_valid_move(-1, 0)
        assert not is_valid
        assert "Invalid source stack" in msg

        is_valid, msg = self.game.is_valid_move(0, 3)
        assert not is_valid
        assert "Invalid destination stack" in msg

    def test_is_solved_initial_state(self):
        """Test that initial state is not solved."""
        assert not self.game.is_solved()

    def test_is_solved_complete_game(self):
        """Test solving a 2-disk game."""
        game = TowerOfHanoi(2)

        # Optimal solution for 2 disks: 0->1, 0->2, 1->2
        moves = [(0, 1), (0, 2), (1, 2)]

        for from_stack, to_stack in moves:
            game.make_move(from_stack, to_stack)

        assert game.is_solved()
        assert game.moves_count == 3

    def test_minimum_moves_calculation(self):
        """Test minimum moves calculation."""
        assert TowerOfHanoi(1).get_minimum_moves() == 1
        assert TowerOfHanoi(2).get_minimum_moves() == 3
        assert TowerOfHanoi(3).get_minimum_moves() == 7
        assert TowerOfHanoi(4).get_minimum_moves() == 15

    def test_complete_3_disk_solution(self):
        """Test complete solution for 3-disk puzzle."""
        # Optimal solution for 3 disks (7 moves)
        moves = [
            (0, 2),  # Move disk 1 to stack 2
            (0, 1),  # Move disk 2 to stack 1
            (2, 1),  # Move disk 1 to stack 1
            (0, 2),  # Move disk 3 to stack 2
            (1, 0),  # Move disk 1 to stack 0
            (1, 2),  # Move disk 2 to stack 2
            (0, 2),  # Move disk 1 to stack 2
        ]

        for i, (from_stack, to_stack) in enumerate(moves):
            assert self.game.make_move(
                from_stack, to_stack
            ), f"Move {i+1} failed: {from_stack} -> {to_stack}"

        assert self.game.is_solved()
        assert self.game.moves_count == 7
        assert self.game.stacks[2] == [3, 2, 1]

    def test_failed_move_doesnt_increment_counter(self):
        """Test that failed moves don't increment the move counter."""
        initial_count = self.game.moves_count

        # Try invalid move
        assert not self.game.make_move(1, 0)  # Empty stack
        assert self.game.moves_count == initial_count

        # Try another invalid move
        assert not self.game.make_move(0, 0)  # Same stack
        assert self.game.moves_count == initial_count


# Parametrized tests for additional coverage
@pytest.mark.parametrize(
    "num_disks,expected_min_moves",
    [
        (1, 1),
        (2, 3),
        (3, 7),
        (4, 15),
        (5, 31),
        (6, 63),
    ],
)
def test_minimum_moves_parametrized(num_disks, expected_min_moves):
    """Test minimum moves calculation for various disk counts."""
    game = TowerOfHanoi(num_disks)
    assert game.get_minimum_moves() == expected_min_moves


@pytest.mark.parametrize(
    "from_stack,to_stack,should_be_valid",
    [
        (0, 1, True),  # Valid: top disk to empty stack
        (0, 2, True),  # Valid: top disk to empty stack
        (0, 0, False),  # Invalid: same stack
        (1, 0, False),  # Invalid: empty stack
        (3, 1, False),  # Invalid: out of range
        (-1, 1, False),  # Invalid: negative index
    ],
)
def test_move_validation_parametrized(from_stack, to_stack, should_be_valid):
    """Test move validation with various inputs."""
    game = TowerOfHanoi(3)
    is_valid, _ = game.is_valid_move(from_stack, to_stack)
    assert is_valid == should_be_valid


@pytest.fixture
def solved_2_disk_game():
    """Fixture providing a solved 2-disk game."""
    game = TowerOfHanoi(2)
    moves = [(0, 1), (0, 2), (1, 2)]
    for from_stack, to_stack in moves:
        game.make_move(from_stack, to_stack)
    return game


def test_solved_game_state(solved_2_disk_game):
    """Test that solved game has correct final state."""
    assert solved_2_disk_game.is_solved()
    assert solved_2_disk_game.stacks[0] == []
    assert solved_2_disk_game.stacks[1] == []
    assert solved_2_disk_game.stacks[2] == [2, 1]
    assert solved_2_disk_game.moves_count == 3

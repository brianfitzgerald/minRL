from typing import List, Dict, Tuple, Union


class TowerOfHanoi:
    def __init__(self, num_disks: int = 3) -> None:
        if num_disks < 1:
            raise ValueError("Number of disks must be at least 1")

        self.num_disks: int = num_disks
        self.moves_count: int = 0
        self.stacks: List[List[int]] = []
        self.reset()

    def reset(self) -> None:
        self.stacks = [
            list(range(self.num_disks, 0, -1)),
            [],
            [],
        ]
        self.moves_count = 0

    def get_state(self) -> Dict[str, Union[List[List[int]], int]]:
        return {
            "stacks": [stack.copy() for stack in self.stacks],
            "moves_count": self.moves_count,
        }

    def display(self) -> None:
        pass

    def is_valid_move(self, from_stack: int, to_stack: int) -> Tuple[bool, str]:
        if from_stack not in [0, 1, 2]:
            return False, f"Invalid source stack: {from_stack}. Must be 0, 1, or 2."

        if to_stack not in [0, 1, 2]:
            return False, f"Invalid destination stack: {to_stack}. Must be 0, 1, or 2."

        if from_stack == to_stack:
            return False, "Cannot move disk to the same stack."

        if not self.stacks[from_stack]:
            return False, f"Stack {from_stack} is empty."

        if self.stacks[to_stack]:
            top_from = self.stacks[from_stack][-1]
            top_to = self.stacks[to_stack][-1]
            if top_from > top_to:
                return False, f"Cannot place disk {top_from} on smaller disk {top_to}."

        return True, ""

    def make_move(self, from_stack: int, to_stack: int) -> bool:
        is_valid, error_msg = self.is_valid_move(from_stack, to_stack)

        if not is_valid:
            return False

        disk = self.stacks[from_stack].pop()
        self.stacks[to_stack].append(disk)
        self.moves_count += 1

        return True

    def is_solved(self) -> bool:
        return len(self.stacks[2]) == self.num_disks and self.stacks[2] == list(
            range(self.num_disks, 0, -1)
        )

    def get_minimum_moves(self) -> int:
        return (2**self.num_disks) - 1

    def play(self) -> None:
        while True:
            if self.is_solved():
                break

            try:
                user_input = input().strip().lower()

                if user_input == "quit":
                    break
                elif user_input == "reset":
                    self.reset()
                    continue

                parts = user_input.split()
                if len(parts) != 2:
                    continue

                from_stack = int(parts[0])
                to_stack = int(parts[1])

                self.make_move(from_stack, to_stack)

            except ValueError:
                pass
            except KeyboardInterrupt:
                break

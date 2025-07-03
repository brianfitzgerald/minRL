from minrl.constants import TaskChoice
from minrl.tasks.connections import (
    ConnectionsDataset,
)
from typing import Protocol, Any

from minrl.tasks.dataset import MinRLDataset
from minrl.tasks.hanoi import HanoiDataset
from minrl.tasks.zork import ZorkDataset


class RewardFunction(Protocol):
    def __call__(self, response: str, sample: dict[str, Any]) -> float: ...


TASK_DEFINITIONS: dict[TaskChoice, type[MinRLDataset]] = {
    "connections": ConnectionsDataset,
    "hanoi": HanoiDataset,
    "zork": ZorkDataset,
}

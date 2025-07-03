from minrl.constants import TaskChoice
from minrl.tasks.connections import (
    ConnectionsDataset,
    connections_reward_func,
)
from typing import TypedDict, Protocol, Any

from minrl.tasks.dataset import MinRLDataset
from minrl.tasks.hanoi import HanoiDataset, hanoi_reward_func
from minrl.tasks.zork import ZorkDataset, zork_reward_func


class RewardFunction(Protocol):
    def __call__(self, response: str, sample: dict[str, Any]) -> float: ...


class TaskDefinition(TypedDict):
    reward_function: RewardFunction
    dataset: type[MinRLDataset]


TASK_DEFINITIONS: dict[TaskChoice, TaskDefinition] = {
    "connections": {
        "reward_function": connections_reward_func,
        "dataset": ConnectionsDataset,
    },
    "hanoi": {
        "reward_function": hanoi_reward_func,
        "dataset": HanoiDataset,
    },
    "zork": {
        "reward_function": zork_reward_func,
        "dataset": ZorkDataset,
    },
}

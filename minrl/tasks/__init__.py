from minrl.tasks.connections import (
    ConnectionsDataset,
    connections_reward_func,
)
from typing import Literal, TypedDict, Protocol, Any

from minrl.tasks.dataset import MinRLDataset
from minrl.tasks.hanoi import HanoiDataset, hanoi_reward_func, tokenize_hanoi_sample

TaskChoice = Literal["connections", "hanoi"]


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
}

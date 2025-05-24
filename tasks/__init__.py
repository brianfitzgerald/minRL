from tasks.connections import (
    ConnectionsDataset,
    connections_reward_func,
    tokenize_connections_sample,
)
from typing import Callable, Literal, TypedDict, Protocol, Any

from tasks.dataset import MinRLDataset

TaskChoice = Literal["connections", "countdown"]


class RewardFunction(Protocol):
    def __call__(self, response: str, sample: dict[str, Any]) -> float: ...


class TaskDefinition(TypedDict):
    name: TaskChoice
    reward_function: RewardFunction
    dataset: type[MinRLDataset]
    postprocess_function: Callable


TASK_DEFINITIONS: dict[TaskChoice, TaskDefinition] = {
    "connections": {
        "name": "connections",
        "reward_function": connections_reward_func,
        "dataset": ConnectionsDataset,
        "postprocess_function": tokenize_connections_sample,
    },
}

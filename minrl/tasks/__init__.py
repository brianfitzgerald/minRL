from minrl.tasks.connections import (
    ConnectionsDataset,
    connections_reward_func,
    tokenize_connections_sample,
)
from typing import Callable, Literal, TypedDict, Protocol, Any

from minrl.tasks.dataset import MinRLDataset
from minrl.tasks.hanoi import HanoiDataset, hanoi_reward_func, tokenize_hanoi_sample

TaskChoice = Literal["connections", "hanoi"]


class RewardFunction(Protocol):
    def __call__(self, response: str, sample: dict[str, Any]) -> float: ...


class TaskDefinition(TypedDict):
    reward_function: RewardFunction
    dataset: type[MinRLDataset]
    postprocess_function: Callable


TASK_DEFINITIONS: dict[TaskChoice, TaskDefinition] = {
    "connections": {
        "reward_function": connections_reward_func,
        "dataset": ConnectionsDataset,
        "postprocess_function": tokenize_connections_sample,
    },
    "hanoi": {
        "reward_function": hanoi_reward_func,
        "dataset": HanoiDataset,
        "postprocess_function": tokenize_hanoi_sample,
    },
}

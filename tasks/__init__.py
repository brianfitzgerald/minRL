from minrl.grpo import RewardFunction
from tasks.connections import ConnectionsDataset, connections_reward_func
from tasks.countdown import CountdownTasksDataset, countdown_reward_function
from typing import Literal, TypedDict
from torch.utils.data import Dataset

TaskChoice = Literal["connections", "countdown"]

class TaskDefinition(TypedDict):
    name: TaskChoice
    reward_function: RewardFunction
    dataset: type[Dataset]

TASK_DEFINITIONS: dict[TaskChoice, TaskDefinition] = {
    "connections": {
        "name": "connections",
        "reward_function": connections_reward_func,
        "dataset": ConnectionsDataset,
    },
    "countdown": {
        "name": "countdown",
        "reward_function": countdown_reward_function,
        "dataset": CountdownTasksDataset,
    },
}
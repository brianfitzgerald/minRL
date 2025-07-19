from typing import TypedDict
from minrl.constants import TaskChoice, RewardFunction
from minrl.tasks.connections import ConnectionsDataset
from minrl.tasks.dataset import MinRLDataset
from minrl.tasks.connections import connections_reward_func
from minrl.tasks.hanoi import HanoiDataset, hanoi_reward_func
from minrl.tasks.zork import ZorkDataset, zork_reward_func


class TaskDefinition(TypedDict):
    dataset: type[MinRLDataset]
    reward_function: RewardFunction
    n_max_turns: int


TASK_DATASETS: dict[TaskChoice, TaskDefinition] = {
    "connections": {
        "dataset": ConnectionsDataset,
        "reward_function": connections_reward_func,
        "n_max_turns": 1,
    },
    "hanoi": {
        "dataset": HanoiDataset,
        "reward_function": hanoi_reward_func,
        "n_max_turns": 1,
    },
    "zork": {
        "dataset": ZorkDataset,
        "reward_function": zork_reward_func,
        "n_max_turns": 1000,
    },
}

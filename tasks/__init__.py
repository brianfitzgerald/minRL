from minrl.grpo import RewardFunction
from tasks.connections import connections_reward_func
from tasks.countdown import countdown_reward_function
from typing import Literal, TypedDict

TaskChoice = Literal["connections", "countdown"]

class TaskDefinition(TypedDict):
    name: TaskChoice
    reward_function: RewardFunction

TASK_DEFINITIONS: dict[TaskChoice, TaskDefinition] = {
    "connections": {
        "name": "connections",
        "reward_function": connections_reward_func,
    },
    "countdown": {
        "name": "countdown",
        "reward_function": countdown_reward_function,
    },
}
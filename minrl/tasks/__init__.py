from minrl.constants import TaskChoice
from minrl.tasks.connections import ConnectionsDataset
from minrl.tasks.dataset import MinRLDataset
from minrl.tasks.hanoi import HanoiDataset

try:
    from minrl.tasks.zork import ZorkDataset, zork_reward_func
except ImportError:
    ZorkDataset = None
    zork_reward_func = None

TASK_DATASETS: dict[TaskChoice, type[MinRLDataset]] = {
    "connections": ConnectionsDataset,
    "hanoi": HanoiDataset,
}

# Add zork task only if the group is installed
if ZorkDataset is not None and zork_reward_func is not None:
    TASK_DATASETS["zork"] = ZorkDataset

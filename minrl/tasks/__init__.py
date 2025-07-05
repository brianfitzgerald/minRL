from minrl.constants import TaskChoice
from minrl.tasks.connections import ConnectionsDataset
from minrl.tasks.dataset import MinRLDataset
from minrl.tasks.hanoi import HanoiDataset
from minrl.tasks.zork import ZorkDataset


TASK_DATASETS: dict[TaskChoice, type[MinRLDataset]] = {
    "connections": ConnectionsDataset,
    "hanoi": HanoiDataset,
    "zork": ZorkDataset,
}

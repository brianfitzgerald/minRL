import fire
from dotenv import load_dotenv
from torch.utils.data import DataLoader
from tasks import TASK_DEFINITIONS, TaskChoice

"""
Evaluate a series of OSS models against prompts and evals for a specific task.
"""

load_dotenv(".env")

INFERENCE_MODELS = ["gpt-4.1", "gpt-4.1-mini", "o4-mini"]


def main(task: TaskChoice = "connections"):
    task_definition = TASK_DEFINITIONS[task]
    dataset, reward_function = task_definition["dataset"](), task_definition["reward_function"]
    loader = DataLoader(dataset, batch_size=1, shuffle=False)

    for batch in loader:
        print(batch)
        break


if __name__ == "__main__":
    fire.Fire(main)

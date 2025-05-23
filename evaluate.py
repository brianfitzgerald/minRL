import fire
from dotenv import load_dotenv
from torch.utils.data import DataLoader
from tasks import TASK_DEFINITIONS, TaskChoice
import litellm
from litellm.types.utils import ModelResponse
from typing import TypedDict

"""
Evaluate a series of OSS models against prompts and evals for a specific task.
"""

load_dotenv(".env")

INFERENCE_MODELS = ["gpt-4.1", "gpt-4.1-mini", "o4-mini"]


class OutRow(TypedDict):
    model: str
    prompt: str
    response: str
    score: float


def main(task: TaskChoice = "connections"):
    task_definition = TASK_DEFINITIONS[task]
    dataset, reward_function = (
        task_definition["dataset"]("eval"),
        task_definition["reward_function"],
    )
    loader = DataLoader(dataset, batch_size=1, shuffle=False)

    out_rows: list[OutRow] = []

    for sample in loader:
        for model in INFERENCE_MODELS:
            response: ModelResponse = litellm.completion(
                model=model,
                messages=[{"role": "user", "content": sample["prompt"]}],
            )  # type: ignore
            print(model, response)
            response_content: str = response.choices[0].message.content  # type: ignore
            score = reward_function(response_content, sample)
            out_rows.append(
                {
                    "model": model,
                    "prompt": sample["prompt"],
                    "response": response_content,
                    "score": score,
                }
            )

    print(out_rows)


if __name__ == "__main__":
    fire.Fire(main)

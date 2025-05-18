import json
import fire
import litellm
from dotenv import load_dotenv
from typing import TypedDict
import pandas as pd

from tasks.connections import ConnectionsDataset

load_dotenv()


def main():
    prompts = json.load(open("data/eval_prompts.json"))
    prompts_df = pd.DataFrame(prompts)
    dataset = ConnectionsDataset(prompts_df)
    print(prompts_df)


if __name__ == "__main__":
    fire.Fire(main)

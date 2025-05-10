import json
import fire


def main(model_id: str = "Qwen/Qwen3-0.6B"):
    prompts = json.load(open("data/eval_prompts.json"))

if __name__ == "__main__":
    fire.Fire(main)

import json
import fire


def main():
    prompts = json.load(open("data/eval_prompts.json"))
    print(prompts)

if __name__ == "__main__":
    fire.Fire(main)

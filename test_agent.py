import os
from openai import OpenAI
import textworld
import textworld.gym
from textworld.gym.envs import TextworldGymEnv
from minrl.constants import EVAL_MODELS, QWEN_3_0_6B
from minrl.tasks.zork import TextWorldAgent
from loguru import logger
from minrl.tasks.zork import ZorkDataset
from torch.utils.data import DataLoader
from transformers.models.auto.tokenization_auto import AutoTokenizer


def test_agent():
    infos = textworld.EnvInfos(
        feedback=True,
        description=True,
        inventory=True,
        verbs=True,
        intermediate_reward=True,
        entities=True,
        facts=True,
    )

    openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    model_name = EVAL_MODELS["gpt_4.1_mini"]["model_id"]

    env_id = textworld.gym.register_game("games/zork.z5", infos)
    env: TextworldGymEnv = textworld.gym.make(env_id)
    obs = env.reset()

    agent = TextWorldAgent()

    env.render()
    total_score, moves, done, infos = 0, 0, False, {}
    while not done:
        conv = agent.conversation(obs.raw)
        response = openai_client.chat.completions.create(
            model=model_name,
            messages=conv,
        )
        response_str = response.choices[0].message.content
        assert response_str is not None, "Response is None"
        inventory = infos["inventory"] if "inventory" in infos else ""
        agent.update(obs, inventory, response_str)
        response_str = response_str.replace("COMMAND: ", "").strip()
        logger.info(f"COMMAND: {response_str}")
        obs, score, done, infos = env.step(response_str)  # type: ignore
        logger.info(env.render(mode="text"))
        total_score += score  # type: ignore
        moves += 1
        print(f"Score: {total_score}, Moves: {moves}")
    env.close()


def test_dataset():
    tokenizer = AutoTokenizer.from_pretrained(QWEN_3_0_6B)
    dataset = ZorkDataset(split="train", host="local", tokenizer=tokenizer)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True)
    for i, batch in enumerate(dataloader):
        print(batch)
        if i > 10:
            break


if __name__ == "__main__":
    test_dataset()

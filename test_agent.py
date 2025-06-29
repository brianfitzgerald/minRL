import os
from tqdm import tqdm
from openai import OpenAI
import textworld
import textworld.gym
from textworld.gym.envs import TextworldGymEnv
from minrl.constants import EVAL_MODELS, QWEN_3_0_6B
from minrl.tasks.dataset import Episode
from minrl.tasks.zork import TextWorldAgent
from loguru import logger
from minrl.tasks.zork import ZorkDataset
from torch.utils.data import DataLoader
from transformers.models.auto.tokenization_auto import AutoTokenizer
from dotenv import load_dotenv

load_dotenv()

model_name = EVAL_MODELS["gpt_4.1_mini"]["model_id"]


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

    env_id = textworld.gym.register_game("games/zork1.z5", infos)
    env: TextworldGymEnv = textworld.gym.make(env_id)
    obs = env.reset()

    agent = TextWorldAgent()

    env.render()
    total_score, moves, done, infos = 0, 0, False, {}
    while not done:
        conv = agent.conversation(obs["raw"])  # type: ignore
        response = openai_client.chat.completions.create(
            model=model_name,
            messages=conv,
        )
        response_str = response.choices[0].message.content
        assert response_str is not None, "Response is None"
        inventory = infos["inventory"] if "inventory" in infos else ""
        agent.update(response_str, infos["description"], obs, inventory)  # type: ignore
        response_str = response_str.replace("COMMAND: ", "").strip()
        logger.info(f"COMMAND: {response_str}")
        obs, score, done, infos = env.step(response_str)  # type: ignore
        logger.info(env.render(mode="text"))
        total_score += score  # type: ignore
        moves += 1
        print(f"Score: {total_score}, Moves: {moves}")
    env.close()


def test_dataset():
    openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    tokenizer = AutoTokenizer.from_pretrained(QWEN_3_0_6B)
    dataset = ZorkDataset(split="train", host="local", tokenizer=tokenizer)
    dataloader = DataLoader(dataset, batch_size=1, collate_fn=lambda x: x)
    for i, batch in enumerate(dataloader):
        convs = [x["conversation"] for x in batch]
        for i, conv in enumerate(tqdm(convs, desc="Inference")):
            print(conv[1]["content"])
            completion = openai_client.chat.completions.create(
                model=model_name,
                messages=conv,
            )
            completion_content = completion.choices[0].message.content
            assert completion_content is not None, "Response is None"
            episode = Episode(
                batch_index=i,
                answer_index=i,
                prefix=conv[0]["content"],
                prefix_token_ids=[],
                generated_token_ids=[],
                text=completion_content,
                reward=0,
                is_finished=False,
                reward_info={},
            )
            dataset.post_generate(episode)
        if i > 10:
            break


if __name__ == "__main__":
    test_dataset()

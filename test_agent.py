import os
from openai import OpenAI
import textworld
import textworld.gym
from textworld.gym.envs import TextworldGymEnv
from minrl.constants import EVAL_MODELS
from minrl.tasks.zork import TextWorldAgent

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
obs, infos = env.reset()

agent = TextWorldAgent()

env.render()
total_score, moves, done, infos = 0, 0, False, {}
while not done:
    conv = agent.conversation(obs)
    response = openai_client.chat.completions.create(
        model=model_name,
        messages=conv,
    )
    response_str = response.choices[0].message.content
    assert response_str is not None, "Response is None"
    inventory = infos["inventory"] if "inventory" in infos else []
    agent.update(obs, inventory, response_str)
    obs, score, done, infos = env.step(response_str)  # type: ignore
    env.render()
    total_score += score  # type: ignore
    moves += 1
    print(f"Score: {total_score}, Moves: {moves}")
env.close()

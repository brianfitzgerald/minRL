import textworld
import textworld.gym
from textworld.gym.envs import TextworldGymEnv
from minrl.constants import INFERENCE_MODELS
from minrl.tasks.zork import TextWorldAgent
from loguru import logger
from dotenv import load_dotenv
import fire
import os
from pathlib import Path

load_dotenv()

model_name = INFERENCE_MODELS["gemini_2.5_flash"]["model_id"]


async def test_agent(game: str = "wurm.z5"):
    infos = textworld.EnvInfos(
        feedback=True,
        description=True,
        inventory=True,
        verbs=True,
        intermediate_reward=True,
        entities=True,
        facts=True,
    )

    games_dir = Path(os.path.expanduser("~/Documents/GitHub/DeepZork/game_files"))
    games_path = games_dir / game
    logger.info(games_path)
    env_id = textworld.gym.register_game(games_path.as_posix(), infos)
    env: TextworldGymEnv = textworld.gym.make(env_id)
    obs = env.reset()

    agent = TextWorldAgent()

    env.render()
    total_score, moves, done, infos, obs = 0, 0, False, {}, ""
    score: int = 0
    while not done:
        response_str = input()
        assert response_str is not None, "Response is None"
        inventory = infos["inventory"] if "inventory" in infos else ""
        agent.update(response_str, obs, inventory, done, score)
        response_str = response_str.replace("COMMAND: ", "").strip()
        obs, score, done, infos = env.step(response_str)  # type: ignore
        logger.info(env.render(mode="text"))
        total_score += score  # type: ignore
        moves += 1
        print(f"Score: {total_score}, Moves: {moves}")
    env.close()


if __name__ == "__main__":
    fire.Fire(test_agent)

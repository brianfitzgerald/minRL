import textworld
import textworld.gym
from textworld.gym.envs import TextworldGymEnv

env_id = textworld.gym.register_game("games/zork.z5")
env: TextworldGymEnv = textworld.gym.make(env_id)
obs, infos = env.reset()
env.render()
total_score, moves, done, infos = 0, 0, False, {}
while not done:
    command = input("> ")
    obs, score, done, infos = env.step(command)  # type: ignore
    env.render()
    total_score += score  # type: ignore
    moves += 1
    print(f"Score: {total_score}, Moves: {moves}")
env.close()

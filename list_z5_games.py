#!/usr/bin/env python3
"""
Script to iterate through all Z5 games and print available information.
"""

import os
import textworld
import textworld.gym
from pathlib import Path
from dotenv import load_dotenv


def main():
    load_dotenv()
    # Get games directory from environment variable
    games_directory = Path(os.getenv("INFORM_GAMES_DIRECTORY", ""))

    if not games_directory.exists():
        print(f"Games directory not found: {games_directory}")
        print("Please set the INFORM_GAMES_DIRECTORY environment variable")
        return

    # Find all Z5 game files
    game_files = [f for f in os.listdir(games_directory) if f.endswith(".z5")]

    if not game_files:
        print(f"No Z5 games found in {games_directory}")
        return

    print(f"Found {len(game_files)} Z5 games in {games_directory}")
    print("=" * 50)

    # Define what information we want to extract
    infos = textworld.EnvInfos(
        feedback=True,
        description=True,
        inventory=True,
        location=True,
        intermediate_reward=True,
        entities=True,
        moves=True,
        score=True,
        max_score=True,
        facts=True,
    )
    good = set()

    for game_file in sorted(game_files):
        game_path = games_directory / game_file
        print(f"Processing {game_file}")
        try:
            # Register and create the game environment
            # segfaults
            if any(
                [k in game_file for k in ("SameGame.z5", "ZTornado.z5", "coloromc.z5")]
            ):
                continue
            env_id = textworld.gym.register_game(game_path.as_posix(), infos)
            env = textworld.gym.make(env_id)

            # Reset the environment to get initial state
            obs, info = env.reset()

            # Print all available information
            valid_keys = [
                k
                for k in info.keys()
                if k not in ("feedback", "description", "inventory")
                and info[k] is not None
            ]
            print(f"{game_file}: {valid_keys}")
            if any([k in valid_keys for k in ("score", "moves", "max_score")]):
                print(f"Adding {game_file} to good games, keys: {valid_keys}")
                good.add(game_file)

            # Try a simple action to see what happens
            try:
                obs, score, done, _ = env.step("look")
            except Exception as e:
                print(f"  Error executing 'look' command: {e}")

            env.close()

        except Exception as e:
            print(f"Error loading game {game_file}: {e}")

    print(f"Good games: {good}")


if __name__ == "__main__":
    main()

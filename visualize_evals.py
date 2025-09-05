import os
import fire
import pandas as pd
from pathlib import Path
from loguru import logger
import matplotlib.pyplot as plt
from minrl.constants import EvalSample, TaskChoice
import json
import numpy as np

from minrl.utils import clean_observation


def eval_single_step(task: TaskChoice = "connections"):
    """Function to evaluate single step inference on the connections task."""
    eval_results_path = Path(f"eval_results/{task}")

    if not eval_results_path.exists():
        logger.error(f"Error: {eval_results_path} folder not found")
        return

    parquet_files = list(eval_results_path.glob("*.parquet"))

    if not parquet_files:
        logger.error(f"No parquet files found in {eval_results_path}")
        return

    all_data = []
    for file_path in parquet_files:
        logger.info(f"Loading {file_path}")
        df = pd.read_parquet(file_path)
        all_data.append(df)

    combined_df = pd.concat(all_data, ignore_index=True)

    avg_scores = combined_df.groupby("model")["score"].agg(["mean", "count"]).round(4)
    avg_scores.columns = ["average_score", "sample_count"]

    logger.info("\nAverage scores per model:")
    logger.info("=" * 50)
    logger.info(avg_scores)

    logger.info("\nOverall statistics:")
    logger.info(f"Total samples: {len(combined_df)}")
    logger.info(f"Number of models: {len(avg_scores)}")
    logger.info(f"Overall average score: {combined_df['score'].mean():.4f}")

    plt.figure(figsize=(12, 6))
    avg_scores["average_score"].plot(kind="bar", color="skyblue", edgecolor="black")
    plt.title("Average Scores by Model", fontsize=14, fontweight="bold")
    plt.xlabel("Model", fontsize=12)
    plt.ylabel("Average Score", fontsize=12)
    plt.xticks(rotation=45, ha="right")
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()

    for i, v in enumerate(avg_scores["average_score"]):
        plt.text(i, v + 0.01, f"{v:.3f}", ha="center", va="bottom", fontweight="bold")

    os.makedirs(f"eval_results/{task}/figures", exist_ok=True)
    plt.savefig(f"eval_results/{task}/figures/average_scores.png")


def render_zork_trajectories():
    """Function to convert zork trajectories from parquet to JavaScript."""
    eval_results_path = Path(
        "eval_results/zork/eval_gemini_2.5_flash_20250905_113525.parquet"
    )
    if not eval_results_path.exists():
        logger.error(f"Error: {eval_results_path} folder not found")
        return
    df = pd.read_parquet(eval_results_path)
    out_rows: list[EvalSample] = []
    for index, row in df.iterrows():
        out_row: EvalSample = row.to_dict()  # type: ignore
        out_rows.append(out_row)

    output_dir = Path("eval_results/zork")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Only write trajectory data as JavaScript file
    _write_trajectory_data_js(out_rows, output_dir)

    logger.info(f"Trajectory data converted to JavaScript in {output_dir}")


def _write_trajectory_data_js(out_rows: list[EvalSample], output_dir: Path):
    """Write trajectory data as a separate JavaScript file."""

    # Convert trajectories to JavaScript data, properly handling numpy arrays
    def convert_numpy_arrays(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {k: convert_numpy_arrays(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_numpy_arrays(item) for item in obj]
        else:
            return obj

    # Convert each trajectory, handling numpy arrays properly
    for row in out_rows:
        for message in row["conversation"]:
            if message["role"] == "user":
                message["content"] = clean_observation(message["content"])

    converted_trajectories = [convert_numpy_arrays(dict(row)) for row in out_rows]
    trajectories_json = json.dumps(converted_trajectories, indent=2)

    # Write the data as a JS file
    js_data_path = output_dir / "trajectory_data.js"
    with open(js_data_path, "w", encoding="utf-8") as f:
        f.write("// Trajectory data loaded from parquet file\n")
        f.write(f"trajectoryData = {trajectories_json};\n")
        f.write("\n// Initialize the trajectory viewer with the data\n")
        f.write("if (typeof initializeTrajectories === 'function') {\n")
        f.write("    initializeTrajectories();\n")
        f.write("}\n")

    logger.info(f"Trajectory data saved to {js_data_path}")


def main(task: TaskChoice = "connections"):
    if task == "connections":
        eval_single_step(task)
    elif task == "zork":
        render_zork_trajectories()


if __name__ == "__main__":
    fire.Fire(main)

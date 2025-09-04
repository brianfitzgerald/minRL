import os
import fire
import pandas as pd
from pathlib import Path
from loguru import logger
import matplotlib.pyplot as plt
from minrl.constants import EvalsOutRow, TaskChoice
import json
import numpy as np
import shutil

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
    """Function to render trajectories of the zork task."""
    eval_results_path = Path("eval_results/zork/eval_gemini_2.5_flash.parquet")
    if not eval_results_path.exists():
        logger.error(f"Error: {eval_results_path} folder not found")
        return
    df = pd.read_parquet(eval_results_path)
    out_rows: list[EvalsOutRow] = []
    for index, row in df.iterrows():
        out_row: EvalsOutRow = row.to_dict()  # type: ignore
        out_rows.append(out_row)

    html_content = _generate_trajectory_html(out_rows)

    output_path = Path("eval_results/zork/trajectories.html")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Copy viewer files to the output directory
    viewer_dir = output_path.parent / "viewer"
    viewer_dir.mkdir(exist_ok=True)

    # Copy CSS file
    css_source = Path("viewer/trajectory.css")
    css_dest = viewer_dir / "trajectory.css"
    if css_source.exists():
        shutil.copy2(css_source, css_dest)
        logger.info(f"Copied CSS file to {css_dest}")

    # Copy JS file
    js_source = Path("viewer/trajectory.js")
    js_dest = viewer_dir / "trajectory.js"
    if js_source.exists():
        shutil.copy2(js_source, js_dest)
        logger.info(f"Copied JavaScript file to {js_dest}")

    # Write trajectory data as separate JS file
    _write_trajectory_data_js(out_rows, viewer_dir)

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html_content)

    logger.info(f"HTML trajectory visualization saved to {output_path}")


def _write_trajectory_data_js(out_rows: list[EvalsOutRow], viewer_dir: Path):
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
    js_data_path = viewer_dir / "trajectory_data.js"
    with open(js_data_path, "w", encoding="utf-8") as f:
        f.write("// Trajectory data loaded from parquet file\n")
        f.write(f"trajectoryData = {trajectories_json};\n")
        f.write("\n// Initialize the trajectory viewer with the data\n")
        f.write("if (typeof initializeTrajectories === 'function') {\n")
        f.write("    initializeTrajectories();\n")
        f.write("}\n")

    logger.info(f"Trajectory data saved to {js_data_path}")


def _generate_trajectory_html(out_rows: list[EvalsOutRow]) -> str:
    """Generate HTML content for trajectory visualization."""

    # Calculate statistics
    if not out_rows:
        stats = {
            "total_trajectories": 0,
            "done_count": 0,
            "error_count": 0,
            "avg_steps": 0,
        }
        trajectories = []
    else:
        done_count = sum(1 for row in out_rows if row["status"] == "done")
        error_count = sum(1 for row in out_rows if row["status"] == "error")
        total_steps = sum(len(row["conversation"]) for row in out_rows)
        avg_steps = total_steps / len(out_rows) if out_rows else 0

        stats = {
            "total_trajectories": len(out_rows),
            "done_count": done_count,
            "error_count": error_count,
            "avg_steps": avg_steps,
        }

        # Store raw data for JavaScript rendering instead of pre-building all trajectories
        trajectories = out_rows

    return _render_html_template(stats, trajectories)


def _render_html_template(stats: dict, _: list[EvalsOutRow]) -> str:
    """Render the complete HTML template."""

    # Load external CSS and JS files
    css_path = Path("viewer/trajectory.css")
    js_path = Path("viewer/trajectory.js")
    template_path = Path("viewer/trajectory_template.html")

    if not css_path.exists():
        logger.error(f"CSS file not found: {css_path}")
        return "<html><body><h1>Error: CSS file not found</h1></body></html>"

    if not js_path.exists():
        logger.error(f"JavaScript file not found: {js_path}")
        return "<html><body><h1>Error: JavaScript file not found</h1></body></html>"

    if not template_path.exists():
        logger.error(f"HTML template file not found: {template_path}")
        return "<html><body><h1>Error: HTML template file not found</h1></body></html>"

    # Read the template file
    with open(template_path, "r", encoding="utf-8") as f:
        html_template = f.read()

    # Update stats in the template
    html_content = html_template.replace(
        '<strong id="total-trajectories">0</strong>',
        f'<strong id="total-trajectories">{stats["total_trajectories"]}</strong>',
    )
    html_content = html_content.replace(
        '<strong id="done-count">0</strong>',
        f'<strong id="done-count">{stats["done_count"]}</strong>',
    )
    html_content = html_content.replace(
        '<strong id="error-count">0</strong>',
        f'<strong id="error-count">{stats["error_count"]}</strong>',
    )
    html_content = html_content.replace(
        '<strong id="avg-steps">0.0</strong>',
        f'<strong id="avg-steps">{stats["avg_steps"]:.1f}</strong>',
    )

    # Replace the inline script with a reference to the external data file
    script_content = """
        // Trajectory data will be loaded from trajectory_data.js
    """
    html_content = html_content.replace(
        "// This script will be replaced with actual data by Python\n        // The trajectoryData will be set here",
        script_content,
    )

    return html_content


def main(task: TaskChoice = "connections"):
    if task == "connections":
        eval_single_step(task)
    elif task == "zork":
        render_zork_trajectories()


if __name__ == "__main__":
    fire.Fire(main)

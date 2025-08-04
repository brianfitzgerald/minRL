import os
from typing import Any
import fire
import pandas as pd
from pathlib import Path
from loguru import logger
import matplotlib.pyplot as plt
from minrl.constants import EvalsOutRow, TaskChoice


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
    eval_results_path = Path("eval_results/zork/eval_gpt-4.1-mini.parquet")
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

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html_content)

    logger.info(f"HTML trajectory visualization saved to {output_path}")


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

        trajectories = [
            _build_trajectory_data(i, row) for i, row in enumerate(out_rows)
        ]

    return _render_html_template(stats, trajectories)


def _build_trajectory_data(index: int, row: EvalsOutRow) -> dict:
    """Build structured data for a single trajectory."""
    conversation = row["conversation"]

    steps = []
    for i, message in enumerate(conversation):
        step_data: dict[str, Any] = {
            "step_num": i + 1,
            "message": {
                "role": message["role"],
                "content": message["content"],
            },
        }

        # Add reasoning if present
        if "reasoning" in message and message["reasoning"]:
            step_data["message"]["reasoning"] = message["reasoning"]

        steps.append(step_data)

    return {
        "index": index + 1,
        "model": row["model"],
        "status": row["status"],
        "steps": steps,
    }


def _render_html_template(stats: dict, trajectories: list[dict]) -> str:
    """Render the complete HTML template."""

    css = """
        body { font-family: sans-serif; max-width: 1200px; margin: 0 auto; padding: 20px; }
        .trajectory { border: 1px solid #ddd; margin-bottom: 20px; }
        .trajectory-header { 
            background: #f5f5f5; 
            padding: 10px; 
            font-weight: bold; 
            cursor: pointer;
            user-select: none;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }
        .trajectory-header:hover { background: #e8e8e8; }
        .toggle-button {
            font-size: 14px;
            color: #666;
            font-weight: normal;
        }
        .trajectory-content {
            padding: 15px;
            display: block;
        }
        .trajectory-content.collapsed {
            display: none;
        }
        .status-done { color: green; }
        .status-error { color: red; }
        .status-running { color: orange; }
        .step { border-left: 3px solid #eee; padding-left: 15px; margin: 15px 0; }
        .action { background: #e3f2fd; padding: 8px; margin: 5px 0; font-family: monospace; }
        .observation { background: #f3e5f5; padding: 8px; margin: 5px 0; white-space: pre-wrap; }
        .response { background: #fff3e0; padding: 8px; margin: 5px 0; }
        .stats { display: flex; gap: 20px; margin: 20px 0; }
        .stat { text-align: center; }
    """

    header = "<h1>Zork Trajectories</h1>"

    stats_html = f"""
    <div class="stats">
        <div class="stat"><strong>{stats["total_trajectories"]}</strong><br>Total</div>
        <div class="stat"><strong>{stats["done_count"]}</strong><br>Done</div>
        <div class="stat"><strong>{stats["error_count"]}</strong><br>Errors</div>
        <div class="stat"><strong>{stats["avg_steps"]:.1f}</strong><br>Avg Steps</div>
    </div>
    """

    trajectories_html = ""
    if not trajectories:
        trajectories_html = "<p>No trajectories found.</p>"
    else:
        for traj in trajectories:
            trajectories_html += _render_trajectory(traj)

    javascript = """
        function toggleTrajectory(trajectoryId) {
            const content = document.getElementById('content-' + trajectoryId);
            const button = document.getElementById('button-' + trajectoryId);
            
            if (content.classList.contains('collapsed')) {
                content.classList.remove('collapsed');
                button.textContent = '▼ Collapse';
            } else {
                content.classList.add('collapsed');
                button.textContent = '▶ Expand';
            }
        }
        
        function collapseAll() {
            const contents = document.querySelectorAll('.trajectory-content');
            const buttons = document.querySelectorAll('.toggle-button');
            
            contents.forEach(content => content.classList.add('collapsed'));
            buttons.forEach(button => button.textContent = '▶ Expand');
        }
        
        function expandAll() {
            const contents = document.querySelectorAll('.trajectory-content');
            const buttons = document.querySelectorAll('.toggle-button');
            
            contents.forEach(content => content.classList.remove('collapsed'));
            buttons.forEach(button => button.textContent = '▼ Collapse');
        }
    """

    return f"""
<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>Zork Trajectories</title>
    <style>{css}</style>
    <script>{javascript}</script>
</head>
<body>
    {header}
    <div style="margin: 10px 0;">
        <button onclick="expandAll()">Expand All</button>
        <button onclick="collapseAll()">Collapse All</button>
    </div>
    {stats_html}
    {trajectories_html}
</body>
</html>
"""


def _render_trajectory(traj: dict) -> str:
    """Render a single trajectory."""
    status_class = f"status-{traj['status']}"

    steps_html = ""
    for step in traj["steps"]:
        step_html = ""

        message = step["message"]
        role = message["role"]
        content = message["content"]

        # Choose background color based on role
        if role == "user":
            bg_class = "action"  # Blue background for user messages
        elif role == "assistant":
            bg_class = "response"  # Orange background for assistant messages
        else:
            bg_class = "observation"  # Purple background for system messages

        step_html += f'<div class="{bg_class}"><strong>{role.title()}:</strong><br><pre>{content}</pre></div>'

        # Add reasoning if present
        if "reasoning" in message:
            step_html += f'<div class="observation"><strong>Reasoning:</strong><br><pre>{message["reasoning"]}</pre></div>'

        steps_html += f'<div class="step">{step_html}</div>'

    return f"""
    <div class="trajectory">
        <div class="trajectory-header" onclick="toggleTrajectory({traj["index"]})">
            <span>
                Trajectory #{traj["index"]} - {traj["model"]} 
                <span class="{status_class}">[{traj["status"].upper()}]</span>
            </span>
            <span class="toggle-button" id="button-{traj["index"]}">▼ Collapse</span>
        </div>
        <div class="trajectory-content" id="content-{traj["index"]}">
            {steps_html}
        </div>
    </div>
    """


def main(task: TaskChoice = "connections"):
    if task == "connections":
        eval_single_step(task)
    elif task == "zork":
        render_zork_trajectories()


if __name__ == "__main__":
    fire.Fire(main)

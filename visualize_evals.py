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

        # Store raw data for JavaScript rendering instead of pre-building all trajectories
        trajectories = out_rows

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
        "game": row.get("game", "Unknown"),
        "steps": steps,
    }


def _render_html_template(stats: dict, trajectories: list[EvalsOutRow]) -> str:
    """Render the complete HTML template."""

    css = """
        body {
            font-family: sans-serif;
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
            font-size: 14px;
        }
        pre {
            white-space: pre-wrap;
            text-wrap: auto;
        }
        .navigation {
            background: #f8f9fa;
            padding: 15px;
            border: 1px solid #ddd;
            margin-bottom: 20px;
            border-radius: 5px;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }
        .nav-button {
            background: #007bff;
            color: white;
            border: none;
            padding: 8px 16px;
            border-radius: 4px;
            cursor: pointer;
            font-size: 14px;
        }
        .nav-button:hover {
            background: #0056b3;
        }
        .nav-button:disabled {
            background: #6c757d;
            cursor: not-allowed;
        }
        .trajectory-info {
            font-weight: bold;
            color: #495057;
        }
        .trajectory { 
            border: 1px solid #ddd; 
            margin-bottom: 20px;
        }
        .trajectory-header { 
            background: #f5f5f5; 
            padding: 15px; 
            font-weight: bold; 
            display: flex;
            justify-content: space-between;
            align-items: center;
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
        .trajectory-content {
            padding: 15px;
        }
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

    trajectories_html = '<div id="trajectory-container" class="trajectory"></div>'

    # Convert trajectories to JavaScript data, properly handling numpy arrays
    import json
    import numpy as np
    
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
    converted_trajectories = [convert_numpy_arrays(dict(row)) for row in trajectories]
    trajectories_json = json.dumps(converted_trajectories, indent=None)
    
    javascript = f"""
        let currentTrajectory = 0;
        let totalTrajectories = 0;
        let trajectoryData = {trajectories_json};
        
        function initializeTrajectories() {{
            totalTrajectories = trajectoryData.length;
            
            if (totalTrajectories > 0) {{
                showTrajectory(0);
            }}
            
            updateNavigation();
        }}
        
        function buildTrajectoryHTML(index, row) {{
            const conversation = row.conversation;
            const statusClass = `status-${{row.status}}`;
            
            let stepsHtml = '';
            for (let i = 0; i < conversation.length; i++) {{
                const message = conversation[i];
                const role = message.role;
                const content = message.content;
                
                let bgClass = 'observation';
                if (role === 'user') {{
                    bgClass = 'action';
                }} else if (role === 'assistant') {{
                    bgClass = 'response';
                }}
                
                let stepHtml = `<div class="${{bgClass}}"><strong>${{role.charAt(0).toUpperCase() + role.slice(1)}}:</strong><br><pre>${{content}}</pre></div>`;
                
                if (message.reasoning) {{
                    stepHtml += `<div class="observation"><strong>Reasoning:</strong><br><pre>${{message.reasoning}}</pre></div>`;
                }}
                
                stepsHtml += `<div class="step">${{stepHtml}}</div>`;
            }}
            
            return `
                <div class="trajectory-header">
                    <span>
                        Trajectory #${{index + 1}} - ${{row.model}} - Game: ${{row.game}}
                        <span class="${{statusClass}}">[${{row.status.toUpperCase()}}]</span>
                    </span>
                </div>
                <div class="trajectory-content">
                    ${{stepsHtml}}
                </div>
            `;
        }}
        
        function showTrajectory(index) {{
            if (index < 0 || index >= totalTrajectories) return;
            
            const container = document.getElementById('trajectory-container');
            const row = trajectoryData[index];
            
            container.innerHTML = buildTrajectoryHTML(index, row);
            currentTrajectory = index;
            updateNavigation();
        }}
        
        function nextTrajectory() {{
            if (currentTrajectory < totalTrajectories - 1) {{
                showTrajectory(currentTrajectory + 1);
            }}
        }}
        
        function previousTrajectory() {{
            if (currentTrajectory > 0) {{
                showTrajectory(currentTrajectory - 1);
            }}
        }}
        
        function updateNavigation() {{
            const prevButton = document.getElementById('prev-button');
            const nextButton = document.getElementById('next-button');
            const trajectoryInfo = document.getElementById('trajectory-info');
            
            if (prevButton) prevButton.disabled = (currentTrajectory === 0);
            if (nextButton) nextButton.disabled = (currentTrajectory === totalTrajectories - 1);
            
            if (trajectoryInfo) {{
                trajectoryInfo.textContent = `Trajectory ${{currentTrajectory + 1}} of ${{totalTrajectories}}`;
            }}
        }}
        
        // Initialize when page loads
        document.addEventListener('DOMContentLoaded', initializeTrajectories);
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
    {stats_html}
    <div class="navigation">
        <button id="prev-button" class="nav-button" onclick="previousTrajectory()">← Previous</button>
        <span id="trajectory-info" class="trajectory-info">Loading...</span>
        <button id="next-button" class="nav-button" onclick="nextTrajectory()">Next →</button>
    </div>
    {trajectories_html}
</body>
</html>
"""




def main(task: TaskChoice = "connections"):
    if task == "connections":
        eval_single_step(task)
    elif task == "zork":
        render_zork_trajectories()


if __name__ == "__main__":
    fire.Fire(main)

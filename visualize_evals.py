import os
import pandas as pd
from pathlib import Path
from loguru import logger
import matplotlib.pyplot as plt


def main():
    eval_results_path = Path("eval_results")

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

    os.makedirs("eval_results/figures", exist_ok=True)
    plt.savefig("eval_results/figures/average_scores.png")


if __name__ == "__main__":
    main()

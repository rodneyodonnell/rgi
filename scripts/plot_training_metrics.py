#!/usr/bin/env python3
"""Plot training metrics from AlphaZero training runs.

Usage:
    python -m scripts.plot_training_metrics training_runs/20250214_123456/metrics.json
"""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def plot_metrics(metrics_file: Path) -> None:
    """Plot training metrics from a metrics.json file."""
    with open(metrics_file) as f:
        metrics = json.load(f)

    iterations = [m["iteration"] for m in metrics]
    win_rates_random = [m["random_opponent"]["random"]["win_rate"] for m in metrics]
    win_rates_mcts = [m["random_opponent"]["random_mcts"]["win_rate"] for m in metrics]

    # Plot win rates
    plt.figure(figsize=(12, 6))
    plt.plot(iterations, win_rates_random, label="vs Random", marker="o")
    plt.plot(iterations, win_rates_mcts, label="vs Random MCTS", marker="o")

    # Add previous model win rates if available
    prev_model_metrics = [m.get("previous_snapshot", {}).get("previous_model", {}).get("win_rate") for m in metrics]
    if any(x is not None for x in prev_model_metrics):
        plt.plot(
            [i for i, x in zip(iterations, prev_model_metrics) if x is not None],
            [x for x in prev_model_metrics if x is not None],
            label="vs Previous Model",
            marker="o",
        )

    plt.xlabel("Iteration")
    plt.ylabel("Win Rate")
    plt.title("AlphaZero Training Progress")
    plt.grid(True)
    plt.legend()

    # Save plot
    output_file = metrics_file.parent / "training_progress.png"
    plt.savefig(output_file)
    print(f"Plot saved to {output_file}")


def main() -> None:
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Plot AlphaZero training metrics.")
    parser.add_argument("metrics_file", type=Path, help="Path to metrics.json file")
    args = parser.parse_args()

    plot_metrics(args.metrics_file)


if __name__ == "__main__":
    main()

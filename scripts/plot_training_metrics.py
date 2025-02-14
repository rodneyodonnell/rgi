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

    # Handle both old and new metrics format
    win_rates_random = []
    game_lengths = []
    for m in metrics:
        if "random_opponent" in m:
            # New format
            win_rates_random.append(m["random_opponent"]["win_rate"])
            game_lengths.append(m["random_opponent"]["avg_game_length"])
        else:
            # Old format
            win_rates_random.append(m["win_rate_vs_random"])
            game_lengths.append(m.get("avg_game_length", 0.0))

    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))

    # Plot win rates
    ax1.plot(iterations, win_rates_random, label="vs Random", marker="o")
    ax1.set_xlabel("Iteration")
    ax1.set_ylabel("Win Rate")
    ax1.set_title("Win Rate vs Random")
    ax1.grid(True)
    ax1.legend()

    # Plot game lengths
    ax2.plot(iterations, game_lengths, label="Average Game Length", marker="o", color="orange")
    ax2.set_xlabel("Iteration")
    ax2.set_ylabel("Moves")
    ax2.set_title("Average Game Length")
    ax2.grid(True)
    ax2.legend()

    # Add previous model win rates if available
    prev_model_metrics = []
    for m in metrics:
        if "previous_snapshot" in m:
            prev_model_metrics.append(m["previous_snapshot"]["win_rate"])
        else:
            prev_model_metrics.append(None)

    if any(x is not None for x in prev_model_metrics):
        ax1.plot(
            [i for i, x in zip(iterations, prev_model_metrics) if x is not None],
            [x for x in prev_model_metrics if x is not None],
            label="vs Previous Model",
            marker="o",
        )
        ax1.legend()

    plt.tight_layout()

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

from __future__ import annotations

"""
Ray-based distributed self-play for AlphaZero training.
"""

import argparse
import dataclasses
import os
from pathlib import Path
from typing import Any, Sequence, cast

import numpy as np
import ray
import tensorflow as tf
from tensorflow.keras import Model
from tqdm import tqdm

# Force CPU usage
tf.config.set_visible_devices([], "GPU")

from rgi.core.archive import RowFileArchiver
from rgi.core.game_runner import GameRunner
from rgi.core.trajectory import GameTrajectory
from rgi.games.count21.count21 import Count21Game, Count21State
from rgi.players.alphazero.alphazero import AlphaZeroPlayer, MCTSData
from rgi.players.alphazero.alphazero_tf import PVNetwork_Count21_TF, TFPVNetworkWrapper


@dataclasses.dataclass
class SelfPlayConfig:
    """Configuration for distributed self-play."""

    num_workers: int = 4  # Number of Ray workers
    num_games: int = 100  # Total number of self-play games
    num_simulations: int = 100  # MCTS simulations per move
    weights_path: str | Path = ""  # Path to model weights
    output_path: str | Path = ""  # Path to save trajectories
    verbose: bool = False  # Whether to print progress

    def __post_init__(self) -> None:
        """Convert paths to strings."""
        self.weights_path = str(self.weights_path)
        self.output_path = str(self.output_path)


# Ray's @remote decorator dynamically adds the 'remote' attribute at runtime
@ray.remote
class SelfPlayWorker:
    """Ray worker for running self-play games."""

    def __init__(self, config: SelfPlayConfig) -> None:
        self.config = config
        self.game = Count21Game(num_players=2, target=21, max_guess=3)

        # Create and load model
        state = self.game.initial_state()
        state_array = np.array([state.score, state.current_player], dtype=np.float32)
        state_dim = state_array.shape[0]
        num_actions = len(self.game.legal_actions(state))
        num_players = self.game.num_players(state)

        self.model = PVNetwork_Count21_TF(state_dim=state_dim, num_actions=num_actions, num_players=num_players)
        # Build model via a dummy forward pass
        dummy_input = tf.convert_to_tensor(state_array.reshape(1, -1))
        _ = self.model(dummy_input)
        self.model.load_weights(config.weights_path)
        self.model_wrapper: TFPVNetworkWrapper = TFPVNetworkWrapper(cast(PVNetwork_Count21_TF, self.model))

    def run_games(self, num_games: int) -> list[GameTrajectory[Count21State, int, MCTSData[int]]]:
        """Run self-play games and return trajectories."""
        trajectories: list[GameTrajectory[Count21State, int, MCTSData[int]]] = []

        # Only show progress if verbose and this is the first worker
        show_progress = self.config.verbose and ray.get_runtime_context().get_worker_id() == 0

        # Use tqdm only if showing progress
        game_range = tqdm(range(num_games), desc="Games", disable=not show_progress)
        for _ in game_range:
            players = [
                AlphaZeroPlayer(self.game, self.model_wrapper, num_simulations=self.config.num_simulations)
                for _ in range(2)
            ]
            runner = GameRunner(self.game, players, verbose=False)  # Always set verbose=False for workers
            trajectory = runner.run()
            trajectories.append(trajectory)

            if show_progress:
                # Update progress bar with game stats
                final_rewards = trajectory.final_reward
                game_range.set_postfix(
                    moves=len(trajectory.actions),
                    p1_reward=f"{final_rewards[0]:.1f}",
                    p2_reward=f"{final_rewards[1]:.1f}",
                )

        return trajectories


def run_distributed_selfplay(config: SelfPlayConfig) -> list[GameTrajectory[Count21State, int, MCTSData[int]]]:
    """Run distributed self-play using Ray.

    Returns:
        List of game trajectories from all workers.
    """
    # Initialize Ray if not already initialized
    if not ray.is_initialized():
        ray.init(
            _system_config={
                "object_store_memory": int(10e9),  # 10GB
                "object_store_full_delay_ms": 100,
                "metrics_report_interval_ms": 2000,  # Report metrics every 2 seconds
                "metrics_export_port": 8080,  # Port for Prometheus metrics
            },
            logging_level="WARNING",  # Reduce Ray logging
            dashboard_host="0.0.0.0",  # Allow external access
            dashboard_port=8265,  # Match the port in docker-compose
            include_dashboard=True,  # Ensure dashboard is enabled
            _temp_dir="/tmp/ray",  # Use shorter path for Ray temp files
        )

    # Initialize workers
    workers: list[Any] = [SelfPlayWorker.remote(config) for _ in range(config.num_workers)]

    # Distribute games among workers
    games_per_worker = config.num_games // config.num_workers
    remaining_games = config.num_games % config.num_workers

    # Launch tasks
    tasks: list[Any] = []
    for i, worker in enumerate(workers):
        num_games = games_per_worker + (1 if i < remaining_games else 0)
        if num_games > 0:
            tasks.append(worker.run_games.remote(num_games))

    # Wait for all tasks and combine results with progress bar
    all_trajectories: list[GameTrajectory[Count21State, int, MCTSData[int]]] = []
    with tqdm(total=len(tasks), desc="Workers", disable=not config.verbose) as pbar:
        while tasks:
            done_id, tasks = ray.wait(tasks)
            trajectories = ray.get(done_id[0])
            all_trajectories.extend(trajectories)
            pbar.update(1)

    if config.verbose:
        print(f"\nGenerated {len(all_trajectories)} trajectories")

    return all_trajectories


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run distributed self-play.")
    parser.add_argument("--num_workers", type=int, default=4, help="Number of Ray workers.")
    parser.add_argument("--num_games", type=int, default=100, help="Total number of games to play.")
    parser.add_argument("--num_simulations", type=int, default=100, help="MCTS simulations per move.")
    parser.add_argument("--weights_path", type=str, required=True, help="Path to model weights.")
    parser.add_argument("--output", type=str, required=True, help="Path to save trajectories.")
    parser.add_argument("--verbose", action="store_true", help="Print progress.")

    args = parser.parse_args()
    config = SelfPlayConfig(
        num_workers=args.num_workers,
        num_games=args.num_games,
        num_simulations=args.num_simulations,
        weights_path=args.weights_path,
        output_path=args.output,
        verbose=args.verbose,
    )

    run_distributed_selfplay(config)

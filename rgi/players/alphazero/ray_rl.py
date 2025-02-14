from __future__ import annotations

"""
Ray-based distributed self-play for AlphaZero training.
"""

import argparse
import dataclasses
import os
from typing import Any, Sequence, cast

import numpy as np
import ray
import tensorflow as tf
from tensorflow.keras import Model
from tqdm import tqdm

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
    weights_path: str = ""  # Path to model weights
    output_path: str = ""  # Path to save trajectories
    verbose: bool = False  # Whether to print progress


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
        self.model_wrapper = TFPVNetworkWrapper(self.model)

    def run_games(self, num_games: int) -> list[GameTrajectory[Count21State, int, MCTSData[int]]]:
        """Run self-play games and return trajectories."""
        trajectories: list[GameTrajectory[Count21State, int, MCTSData[int]]] = []
        for _ in range(num_games):
            players = [
                AlphaZeroPlayer(self.game, self.model_wrapper, num_simulations=self.config.num_simulations)
                for _ in range(2)
            ]
            runner = GameRunner(self.game, players, verbose=self.config.verbose)
            trajectory = runner.run()
            trajectories.append(trajectory)
        return trajectories


def run_distributed_selfplay(config: SelfPlayConfig) -> None:
    """Run distributed self-play using Ray."""
    # Initialize Ray if not already initialized
    if not ray.is_initialized():
        ray.init()

    # Initialize workers
    workers = [SelfPlayWorker.remote(config) for _ in range(config.num_workers)]

    # Distribute games among workers
    games_per_worker = config.num_games // config.num_workers
    remaining_games = config.num_games % config.num_workers

    # Launch tasks
    tasks = []
    for i, worker in enumerate(workers):
        num_games = games_per_worker + (1 if i < remaining_games else 0)
        if num_games > 0:
            tasks.append(worker.run_games.remote(num_games))

    # Wait for all tasks and combine results
    all_trajectories: list[GameTrajectory[Count21State, int, MCTSData[int]]] = []
    for trajectories in ray.get(tasks):
        all_trajectories.extend(trajectories)

    # Save trajectories
    archiver = RowFileArchiver()
    archiver.write_items(config.output_path, all_trajectories)
    if config.verbose:
        print(f"Wrote {len(all_trajectories)} trajectories to {config.output_path}")


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

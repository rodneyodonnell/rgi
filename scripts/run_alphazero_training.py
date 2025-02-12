#!/usr/bin/env python3
"""Script for continuous AlphaZero training with model snapshots and evaluation.

Usage:
./scripts/run_alphazero_training.py --iterations 1000 --games-per-iter 100 --mcts-sims 50 --save-freq 50
nohup ./scripts/run_alphazero_training.py --iterations 100 --games-per-iter 100 --mcts-sims 50 --epochs 10 --eval-games 50 --save-freq 5 > training.log 2>&1

Fast Run:
./scripts/run_alphazero_training.py --iterations 2 --games-per-iter 5 --mcts-sims 10 --epochs 5 --eval-games 10 --save-freq 1

"""

import argparse
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any, NamedTuple

import numpy as np
import tensorflow as tf
from tqdm import tqdm

from rgi.core.archive import RowFileArchiver
from rgi.core.game_runner import GameRunner
from rgi.core.trajectory import GameTrajectory
from rgi.games.count21.count21 import Count21Game, Count21State
from rgi.players.alphazero.alphazero import AlphaZeroPlayer, MCTSData
from rgi.players.alphazero.alphazero_tf import PVNetwork_Count21_TF, TFPVNetworkWrapper, train_model
from rgi.players.random_player.random_player import RandomPlayer


class TrainingConfig(NamedTuple):
    """Configuration for training run."""

    num_iterations: int = 100  # Number of training iterations
    games_per_iteration: int = 100  # Self-play games per iteration
    mcts_simulations: int = 100  # MCTS simulations per move
    training_epochs: int = 20  # Training epochs per iteration
    eval_games: int = 100  # Number of evaluation games
    save_frequency: int = 10  # Save model every N iterations
    output_dir: str = "training_runs"  # Directory to save models and metrics


def create_initial_model() -> PVNetwork_Count21_TF:
    """Create and initialize the model."""
    game = Count21Game(num_players=2, target=21, max_guess=3)
    state = game.initial_state()
    state_array = np.array([state.score, state.current_player], dtype=np.float32)
    state_dim = state_array.shape[0]
    num_actions = len(game.legal_actions(state))
    num_players = game.num_players(state)

    model = PVNetwork_Count21_TF(state_dim=state_dim, num_actions=num_actions, num_players=num_players)
    # Build model via a dummy forward pass
    model(tf.convert_to_tensor(state_array.reshape(1, -1)))
    return model


def evaluate_model(
    model: PVNetwork_Count21_TF,
    opponent_model: PVNetwork_Count21_TF | None,
    num_games: int,
    mcts_simulations: int,
) -> dict[str, float]:
    """Evaluate model against opponent (random player if opponent_model is None)."""
    game = Count21Game(num_players=2, target=21, max_guess=3)
    model_wrapper = TFPVNetworkWrapper(model)
    wins = 0
    total_moves = 0

    for game_idx in range(num_games):
        # Alternate playing first and second
        if game_idx % 2 == 0:
            if opponent_model is None:
                players = [
                    AlphaZeroPlayer(game, model_wrapper, num_simulations=mcts_simulations),
                    RandomPlayer(),
                ]
            else:
                opponent_wrapper = TFPVNetworkWrapper(opponent_model)
                players = [
                    AlphaZeroPlayer(game, model_wrapper, num_simulations=mcts_simulations),
                    AlphaZeroPlayer(game, opponent_wrapper, num_simulations=mcts_simulations),
                ]
        else:
            if opponent_model is None:
                players = [
                    RandomPlayer(),
                    AlphaZeroPlayer(game, model_wrapper, num_simulations=mcts_simulations),
                ]
            else:
                opponent_wrapper = TFPVNetworkWrapper(opponent_model)
                players = [
                    AlphaZeroPlayer(game, opponent_wrapper, num_simulations=mcts_simulations),
                    AlphaZeroPlayer(game, model_wrapper, num_simulations=mcts_simulations),
                ]

        runner = GameRunner(game, players, verbose=False)
        trajectory = runner.run()

        # Check if model won (accounting for playing first/second)
        model_idx = 0 if game_idx % 2 == 0 else 1
        if trajectory.final_reward[model_idx] > 0:
            wins += 1
        total_moves += len(trajectory.actions)

    metrics = {
        "win_rate": wins / num_games,
        "avg_game_length": total_moves / num_games,
    }
    return metrics


def main(config: TrainingConfig) -> None:
    """Main training loop."""
    # Set up output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = Path(config.output_dir) / timestamp
    models_dir = run_dir / "models"
    models_dir.mkdir(parents=True)

    # Register custom model class
    tf.keras.utils.get_custom_objects()["PVNetwork_Count21_TF"] = PVNetwork_Count21_TF

    # Save config
    with open(run_dir / "config.json", "w") as f:
        json.dump(config._asdict(), f, indent=2)

    # Initialize model and metrics tracking
    current_model = create_initial_model()
    metrics_history: list[dict[str, Any]] = []
    archiver = RowFileArchiver()

    # Main training loop
    saved_model_paths: list[Path] = []
    for iteration in range(config.num_iterations):
        print(f"\nIteration {iteration + 1}/{config.num_iterations}")

        # Self-play phase
        print("Self-play phase...")
        model_wrapper = TFPVNetworkWrapper(current_model)
        trajectories: list[GameTrajectory[Count21State, int, MCTSData[int]]] = []

        for _ in tqdm(range(config.games_per_iteration)):
            players = [
                AlphaZeroPlayer(
                    Count21Game(num_players=2, target=21, max_guess=3),
                    model_wrapper,
                    num_simulations=config.mcts_simulations,
                )
                for _ in range(2)
            ]
            runner = GameRunner(Count21Game(num_players=2, target=21, max_guess=3), players, verbose=False)
            trajectory = runner.run()
            trajectories.append(trajectory)

        # Save trajectories
        trajectory_file = run_dir / f"trajectories_iter_{iteration}.npz"
        archiver.write_items(trajectories, str(trajectory_file))

        # Training phase
        print("Training phase...")
        current_model = train_model(trajectories, num_epochs=config.training_epochs)

        # Evaluation phase
        print("Evaluation phase...")
        metrics = {
            "iteration": iteration,
            "random_opponent": evaluate_model(current_model, None, config.eval_games, config.mcts_simulations),
        }

        # Save model snapshot
        if (iteration + 1) % config.save_frequency == 0:
            model_path = models_dir / f"model_iter_{iteration}.weights.h5"
            current_model.save_weights(str(model_path))
            print(f"Saved model snapshot to {model_path}")
            saved_model_paths.append(model_path)

            if len(saved_model_paths) > 1:
                previous_model_path = saved_model_paths[-2]
                if previous_model_path.exists():
                    previous_model = create_initial_model()
                    previous_model.load_weights(str(previous_model_path))
                    metrics["previous_snapshot"] = evaluate_model(
                        current_model, previous_model, config.eval_games, config.mcts_simulations
                    )

        metrics_history.append(metrics)

        # Save metrics
        with open(run_dir / "metrics.json", "w") as f:
            json.dump(metrics_history, f, indent=2)

        # Print current metrics
        print(f"\nCurrent metrics:")
        print(f"Win rate vs random: {metrics['random_opponent']['win_rate']:.2%}")
        if "previous_snapshot" in metrics:
            print(f"Win rate vs previous snapshot: {metrics['previous_snapshot']['win_rate']:.2%}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run continuous AlphaZero training.")
    parser.add_argument("--iterations", type=int, default=100, help="Number of training iterations")
    parser.add_argument("--games-per-iter", type=int, default=100, help="Self-play games per iteration")
    parser.add_argument("--mcts-sims", type=int, default=100, help="MCTS simulations per move")
    parser.add_argument("--epochs", type=int, default=20, help="Training epochs per iteration")
    parser.add_argument("--eval-games", type=int, default=100, help="Number of evaluation games")
    parser.add_argument("--save-freq", type=int, default=10, help="Save model every N iterations")
    parser.add_argument("--output-dir", type=str, default="training_runs", help="Output directory")

    args = parser.parse_args()
    config = TrainingConfig(
        num_iterations=args.iterations,
        games_per_iteration=args.games_per_iter,
        mcts_simulations=args.mcts_sims,
        training_epochs=args.epochs,
        eval_games=args.eval_games,
        save_frequency=args.save_freq,
        output_dir=args.output_dir,
    )

    main(config)

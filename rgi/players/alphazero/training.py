#!/usr/bin/env python3
"""AlphaZero training script with distributed self-play and continuous evaluation.

Usage:
    python -m rgi.players.alphazero.training --iterations 100 --games-per-iter 100 --mcts-sims 50 --save-freq 50
    nohup python -m rgi.players.alphazero.training [args] > training_$(date +%Y%m%d_%H%M%S).log 2>&1 &

Fast Run:
    python -m rgi.players.alphazero.training --iterations 2 --games-per-iter 5 --mcts-sims 10 --epochs 5 --eval-games 10 --save-freq 1
"""

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import NamedTuple, NotRequired, TypedDict

import numpy as np
import ray
import tensorflow as tf
from tqdm import tqdm

# Force CPU usage
tf.config.set_visible_devices([], "GPU")

from rgi.core.archive import RowFileArchiver
from rgi.core.game_runner import GameRunner
from rgi.core.trajectory import GameTrajectory
from rgi.games.count21.count21 import Count21Game, Count21State
from rgi.players.alphazero.alphazero import AlphaZeroPlayer, MCTSData
from rgi.players.alphazero.alphazero_tf import PVNetwork_Count21_TF, TFPVNetworkWrapper, train_model
from rgi.players.alphazero.ray_rl import SelfPlayConfig, run_distributed_selfplay
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
    num_workers: int = 4  # Number of Ray workers for distributed self-play


class OpponentMetrics(TypedDict):
    win_rate: float
    avg_game_length: NotRequired[float]


class EvaluationMetrics(TypedDict):
    random: OpponentMetrics
    random_mcts: OpponentMetrics
    previous_model: NotRequired[OpponentMetrics]


class IterationMetrics(TypedDict):
    iteration: int
    random_opponent: EvaluationMetrics
    previous_snapshot: NotRequired[EvaluationMetrics]


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
    dummy_input = tf.convert_to_tensor(state_array.reshape(1, -1))
    _ = model(dummy_input)  # type: ignore[func-returns-value]
    return model  # type: ignore[return-value]


def evaluate_model(
    model: PVNetwork_Count21_TF,
    baseline_model: PVNetwork_Count21_TF,
    num_games: int,
    mcts_simulations: int,
) -> EvaluationMetrics:
    """Evaluate model against different opponents.

    Returns:
        Dictionary containing win rates and game lengths against different opponents.
    """
    game = Count21Game(num_players=2, target=21, max_guess=3)
    model_wrapper = TFPVNetworkWrapper(model)
    baseline_wrapper = TFPVNetworkWrapper(baseline_model)

    # Track metrics for different opponents
    random_wins = 0
    mcts_wins = 0
    total_moves = 0

    # First evaluate against random player
    for game_idx in range(num_games):
        # Alternate playing first and second
        if game_idx % 2 == 0:
            players = [
                AlphaZeroPlayer(game, model_wrapper, num_simulations=mcts_simulations),
                RandomPlayer(),
            ]
        else:
            players = [
                RandomPlayer(),
                AlphaZeroPlayer(game, model_wrapper, num_simulations=mcts_simulations),
            ]

        runner = GameRunner(game, players, verbose=False)
        trajectory = runner.run()

        # Check if model won (accounting for playing first/second)
        model_idx = 0 if game_idx % 2 == 0 else 1
        if trajectory.final_reward[model_idx] > 0:
            random_wins += 1
        total_moves += len(trajectory.actions)

    # Then evaluate against random MCTS
    for game_idx in range(num_games):
        if game_idx % 2 == 0:
            players = [
                AlphaZeroPlayer(game, model_wrapper, num_simulations=mcts_simulations),
                AlphaZeroPlayer(game, baseline_wrapper, num_simulations=mcts_simulations),
            ]
        else:
            players = [
                AlphaZeroPlayer(game, baseline_wrapper, num_simulations=mcts_simulations),
                AlphaZeroPlayer(game, model_wrapper, num_simulations=mcts_simulations),
            ]

        runner = GameRunner(game, players, verbose=False)
        trajectory = runner.run()

        # Check if model won (accounting for playing first/second)
        model_idx = 0 if game_idx % 2 == 0 else 1
        if trajectory.final_reward[model_idx] > 0:
            mcts_wins += 1

    metrics: EvaluationMetrics = {
        "random": {
            "win_rate": random_wins / num_games,
            "avg_game_length": total_moves / num_games,
        },
        "random_mcts": {
            "win_rate": mcts_wins / num_games,
        },
    }

    return metrics


def main(config: TrainingConfig) -> None:
    """Main training loop."""
    # Set up output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = Path(config.output_dir) / timestamp
    models_dir = run_dir / "models"
    models_dir.mkdir(parents=True, exist_ok=True)

    try:
        # Initialize Ray if not already initialized
        if not ray.is_initialized():
            ray.init(
                logging_level="WARNING",  # Reduce Ray logging
                dashboard_host="0.0.0.0",  # Allow external access
                dashboard_port=8265,  # Match the port in docker-compose
                include_dashboard=True,  # Ensure dashboard is enabled
                _system_config={
                    "metrics_report_interval_ms": 2000,  # Report metrics every 2 seconds
                    "metrics_export_port": 8080,  # Port for Prometheus metrics
                },
                _temp_dir="/tmp/ray",  # Use shorter path for Ray temp files
            )

        # Save config
        with open(run_dir / "config.json", "w") as f:
            json.dump(config._asdict(), f, indent=2)

        # Initialize models and metrics tracking
        current_model = create_initial_model()
        baseline_model = create_initial_model()  # Create a fixed random model for evaluation
        metrics_history: list[IterationMetrics] = []
        archiver = RowFileArchiver()

        print("\nStarting training run with config:")
        print(f"  Iterations: {config.num_iterations}")
        print(f"  Games per iteration: {config.games_per_iteration}")
        print(f"  MCTS simulations: {config.mcts_simulations}")
        print(f"  Training epochs: {config.training_epochs}")
        print(f"  Eval games: {config.eval_games}")
        print(f"  Save frequency: {config.save_frequency}")
        print(f"  Ray workers: {config.num_workers}")
        print(f"  Output directory: {run_dir}\n")

        # Main training loop
        saved_model_paths: list[Path] = []
        with tqdm(total=config.num_iterations, desc="Training Progress") as pbar:
            for iteration in range(config.num_iterations):
                print(f"\n{'='*80}")
                print(f"Iteration {iteration + 1}/{config.num_iterations}")
                print(f"{'='*80}")

                # Save current model weights for distributed workers
                temp_weights_path = models_dir / f"temp_weights_iter_{iteration}.weights.h5"
                current_model.save_weights(str(temp_weights_path))

                # Self-play phase using Ray
                print("\nSelf-play phase...")
                trajectory_file = run_dir / f"trajectories_iter_{iteration}.npz"
                selfplay_config = SelfPlayConfig(
                    num_workers=config.num_workers,
                    num_games=config.games_per_iteration,
                    num_simulations=config.mcts_simulations,
                    weights_path=str(temp_weights_path),
                    output_path=str(trajectory_file),
                    verbose=True,
                )
                trajectories = run_distributed_selfplay(selfplay_config)

                # Save trajectories
                archiver.write_items(trajectories, str(trajectory_file))  # type: ignore[arg-type]
                print(f"Wrote {len(trajectories)} trajectories to {trajectory_file}")

                # Training phase
                print("\nTraining phase...")
                current_model = train_model(trajectories, num_epochs=config.training_epochs)

                # Clean up temporary weights
                temp_weights_path.unlink()

                # Evaluation phase
                print("\nEvaluation phase...")
                eval_metrics = evaluate_model(current_model, baseline_model, config.eval_games, config.mcts_simulations)

                metrics: IterationMetrics = {
                    "iteration": iteration,
                    "random_opponent": eval_metrics,
                }

                # Save model snapshot
                if (iteration + 1) % config.save_frequency == 0:
                    model_path = models_dir / f"model_iter_{iteration}.weights.h5"
                    current_model.save_weights(str(model_path))
                    print(f"\nSaved model snapshot to {model_path}")
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

                # Print current metrics in a clear format
                print("\nCurrent Metrics:")
                print("-" * 40)
                print(f"Win rate vs random:      {metrics['random_opponent']['random']['win_rate']:.1%}")
                print(f"Win rate vs random MCTS: {metrics['random_opponent']['random_mcts']['win_rate']:.1%}")
                print(f"Avg game length:         {metrics['random_opponent']['random']['avg_game_length']:.1f}")

                previous_snapshot = metrics.get("previous_snapshot")
                if previous_snapshot is not None:
                    previous_model = previous_snapshot.get("previous_model")
                    if previous_model is not None:
                        print(f"Win rate vs prev model:  {previous_model['win_rate']:.1%}")
                print("-" * 40)

                # Update progress bar with win rates
                pbar.set_postfix(
                    random=f"{metrics['random_opponent']['random']['win_rate']:.1%}",
                    mcts=f"{metrics['random_opponent']['random_mcts']['win_rate']:.1%}",
                )
                pbar.update(1)

        print("\nTraining complete!")
        print(f"Models and metrics saved in {run_dir}")
    finally:
        # Ensure Ray is shut down
        if ray.is_initialized():
            ray.shutdown()
            print("\nRay has been shut down.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run continuous AlphaZero training.")
    parser.add_argument("--iterations", type=int, default=100, help="Number of training iterations")
    parser.add_argument("--games-per-iter", type=int, default=100, help="Self-play games per iteration")
    parser.add_argument("--mcts-sims", type=int, default=100, help="MCTS simulations per move")
    parser.add_argument("--epochs", type=int, default=20, help="Training epochs per iteration")
    parser.add_argument("--eval-games", type=int, default=100, help="Number of evaluation games")
    parser.add_argument("--save-freq", type=int, default=10, help="Save model every N iterations")
    parser.add_argument("--output-dir", type=str, default="training_runs", help="Output directory")
    parser.add_argument("--num-workers", type=int, default=4, help="Number of Ray workers")

    args = parser.parse_args()
    config = TrainingConfig(
        num_iterations=args.iterations,
        games_per_iteration=args.games_per_iter,
        mcts_simulations=args.mcts_sims,
        training_epochs=args.epochs,
        eval_games=args.eval_games,
        save_frequency=args.save_freq,
        output_dir=args.output_dir,
        num_workers=args.num_workers,
    )

    main(config)

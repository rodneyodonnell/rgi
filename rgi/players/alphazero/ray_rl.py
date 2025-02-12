"""
Ray-based distributed self-play for AlphaZero training.
"""

import argparse
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, TypeVar, cast

import numpy as np
import ray  # pylint: disable=no-member
import tensorflow as tf
from tqdm import tqdm

from rgi.core.archive import RowFileArchiver
from rgi.core.game_runner import GameRunner
from rgi.core.trajectory import GameTrajectory
from rgi.games.count21.count21 import Count21Game, Count21State
from rgi.players.alphazero.alphazero import AlphaZeroPlayer
from rgi.players.alphazero.alphazero_tf import PVNetwork_Count21_TF, TFPVNetworkWrapper

T = TypeVar("T")
# Ray's ObjectRef type is used for async task references
ObjectRef = ray.ObjectRef[T]

if TYPE_CHECKING:
    # Type hints for Ray's remote functionality
    class RemoteSelfPlayWorker:
        @staticmethod
        def remote(*_args: Any, **_kwargs: Any) -> Any: ...

        def run_game(self, *_args: Any, **_kwargs: Any) -> GameTrajectory[Count21State, int]: ...


@dataclass
class SelfPlayConfig:
    """Configuration for distributed self-play."""

    num_workers: int
    num_games: int
    num_simulations: int
    weights_path: str
    output_path: str
    verbose: bool = False


# Ray's @remote decorator dynamically adds the 'remote' attribute at runtime
@ray.remote
class SelfPlayWorker:  # pylint: disable=too-few-public-methods
    """Worker class for distributed self-play using Ray."""

    def __init__(self, num_simulations: int) -> None:
        self.num_simulations = num_simulations

    def run_game(self, weights_path: str) -> GameTrajectory[Count21State, int]:
        """Run a single self-play game using the provided model weights."""
        game = Count21Game(num_players=2, target=21, max_guess=3)
        # Create a new model and load weights.
        state = game.initial_state()
        state_array = np.array([state.score, state.current_player], dtype=np.float32)
        state_dim = state_array.shape[0]
        num_actions = len(game.legal_actions(state))
        num_players = game.num_players(state)

        # Create and build the model
        model = PVNetwork_Count21_TF(state_dim=state_dim, num_actions=num_actions, num_players=num_players)
        # Build model via a dummy forward pass.
        model(tf.convert_to_tensor(state_array.reshape(1, -1)))
        model.load_weights(weights_path)
        # We know this is our custom model type
        wrapped_model = TFPVNetworkWrapper(model)  # type: ignore[arg-type]  # Model is actually PVNetwork_Count21_TF

        # Create players.
        players = [
            AlphaZeroPlayer[Count21Game, Count21State, int](game, wrapped_model, num_simulations=self.num_simulations)
            for _ in range(num_players)
        ]

        # Run game.
        runner = GameRunner(game, players, verbose=False)
        return runner.run()


def run_distributed_selfplay(config: SelfPlayConfig) -> None:
    """Run distributed self-play using Ray."""
    # Initialize Ray.
    if not ray.is_initialized():
        ray.init()

    # Create workers.
    # We use RemoteSelfPlayWorker for better type hints with Ray
    workers: list[Any] = [
        cast(RemoteSelfPlayWorker, SelfPlayWorker).remote(config.num_simulations)  # pylint: disable=no-member
        for _ in range(config.num_workers)
    ]

    # Distribute games among workers.
    games_per_worker = [config.num_games // config.num_workers] * config.num_workers
    for i in range(config.num_games % config.num_workers):
        games_per_worker[i] += 1

    # Launch tasks.
    tasks: list[ObjectRef[GameTrajectory[Count21State, int]]] = []
    for worker, n_games in zip(workers, games_per_worker):
        tasks.extend([worker.run_game.remote(config.weights_path) for _ in range(n_games)])  # pylint: disable=no-member

    # Collect results with progress bar.
    trajectories: list[GameTrajectory[Count21State, int]] = []
    for _ in tqdm(range(len(tasks)), desc="Self-play games", disable=not config.verbose):
        done_id, tasks = ray.wait(tasks)
        trajectory = cast(GameTrajectory[Count21State, int], ray.get(done_id[0]))
        trajectories.append(trajectory)

    # Save trajectories.
    archiver = RowFileArchiver()
    archiver.write_items(trajectories, config.output_path)
    if config.verbose:
        print(f"Wrote {len(trajectories)} trajectories to {config.output_path}")


def main() -> None:
    """Main entry point for distributed self-play."""
    parser = argparse.ArgumentParser(description="Run distributed self-play using Ray.")
    parser.add_argument("--num_workers", type=int, default=4, help="Number of Ray workers.")
    parser.add_argument("--num_games", type=int, default=100, help="Total number of games to play.")
    parser.add_argument("--num_simulations", type=int, default=50, help="MCTS simulations per move.")
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


if __name__ == "__main__":
    main()

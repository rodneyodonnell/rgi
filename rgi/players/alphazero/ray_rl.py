"""
Ray-based implementation of AlphaZero training loop.

This version parallelizes self-play across multiple workers for better performance.
Each worker runs complete MCTS games independently, with a shared neural network.
"""

from typing import List, Sequence
import ray
import numpy as np
from tqdm import tqdm
import tensorflow as tf
import time

from rgi.core.trajectory import GameTrajectory
from rgi.games.count21.count21 import Count21Game, Count21State, Count21Action
from rgi.players.alphazero.alphazero import AlphaZeroPlayer
from rgi.players.alphazero.alphazero_tf import PVNetwork_Count21_TF, TFPVNetworkWrapper


@ray.remote
class SelfPlayWorker:
    """Worker that runs self-play games in parallel using a shared neural network."""

    def __init__(self, weights_path: str, game_config: dict):
        """Initialize worker with a game and model weights path."""
        self.game = Count21Game(**game_config)

        # Create and initialize model
        init_state = self.game.initial_state()
        state_dim = 2  # score and current_player
        num_actions = len(self.game.legal_actions(init_state))
        num_players = self.game.num_players(init_state)

        model = PVNetwork_Count21_TF(state_dim=state_dim, num_actions=num_actions, num_players=num_players)
        # Build model with a dummy forward pass
        dummy_state = np.array([init_state.score, init_state.current_player], dtype=np.float32)
        model(tf.convert_to_tensor(dummy_state.reshape(1, -1)))
        # Load weights
        model.load_weights(weights_path)

        self.network = TFPVNetworkWrapper(model)

    def run_games(self, num_games: int, num_simulations: int) -> list[GameTrajectory[Count21State, Count21Action]]:
        """Run a batch of self-play games."""
        trajectories = []
        for _ in range(num_games):
            state = self.game.initial_state()
            states = [state]
            actions = []

            # Play until game is over
            while not self.game.is_terminal(state):
                legal_actions = self.game.legal_actions(state)
                # Create a new player for each move to ensure fresh MCTS tree
                player = AlphaZeroPlayer(self.game, self.network, num_simulations=num_simulations)
                action = player.select_action(state, legal_actions)
                actions.append(action)
                state = self.game.next_state(state, action)
                states.append(state)

            # Get final rewards and create trajectory
            rewards = self.game.reward_array(state)
            trajectory = GameTrajectory(
                game_states=states,
                actions=actions,
                final_reward=rewards.tolist(),
                action_player_ids=[self.game.current_player_id(s) for s in states[:-1]],
                incremental_rewards=[0.0] * len(actions),  # No incremental rewards in Count21
                num_players=self.game.num_players(state),
            )
            trajectories.append(trajectory)
        return trajectories


def generate_selfplay_data(
    model: PVNetwork_Count21_TF,
    num_games: int,
    num_simulations: int,
    num_workers: int,
    game_config: dict,
) -> list[GameTrajectory[Count21State, Count21Action]]:
    """
    Generate self-play data using multiple Ray workers.
    Each worker runs complete games independently.
    """
    # Initialize Ray if not already done
    if not ray.is_initialized():
        ray.init()

    # Save model weights to a temporary file
    temp_weights_path = "temp_model.weights.h5"
    model.save_weights(temp_weights_path)

    # Create workers with path to weights instead of model
    workers = [SelfPlayWorker.remote(temp_weights_path, game_config) for _ in range(num_workers)]

    # Distribute games across workers
    games_per_worker = [
        num_games // num_workers + (1 if i < num_games % num_workers else 0) for i in range(num_workers)
    ]
    trajectory_refs = [
        worker.run_games.remote(n_games, num_simulations) for worker, n_games in zip(workers, games_per_worker)
    ]

    # Wait for all games to complete and combine results
    all_trajectories = []
    for trajectories in tqdm(ray.get(trajectory_refs), desc="Collecting trajectories", total=len(trajectory_refs)):
        all_trajectories.extend(trajectories)

    # Clean up temporary file
    import os

    if os.path.exists(temp_weights_path):
        os.remove(temp_weights_path)

    return all_trajectories


def main():
    """Example usage of parallel self-play."""
    import argparse

    parser = argparse.ArgumentParser(description="Generate self-play data using Ray parallelization")
    parser.add_argument("--num_games", type=int, default=100, help="Number of games to play")
    parser.add_argument("--num_simulations", type=int, default=50, help="MCTS simulations per move")
    parser.add_argument("--num_workers", type=int, default=4, help="Number of parallel workers")
    parser.add_argument("--target", type=int, default=21, help="Target value for Count21Game")
    parser.add_argument("--max_guess", type=int, default=3, help="Maximum guess value")
    args = parser.parse_args()

    # Initialize game and model
    game_config = {"num_players": 2, "target": args.target, "max_guess": args.max_guess}
    game = Count21Game(**game_config)

    # Create and initialize model
    init_state = game.initial_state()
    state_dim = 2  # score and current_player
    num_actions = len(game.legal_actions(init_state))
    num_players = game.num_players(init_state)

    model = PVNetwork_Count21_TF(state_dim=state_dim, num_actions=num_actions, num_players=num_players)
    # Build model with a dummy forward pass
    dummy_state = np.array([init_state.score, init_state.current_player], dtype=np.float32)
    model(tf.convert_to_tensor(dummy_state.reshape(1, -1)))

    # Initialize Ray if not already done
    if not ray.is_initialized():
        ray.init()

    # Generate self-play data
    start_time = time.time()
    trajectories = generate_selfplay_data(
        model=model,
        num_games=args.num_games,
        num_simulations=args.num_simulations,
        num_workers=args.num_workers,
        game_config=game_config,
    )
    end_time = time.time()
    elapsed = end_time - start_time

    print(f"Generated {len(trajectories)} game trajectories in {elapsed:.2f} seconds")
    print(f"Games per second: {len(trajectories)/elapsed:.2f}")

    # Print some statistics
    win_counts = [0] * game.num_players(init_state)
    for traj in trajectories:
        winner = int(np.argmax(traj.final_reward))
        win_counts[winner] += 1
    for i, cnt in enumerate(win_counts):
        print(f"Player {i+1} wins: {(cnt/len(trajectories))*100:.1f}%")


if __name__ == "__main__":
    main()

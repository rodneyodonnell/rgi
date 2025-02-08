"""
Automated RL loop for AlphaZero training on Count21.

The loop:
  1. Generates self-play trajectories using the current PolicyValueNetwork.
  2. Trains a new TF policy-value network on these trajectories.
  3. Saves updated weights and updates the player.
  4. Evaluates the new model (e.g. win percentage).

Usage:
  python -m rgi.players.alphazero.automated_rl --num_iterations 10 --num_selfplay_games 100 \
    --num_simulations 50 --num_epochs 10
"""

import argparse
import cProfile
import pstats
from typing import Any, List
import numpy as np
from numpy.typing import NDArray
import tensorflow as tf
from tqdm import tqdm

from rgi.games.count21.count21 import Count21Game, Count21State, Count21Action
from rgi.core.trajectory import GameTrajectory
from rgi.core.game_runner import GameRunner
from rgi.players.alphazero.alphazero import AlphaZeroPlayer, PolicyValueNetwork
from rgi.players.alphazero.alphazero_tf import PVNetwork_Count21_TF, TFPVNetworkWrapper, train_model


def initialize_model(game: Count21Game) -> TFPVNetworkWrapper:
    """
    Initialize a new TF PV network based on the game state dimensions,
    and return it wrapped to implement PolicyValueNetwork.
    """
    init_state: Count21State = game.initial_state()
    # For Count21, we assume a flat state vector: e.g. [score, current_player].
    state_np: NDArray[np.float32] = np.array([init_state.score, init_state.current_player], dtype=np.float32)
    state_dim: int = state_np.shape[0]
    num_actions: int = len(game.legal_actions(init_state))
    num_players: int = game.num_players(init_state)

    tf_model: PVNetwork_Count21_TF = PVNetwork_Count21_TF(
        state_dim=state_dim, num_actions=num_actions, num_players=num_players
    )
    # Build model via a dummy forward pass
    tf_model(tf.convert_to_tensor(state_np.reshape(1, -1)))
    return TFPVNetworkWrapper(tf_model)


def generate_selfplay_trajectories(
    game: Count21Game,
    pv_network: PolicyValueNetwork[Count21Game, Count21State, int],
    num_simulations: int,
    num_games: int,
    verbose: bool = False,
) -> list[GameTrajectory[Count21State, int]]:
    """
    Generate self-play trajectories using the provided policy–value network.
    """
    trajectories: list[GameTrajectory[Count21State, int]] = []
    num_players: int = game.num_players(game.initial_state())
    # Create an AlphaZeroPlayer for each position.
    players: list[AlphaZeroPlayer[Count21Game, Count21State, int]] = [
        AlphaZeroPlayer(game, pv_network, num_simulations=num_simulations) for _ in range(num_players)
    ]
    for _ in tqdm(range(num_games), desc="Self-play games"):
        runner: GameRunner[Count21State, int, None] = GameRunner(game, players, verbose=verbose)
        traj: GameTrajectory[Count21State, int] = runner.run()
        trajectories.append(traj)
    return trajectories


def evaluate_model(
    game: Count21Game,
    pv_network: PolicyValueNetwork[Count21Game, Count21State, int],
    num_simulations: int,
    num_games: int,
) -> None:
    """
    Simple evaluation: run self-play games with the current PV network and
    compute win percentages.
    """
    trajectories: list[GameTrajectory[Count21State, int]] = generate_selfplay_trajectories(
        game, pv_network, num_simulations, num_games, verbose=False
    )
    num_players: int = game.num_players(game.initial_state())
    win_counts: list[int] = [0] * num_players
    for traj in trajectories:
        # Determine the winner as the index of the highest final reward.
        winner: int = int(np.argmax(traj.final_reward))
        win_counts[winner] += 1
    print("Evaluation results:")
    for i, cnt in enumerate(win_counts):
        print(f"  Player {i+1}: {(cnt/len(trajectories))*100:.1f}% wins")


def main() -> None:
    parser: argparse.ArgumentParser = argparse.ArgumentParser(
        description="Automated RL loop for AlphaZero training on Count21."
    )
    parser.add_argument("--num_iterations", type=int, default=10, help="Number of RL iterations.")
    parser.add_argument("--num_selfplay_games", type=int, default=100, help="Self-play games per iteration.")
    parser.add_argument("--num_simulations", type=int, default=50, help="MCTS simulations per move.")
    parser.add_argument("--num_epochs", type=int, default=10, help="Training epochs per iteration.")
    parser.add_argument("--verbose", action="store_true", help="Print verbose game logs.")
    parser.add_argument("--target", type=int, default=8, help="Target value for Count21Game.")
    parser.add_argument("--max_guess", type=int, default=3, help="Maximum guess value for Count21Game.")
    parser.add_argument("--profile", action="store_true", help="Enable cProfile profiling.")
    args = parser.parse_args()

    if args.profile:
        profiler = cProfile.Profile()
        profiler.enable()

    # Initialize game with user-specified target and max_guess
    game: Count21Game = Count21Game(num_players=2, target=args.target, max_guess=args.max_guess)
    pv_network_wrapper: TFPVNetworkWrapper = initialize_model(game)

    for iteration in range(1, args.num_iterations + 1):
        print(f"\n\n\n=== RL Iteration {iteration} ===")

        # 1. Self-play: generate trajectories.
        trajectories: list[GameTrajectory[Count21State, int]] = generate_selfplay_trajectories(
            game, pv_network_wrapper, args.num_simulations, args.num_selfplay_games, verbose=args.verbose
        )
        print(f"Generated {len(trajectories)} trajectories.")

        # Optionally, print a simple stat (win distribution).
        num_players: int = game.num_players(game.initial_state())
        win_counts: list[int] = [0] * num_players
        first_action_counts: dict[Any, int] = {}
        for traj in trajectories:
            win_counts[int(np.argmax(traj.final_reward))] += 1
            first_action_counts[traj.actions[0]] = first_action_counts.get(traj.actions[0], 0) + 1
        for i, cnt in enumerate(win_counts):
            print(f"  Self-play: Player {i+1} wins {(cnt/len(trajectories))*100:.1f}%")
        for action, cnt in first_action_counts.items():
            print(f"  First action {action}: {cnt/len(trajectories)*100:.1f}%")

        # 2. Train model on generated trajectories.
        # Note: train_model (from train_tf_pv.py) returns a new TFPVNetwork.
        new_tf_model: PVNetwork_Count21_TF = train_model(trajectories, num_epochs=args.num_epochs)
        weight_file: str = f"tf_pv_network_iter{iteration}.weights.h5"
        new_tf_model.save_weights(weight_file)
        print(f"Saved model weights to {weight_file}")

        wrapped_model: TFPVNetworkWrapper = TFPVNetworkWrapper(new_tf_model)
        policy_logits, value = wrapped_model.predict(
            game, game.initial_state(), game.legal_actions(game.initial_state())
        )
        print(f"Policy logits[iter {iteration}]: {policy_logits}")
        softmax_probs = np.exp(policy_logits) / np.sum(np.exp(policy_logits)) * 100.0
        print(f"Policy softmax_pc[iter {iteration}]: [{', '.join(f'{p:.6f}' for p in softmax_probs)}]")
        print(f"Value[iter {iteration}]: {value}")

        # 3. Update the current model in our wrapper.
        pv_network_wrapper.tf_model = new_tf_model

        # 4. Evaluate updated model.
        evaluate_model(game, pv_network_wrapper, args.num_simulations, num_games=20)

    if args.profile:
        profiler.disable()
        # Sort by cumulative time
        stats = pstats.Stats(profiler).sort_stats("cumulative")
        stats.print_stats(50)  # Print top 50 functions by time


if __name__ == "__main__":
    main()

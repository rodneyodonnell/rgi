"""
Run self-play games to generate and archive trajectories.

Usage:
  python -m rgi.players.alphazero.selfplay --num_games 100 --output trajectory.npz --progress
"""

import argparse
from typing import Any
import numpy as np
from tqdm import tqdm  # Optional dependency; install with `pip install tqdm` if needed

from rgi.core.game_runner import GameRunner
from rgi.players.alphazero.alphazero import AlphaZeroPlayer, DummyPolicyValueNetwork
from rgi.games.count21.count21 import Count21Game
from rgi.core.trajectory import GameTrajectory  # For type annotation
from rgi.core.archive import RowFileArchiver


def print_stats(trajectories: list[GameTrajectory[Any, Any]], num_players: int) -> None:
    """
    Compute and print win percentages per player and the frequency distribution of the first moves.
    Assumes that final_reward is a vector where the winning player's entry is the highest value.
    """
    win_counts: list[int] = [0] * num_players
    first_move_counts: dict[Any, int] = {}
    total_games = len(trajectories)

    for traj in trajectories:
        # Determine winner as the index of highest reward.
        # In a well-defined terminal state, one player should have the highest value.
        winner_index = int(np.argmax(traj.final_reward))
        win_counts[winner_index] += 1

        if traj.actions:
            first_move = traj.actions[0]
            first_move_counts[first_move] = first_move_counts.get(first_move, 0) + 1

    print("\nWin percentages by player:")
    for i in range(num_players):
        percentage = (win_counts[i] / total_games) * 100
        print(f"  Player {i+1}: {percentage:.1f}% wins")

    print("\nFirst move frequency distribution:")
    for move, count in first_move_counts.items():
        percentage = (count / total_games) * 100
        print(f"  Move {move}: {percentage:.1f}%")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate self-play data using Count21.")
    parser.add_argument("--num_players", type=int, default=2, help="Number of players.")
    parser.add_argument("--target", type=int, default=21, help="Game target value.")
    parser.add_argument("--max_guess", type=int, default=3, help="Maximum guess per turn.")
    parser.add_argument("--num_games", type=int, default=1, help="Number of games to play.")
    parser.add_argument("--verbose", action="store_true", help="Print game steps.")
    parser.add_argument("--progress", action="store_true", help="Show progress bar.")
    parser.add_argument("--output", type=str, default="trajectories.npz", help="File to save the trajectories.")
    args = parser.parse_args()

    # Create game instance.
    game = Count21Game(num_players=args.num_players, target=args.target, max_guess=args.max_guess)
    # Instantiate dummy network (to be replaced by a trainable network later).
    dummy_network: DummyPolicyValueNetwork = DummyPolicyValueNetwork()
    # Create players. In self-play all players use the same algorithm.
    players = [AlphaZeroPlayer(game, dummy_network) for _ in range(args.num_players)]

    trajectories: list[GameTrajectory[Any, Any]] = []
    games_range = range(args.num_games)
    if args.progress:
        games_range = tqdm(games_range, desc="Self-play games")

    # Play multiple games.
    for _ in games_range:
        runner = GameRunner(game, players, verbose=args.verbose)
        trajectory = runner.run()
        trajectories.append(trajectory)

    # Archive trajectories (replace with actual archive save as needed).
    print(f"\nGenerated {len(trajectories)} trajectories. Saving to {args.output}...")
    # Example: trajectory.save(args.output) if implemented

    # Compute and print statistics.
    print_stats(trajectories, game.num_players(game.initial_state()))

    archive = RowFileArchiver()
    archive.write_items(trajectories, args.output)
    print(f"Wrote {len(trajectories)} trajectories to {args.output}")


if __name__ == "__main__":
    main()

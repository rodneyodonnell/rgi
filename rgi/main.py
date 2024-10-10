"""
RGI Main Script

This script provides a command-line interface for various operations in the RGI project,
including playing games, generating trajectories, and training AI models.

Usage examples:

1. Play a game of Connect4 with a human player against a random AI:
   python rgi/main.py --game connect4 --player1 human --player2 random

2. Generate trajectories by playing 100 games of Connect4 with two random AIs:
   python rgi/main.py --game connect4 --player1 random --player2 random --num_games 100 --save_trajectories

3. Train the ZeroZero model on Connect4 trajectories:
   python rgi/main.py --train --game connect4 --trajectories_dir data/trajectories/connect4 --checkpoint_dir data/checkpoints/zerozero_connect4

4. Play a game of Connect4 using a trained ZeroZero model:
   python rgi/main.py --game connect4 --player1 zerozero --player2 human --checkpoint_dir checkpoints/zerozero_connect4

5. Generate trajectories using a trained ZeroZero model:
   python rgi/main.py --game connect4 --player1 zerozero --player2 random --num_games 100 --save_trajectories --checkpoint_dir data/checkpoints/zerozero_connect4

For more information on available options, run:
python rgi/main.py --help
"""

import argparse
from typing import Literal, Any
from collections import defaultdict
import jax.numpy as jnp
from tqdm import tqdm
import os
from datetime import datetime
from rgi.core.base import Game, Player, TPlayerId
from rgi.core import game_registry
from rgi.core.trajectory import Trajectory, encode_trajectory, save_trajectories, load_trajectories
from rgi.players.zerozero.zerozero_trainer import ZeroZeroTrainer
from rgi.players.zerozero.zerozero_model import ZeroZeroModel
import jax

GAMES: dict[str, game_registry.RegisteredGame[Any, Any, Any]] = game_registry.GAME_REGISTRY
PLAYERS: dict[str, Any] = game_registry.PLAYER_REGISTRY

PlayerType = str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run RGI games or train the ZeroZero model")
    parser.add_argument(
        "--game",
        type=str,
        default="connect4",
        choices=list(GAMES.keys()),
        help="Game to play",
    )
    parser.add_argument(
        "--player1",
        type=str,
        default="human",
        choices=list(PLAYERS.keys()),
        help="Type of player 1",
    )
    parser.add_argument(
        "--player2",
        type=str,
        default="random",
        choices=list(PLAYERS.keys()),
        help="Type of player 2",
    )
    parser.add_argument("--num_games", type=int, default=1, help="Number of games to play")
    parser.add_argument("--save_trajectories", action="store_true", help="Save game trajectories")
    parser.add_argument(
        "--trajectories_dir",
        type=str,
        default="data/trajectories",
        help="Directory to save trajectories",
    )
    parser.add_argument("-v", "--verbose", action="store_true", help="Increase output verbosity")
    parser.add_argument("--train", action="store_true", help="Train the ZeroZero model")
    parser.add_argument("--checkpoint_dir", type=str, default="", help="Directory to save/load model checkpoints")
    return parser.parse_args()


def create_player(
    args: argparse.Namespace,
    player_type: PlayerType,
    game: Game[Any, Any, Any],
    registered_game: game_registry.RegisteredGame[Any, Any, Any],
    player_id: int,
    params: dict | None = None,
) -> Player[Any, Any, Any]:
    player_creator = PLAYERS[player_type]
    return player_creator(args, game, registered_game, player_id, params or {})


def print_aggregate_stats(stats: list[dict[str, list[int] | int]], num_games: int) -> None:
    wins: dict[int | str, int] = defaultdict(int)
    total_moves = sum(stat["moves"] for stat in stats if isinstance(stat["moves"], int))

    for stat in stats:
        winners = stat["winners"]
        if isinstance(winners, list) and len(winners) == 1:
            wins[winners[0]] += 1
        else:
            wins["draw"] += 1

    print(f"\nAggregate Statistics over {num_games} games:")
    print(f"Total moves: {total_moves}")
    print(f"Average moves per game: {total_moves / num_games:.2f}")
    print("Wins:")
    for player, win_count in wins.items():
        if player != "draw":
            print(f"  Player {player}: {win_count} ({win_count/num_games*100:.2f}%)")
    print(f"Draws: {wins['draw']} ({wins['draw']/num_games*100:.2f}%)")


def run_games(
    game: Game[Any, TPlayerId, Any],
    registered_game: game_registry.RegisteredGame[Any, Any, Any],
    players: dict[TPlayerId, Player[Any, Any, Any]],
    num_games: int,
    save_trajectories_path: str | None = None,
    verbose: bool = False,
) -> None:
    stats: list[dict[str, Any]] = []
    all_trajectories = []

    serializer = registered_game.serializer_fn()

    # Create a progress bar
    pbar = tqdm(total=num_games, disable=verbose or num_games == 1)

    for i in range(num_games):
        if verbose or num_games == 1:
            print(f"Starting game {i+1}")

        move_count = 0
        state = game.initial_state()
        trajectory_states = [state]
        trajectory_actions = []
        trajectory_state_rewards = []
        trajectory_player_ids = []

        while not game.is_terminal(state):
            current_player = game.current_player_id(state)
            action = players[current_player].select_action(state, game.legal_actions(state))

            trajectory_actions.append(action)
            trajectory_player_ids.append(current_player)
            trajectory_state_rewards.append(0)  # Assuming no intermediate rewards

            state = game.next_state(state, action)
            trajectory_states.append(state)
            move_count += 1

        if verbose or num_games == 1:
            print(f"Game {i+1} ended")
            print(game.pretty_str(state))

        if game.is_terminal(state):
            all_players_ids = game.all_player_ids(state)
            rewards = [game.reward(state, player_id) for player_id in all_players_ids]
            max_reward = max(rewards)
            winners = [player_id for player_id, reward in zip(all_players_ids, rewards) if reward == max_reward]

            if verbose or num_games == 1:
                print("Final rewards:")
                for player_id, reward in zip(all_players_ids, rewards):
                    print(f"Player {player_id}: {reward}")

                if len(winners) == 1:
                    print(f"Player {winners[0]} wins")
                elif len(winners) == len(all_players_ids):
                    print("The game ended in a draw")
                else:
                    print(f"Players {', '.join(map(str, winners))} tied for the win")
        else:
            if verbose or num_games == 1:
                print("The game ended in an unexpected state")
            winners = []
            rewards = [0] * len(game.all_player_ids(state))

        stats.append({"winners": winners, "moves": move_count})

        # Create and store the trajectory
        trajectory = Trajectory(
            states=trajectory_states,
            actions=trajectory_actions,
            state_rewards=trajectory_state_rewards,
            player_ids=trajectory_player_ids,
            final_rewards=rewards,
        )
        all_trajectories.append(encode_trajectory(game, trajectory, serializer))

        if verbose or num_games == 1:
            print()

        # Update the progress bar
        pbar.update(1)

    # Close the progress bar
    pbar.close()

    if num_games > 1:
        print_aggregate_stats(stats, num_games)

    if save_trajectories_path:
        save_trajectories(all_trajectories, save_trajectories_path)
        print(f"Trajectories saved to {save_trajectories_path}")


def play_game(
    args: argparse.Namespace, registered_game: game_registry.RegisteredGame[Any, Any, Any], game: Game[Any, Any, Any]
) -> None:

    players: dict[int, Player[Any, Any, Any]] = {
        1: create_player(args, args.player1, game, registered_game, player_id=1),
        2: create_player(args, args.player2, game, registered_game, player_id=2),
    }

    if args.save_trajectories:
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        filename = f"{args.game}.n_{args.num_games}.{timestamp}.trajectory.npy"
        save_trajectories_path = os.path.join(args.trajectories_dir, args.game, filename)
        os.makedirs(os.path.dirname(save_trajectories_path), exist_ok=True)
    else:
        save_trajectories_path = None

    run_games(game, registered_game, players, args.num_games, save_trajectories_path, args.verbose)


def train_zerozero_model(
    args: argparse.Namespace, registered_game: game_registry.RegisteredGame[Any, Any, Any], game: Game[Any, Any, Any]
) -> None:

    serializer = registered_game.serializer_fn()
    state_embedder = registered_game.state_embedder_fn()
    action_embedder = registered_game.action_embedder_fn()

    model = ZeroZeroModel(
        state_embedder=state_embedder,
        action_embedder=action_embedder,
        possible_actions=game.all_actions(),
        embedding_dim=64,
        hidden_dim=128,
        shared_dim=256,
    )

    trainer = ZeroZeroTrainer(model, serializer, game)

    # TODO: Does this work? Do we need to do something with the params?
    if args.checkpoint_dir:
        absolute_checkpoint_dir = os.path.abspath(args.checkpoint_dir)
    else:
        absolute_checkpoint_dir = os.path.abspath(os.path.join("data", "checkpoints", args.game))
    trainer.load_checkpoint(absolute_checkpoint_dir)

    trajectories_glob = os.path.join(args.trajectories_dir, args.game, "*.trajectory.npy")
    trajectories = load_trajectories(trajectories_glob)

    if not trajectories:
        raise ValueError(f"No trajectory files found matching pattern: {trajectories_glob}")

    trainer.train(trajectories, num_epochs=10, batch_size=32)

    # Save the trained model
    trainer.save_checkpoint(absolute_checkpoint_dir)
    print(f"Model training completed. Checkpoint saved to {args.checkpoint_dir}")


def main() -> None:
    args = parse_args()

    registered_game = GAMES.get(args.game)
    if registered_game is None:
        print(f"Game {args.game} not implemented yet")
        return

    game = registered_game.game_fn()

    if args.train:
        train_zerozero_model(args, registered_game, game)
    else:
        play_game(args, registered_game, game)


if __name__ == "__main__":
    main()

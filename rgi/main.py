import argparse
from typing import Literal, Any
from collections import defaultdict
from rgi.core.base import Game, Player, TGameState, TPlayerId, TAction
from rgi.core import game_registry
from rgi.players.random_player import RandomPlayer
from rgi.players.minimax_player import MinimaxPlayer
from rgi.players.human_player import HumanPlayer

GAMES: dict[str, type[Game[Any, Any, Any]]] = {name: reg.game_fn for (name, reg) in game_registry.GAME_REGISTRY.items()}


PlayerType = Literal["random", "human", "minimax"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run RGI games")
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
        choices=["random", "human", "minimax"],
        help="Type of player 1",
    )
    parser.add_argument(
        "--player2",
        type=str,
        default="random",
        choices=["random", "human", "minimax"],
        help="Type of player 2",
    )
    parser.add_argument("--num_games", type=int, default=1, help="Number of games to play")
    return parser.parse_args()


def create_player(player_type: PlayerType, game: Game[Any, Any, Any], player_id: int) -> Player[Any, Any, Any]:
    if player_type == "random":
        return RandomPlayer()
    elif player_type == "minimax":
        return MinimaxPlayer(game, player_id)
    elif player_type == "human":
        return HumanPlayer(game)
    else:
        raise ValueError(f"Unknown player type: {player_type}")


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
    game: Game[TGameState, TPlayerId, TAction],
    players: dict[TPlayerId, Player[Any, Any, Any]],
    num_games: int,
) -> None:
    stats: list[dict[str, Any]] = []

    for i in range(num_games):
        print(f"Starting game {i+1}")

        move_count = 0
        state = game.initial_state()
        while not game.is_terminal(state):
            current_player = game.current_player_id(state)
            action = players[current_player].select_action(state, game.legal_actions(state))
            state = game.next_state(state, action)
            move_count += 1

        print(f"Game {i+1} ended")
        print(game.pretty_str(state))

        if game.is_terminal(state):
            all_players_ids = game.all_player_ids(state)
            rewards = [game.reward(state, player_id) for player_id in all_players_ids]
            max_reward = max(rewards)
            winners = [player_id for player_id, reward in zip(all_players_ids, rewards) if reward == max_reward]

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
            print("The game ended in an unexpected state")
            winners = []

        stats.append({"winners": winners, "moves": move_count})

        print()

    if num_games > 1:
        print_aggregate_stats(stats, num_games)


def main() -> None:
    args = parse_args()

    game_class = GAMES.get(args.game)
    if game_class is None:
        print(f"Game {args.game} not implemented yet")
        return

    game = game_class()
    players: dict[int, Player[Any, Any, Any]] = {
        1: create_player(args.player1, game, player_id=1),
        2: create_player(args.player2, game, player_id=2),
    }

    run_games(game, players, args.num_games)


if __name__ == "__main__":
    main()

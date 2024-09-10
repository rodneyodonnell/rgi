import argparse
from typing import Type
from rgi.core.base import Game
from rgi.core.game_runner import GameRunner
from rgi.games.connect4 import Connect4Game
from rgi.players.random_player import RandomPlayer
from rgi.players.human_player import HumanPlayer

GAMES = {
    "connect4": Connect4Game
}

def parse_args():
    parser = argparse.ArgumentParser(description="Run RGI games")
    parser.add_argument("--game", type=str, default="connect4", choices=list(GAMES.keys()), help="Game to play")
    parser.add_argument("--player1", type=str, default="human", choices=["random", "human"], help="Type of player 1")
    parser.add_argument("--player2", type=str, default="random", choices=["random", "human"], help="Type of player 2")
    parser.add_argument("--num_games", type=int, default=1, help="Number of games to play")
    return parser.parse_args()

def create_player(player_type: str, game: Game):
    if player_type == "random":
        return RandomPlayer()
    elif player_type == "human":
        return HumanPlayer(game, lambda actions: actions[int(input("Enter the index of your chosen action: "))])
    else:
        raise ValueError(f"Unknown player type: {player_type}")

def run_games(game: Game, players: dict, num_games: int):
    runner = GameRunner(game, players)
    
    for i in range(num_games):
        print(f"Starting game {i+1}")
        final_state = runner.run_game()
        print(f"Game {i+1} ended")
        print(game.pretty_str(final_state))
        
        if game.is_terminal(final_state):
            all_players_ids = game.all_player_ids(final_state)
            rewards = [game.reward(final_state, player_id) for player_id in all_players_ids]
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
        
        print()

def main():
    args = parse_args()
    
    game_class = GAMES.get(args.game)
    if game_class is None:
        print(f"Game {args.game} not implemented yet")
        return

    game = game_class()
    players = {
        1: create_player(args.player1, game),
        2: create_player(args.player2, game)
    }
    
    run_games(game, players, args.num_games)

if __name__ == "__main__":
    main()
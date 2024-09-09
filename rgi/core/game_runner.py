from typing import Any
from rgi.core.base import Game, TState, TAction, TPlayer
from rgi.core.player import Player

class GameRunner:
    def __init__(self, game: Game[TState, TAction, TPlayer], players: list[Player[TState, TAction, Any]]):
        self.game = game
        self.players = players

    def run_game(self) -> list[float]:
        state = self.game.initial_state()
        
        while not self.game.is_terminal(state):
            current_player = self.game.current_player(state, self.players)
            
            action = current_player.select_action(self.game, state)
            new_state = self.game.next_state(state, action)
            
            for player in self.players:
                player.update(state, action, new_state)
            
            state = new_state
            
            print(f"\nPlayer {self.players.index(current_player)} took action: {self.game.action_to_string(action)}")
            print(f"New game state:\n{self.game.state_to_string(state)}")

        return [self.game.reward(state, player) for player in self.players]

def run_match(game: Game[TState, TAction, TPlayer], 
              players: list[Player[TState, TAction, Any]], 
              num_games: int) -> list[float]:
    runner = GameRunner(game, players)
    total_rewards = [0.0] * len(players)
    
    for i in range(num_games):
        print(f"\n--- Starting Game {i+1} ---")
        game_rewards = runner.run_game()
        
        for j, reward in enumerate(game_rewards):
            total_rewards[j] += reward
        
        print(f"Game {i+1} finished. Rewards: {game_rewards}")

    return [reward / num_games for reward in total_rewards]
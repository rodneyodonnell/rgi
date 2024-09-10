from rgi.core.base import Game, Player, GameObserver, TGameState, TPlayerId, TAction
from typing import Any

class GameRunner:
    def __init__(self, game: Game[TGameState, TPlayerId, TAction], players: dict[TPlayerId, Player[TGameState, Any, TAction]], observer: GameObserver[TGameState, TPlayerId] | None = None):
        self.game = game
        self.players = players
        self.observer = observer

    def run_game(self) -> TGameState:
        state = self.game.initial_state()
        
        if self.observer:
            self.observer.observe_initial_state(state)

        while not self.game.is_terminal(state):
            current_player_id = self.game.current_player_id(state)
            current_player = self.players[current_player_id]
            
            legal_actions = self.game.legal_actions(state)
            action = current_player.select_action(state, legal_actions)
            
            if self.observer:
                self.observer.observe_action(state, current_player_id, action)
            
            new_state = self.game.next_state(state, action)
            
            if self.observer:
                self.observer.observe_state_transition(state, new_state)
            
            for player in self.players.values():
                player.update_state(new_state, action)
            
            state = new_state

        if self.observer:
            self.observer.observe_game_end(state)

        return state

    def run_games(self, num_games: int) -> list[TGameState]:
        return [self.run_game() for _ in range(num_games)]
import random
from typing import Any
from rgi.core.player import Player
from rgi.core.game import Game, TState, TAction, TPlayer

class RandomPlayer(Player[TState, TAction, None]):
    def select_action(self, game: Game[TState, TAction, TPlayer], state: TState) -> TAction:
        return random.choice(game.legal_actions(state))

    def update(self, state: TState, action: TAction, new_state: TState) -> None:
        pass  # RandomPlayer doesn't need to update internal state

    def get_state(self) -> None:
        return None  # RandomPlayer has no internal state

    def set_state(self, state: None) -> None:
        pass  # RandomPlayer has no internal state to set
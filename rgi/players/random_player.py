import random
from typing import Literal
from typing_extensions import override
from rgi.core.base import Player, TGameState, TAction

TPlayerState = Literal[None]


class RandomPlayer(Player[TGameState, TPlayerState, TAction]):
    @override
    def select_action(self, game_state: TGameState, legal_actions: list[TAction]) -> TAction:
        return random.choice(legal_actions)

    @override
    def update_state(self, game_state: TGameState, action: TAction) -> None:
        # Random player doesn't need to maintain any state
        del game_state, action  # Unused variables

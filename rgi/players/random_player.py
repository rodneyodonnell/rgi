from rgi.core.base import Player, TGameState, TAction
from typing_extensions import override
import random


class RandomPlayer(Player[TGameState, None, TAction]):
    @override
    def select_action(self, game_state: TGameState, legal_actions: list[TAction]) -> TAction:
        return random.choice(legal_actions)

    @override
    def update_state(self, game_state: TGameState, action: TAction):
        # Random player doesn't need to maintain any state
        pass

import random
from typing import Literal, Sequence
from typing_extensions import override
from rgi.core.base import Player, TGameState, TAction

TPlayerState = Literal[None]


class RandomPlayer(Player[TGameState, TPlayerState, TAction]):
    def __init__(self, seed: int | None = None) -> None:
        self.rng = random.Random(seed)

    @override
    def select_action(self, game_state: TGameState, legal_actions: Sequence[TAction]) -> TAction:
        return self.rng.choice(legal_actions)

    @override
    def update_state(self, game_state: TGameState, action: TAction) -> None:
        # Random player doesn't need to maintain any state
        del game_state, action  # Unused variables

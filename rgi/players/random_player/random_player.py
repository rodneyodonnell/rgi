import random
from typing import Literal, Sequence

from typing_extensions import override

from rgi.core.base import ActionResult, Player, TAction, TGameState

TPlayerState = Literal[None]
TPlayerData = Literal[None]


class RandomPlayer(Player[TGameState, TPlayerState, TAction, TPlayerData]):
    def __init__(self, seed: int | None = None) -> None:
        self.rng = random.Random(seed)

    @override
    def select_action(
        self, game_state: TGameState, legal_actions: Sequence[TAction]
    ) -> ActionResult[TAction, TPlayerData]:
        return ActionResult(self.rng.choice(legal_actions), None)

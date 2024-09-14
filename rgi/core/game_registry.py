# rgi/core/game_registry.py

from dataclasses import dataclass
from typing import Type
from rgi.core.base import Game, GameSerializer
from rgi.games import connect4
from rgi.games import othello

# Import additional games here


@dataclass
class RegisteredGame:
    game_fn: type[Game]
    serializer_fn: type[GameSerializer]


GAME_REGISTRY: dict[str, Type[Game]] = {
    "connect4": RegisteredGame(connect4.Connect4Game, connect4.Connect4Serializer),
    "othello": RegisteredGame(othello.OthelloGame, othello.OthelloSerializer),
    # Add new games here
}

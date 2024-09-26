# rgi/core/game_registry.py

from dataclasses import dataclass
from typing import Generic, Any

from rgi.core.base import Game, GameSerializer, TGameState, TAction, TPlayerId
from rgi.games import connect4
from rgi.games import othello


@dataclass
class RegisteredGame(Generic[TGameState, TPlayerId, TAction]):
    game_fn: type[Game[TGameState, TPlayerId, TAction]]
    # Ideally this type would be stricter, but this gets a bit tricky.
    serializer_fn: type[GameSerializer[Any, TGameState, TAction]]


GAME_REGISTRY: dict[str, RegisteredGame[Any, Any, Any]] = {
    "connect4": RegisteredGame(connect4.Connect4Game, connect4.Connect4Serializer),
    "othello": RegisteredGame(othello.OthelloGame, othello.OthelloSerializer),
    # Add new games here
}

# rgi/core/game_registry.py

import argparse
from dataclasses import dataclass
from typing import Any, Callable, Generic

from rgi.core.base import Game, GameSerializer, Player, TAction, TGameState
from rgi.games.connect4 import connect4
from rgi.games.count21 import count21
from rgi.games.othello import othello
from rgi.players.human_player.human_player import HumanPlayer
from rgi.players.minimax_player.minimax_player import MinimaxPlayer
from rgi.players.random_player.random_player import RandomPlayer


@dataclass
class RegisteredGame(Generic[TGameState, TAction]):
    game_fn: type[Game[TGameState, TAction]]
    serializer_fn: type[GameSerializer[Any, TGameState, TAction]]


# TODO: We should auto-discover games instead of hardcoding them here.
GAME_REGISTRY: dict[str, RegisteredGame[Any, Any]] = {
    "connect4": RegisteredGame(connect4.Connect4Game, connect4.Connect4Serializer),
    "othello": RegisteredGame(othello.OthelloGame, othello.OthelloSerializer),
    "count21": RegisteredGame(count21.Count21Game, count21.Count21Serializer),
    # Add new games here
}


PLAYER_REGISTRY: dict[
    str,
    Callable[
        [argparse.Namespace | None, Game[Any, Any], RegisteredGame[Any, Any], int, dict[str, Any]],
        Player[Any, Any, Any],
    ],
] = {
    "human": lambda args, game, registered_game, player_id, params: HumanPlayer(game),
    "random": lambda args, game, registered_game, player_id, params: RandomPlayer(),
    "minimax": lambda args, game, registered_game, player_id, params: MinimaxPlayer(game, player_id),
}

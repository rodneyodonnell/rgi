# rgi/core/game_registry.py

from dataclasses import dataclass
from typing import Generic, Any, Optional, Callable

import jax
from rgi.core.base import Game, GameSerializer, TGameState, TPlayerState, TAction, TPlayerId, Player
from rgi.games import connect4
from rgi.games import infiltr8
from rgi.games import othello
from rgi.players.zerozero.zerozero_player import ZeroZeroPlayer
from rgi.players.minimax_player import MinimaxPlayer
from rgi.players.random_player import RandomPlayer
from rgi.players.human_player import HumanPlayer
from rgi.players.zerozero.zerozero_model import StateEmbedder, ActionEmbedder, ZeroZeroModel


@dataclass
class RegisteredGame(Generic[TGameState, TPlayerId, TAction]):
    game_fn: type[Game[TGameState, TPlayerId, TAction]]
    # Ideally this type would be stricter, but this gets a bit tricky.
    serializer_fn: type[GameSerializer[Any, TGameState, TAction]]
    state_embedder_fn: type[StateEmbedder[TGameState]] | None = None
    action_embedder_fn: type[ActionEmbedder[TAction]] | None = None


GAME_REGISTRY: dict[str, RegisteredGame[Any, Any, Any]] = {
    "connect4": RegisteredGame(
        connect4.Connect4Game,
        connect4.Connect4Serializer,
        connect4.Connect4StateEmbedder,
        connect4.Connect4ActionEmbedder,
    ),
    "othello": RegisteredGame(othello.OthelloGame, othello.OthelloSerializer),
    "infiltr8": RegisteredGame(infiltr8.Infiltr8Game, infiltr8.Infiltr8Serializer),
    # Add new games here
}


def load_zerozero_player(game: Game, registered_game: RegisteredGame) -> "ZeroZeroPlayer[Any, Any, Any]":

    rg = registered_game
    assert rg.state_embedder_fn is not None
    state_embedder = rg.state_embedder_fn(embedding_dim=64)

    assert rg.action_embedder_fn is not None
    action_embedder = rg.action_embedder_fn(embedding_dim=64)

    all_actions = game.all_actions()
    assert all_actions is not None

    zerozero_model = ZeroZeroModel(state_embedder, action_embedder, all_actions)
    key = jax.random.PRNGKey(0)
    dummy_state = game.initial_state()
    dummy_action = all_actions[0]

    # Initialize random initial model until we have one to load...
    zerozero_model_params = zerozero_model.init(key, dummy_state, dummy_action)

    return ZeroZeroPlayer(zerozero_model, zerozero_model_params)  # type: ignore


PLAYER_REGISTRY: dict[
    str, Callable[[Game[Any, Any, Any], RegisteredGame[Any, Any, Any], int, dict[str, Any]], Player[Any, Any, Any]]
] = {
    "human": lambda game, registered_game, player_id, params: HumanPlayer(game, **params),
    "random": lambda game, registered_game, player_id, params: RandomPlayer(**params),
    "minimax": lambda game, registered_game, player_id, params: MinimaxPlayer(game, player_id, **params),
    "zerozero": lambda game, registered_game, player_id, params: load_zerozero_player(game, registered_game, **params),
}

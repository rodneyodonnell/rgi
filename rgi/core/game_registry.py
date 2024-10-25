# rgi/core/game_registry.py

from dataclasses import dataclass
from typing import Generic, Any, Optional, Callable

from flax.training import train_state, checkpoints
import optax
import jax.numpy as jnp
import jax
import argparse
from rgi.core.base import (
    Game,
    GameSerializer,
    TGameState,
    TPlayerState,
    TAction,
    TPlayerId,
    Player,
)
from rgi.games import connect4
from rgi.games import infiltr8
from rgi.games import othello
from rgi.players.zerozero.zerozero_player import ZeroZeroPlayer
from rgi.players.minimax_player import MinimaxPlayer
from rgi.players.random_player import RandomPlayer
from rgi.players.human_player import HumanPlayer
from rgi.players.zerozero.zerozero_model import (
    StateEmbedder,
    ActionEmbedder,
    ZeroZeroModel,
)
import os
from flax.training import checkpoints


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


def load_zerozero_player(
    args: argparse.Namespace,
    game: Game,
    registered_game: RegisteredGame,
    player_id: int,
) -> "ZeroZeroPlayer[Any, Any, Any]":
    rg = registered_game
    assert rg.state_embedder_fn is not None
    state_embedder = rg.state_embedder_fn(embedding_dim=64)

    assert rg.action_embedder_fn is not None
    action_embedder = rg.action_embedder_fn(embedding_dim=64)

    all_actions = game.all_actions()
    assert all_actions is not None

    serializer = rg.serializer_fn()
    dummy_state = serializer.state_to_jax_array(game, game.initial_state())
    dummy_action = serializer.action_to_jax_array(game, game.all_actions()[0])
    # Add batch dimension to dummy inputs
    dummy_state_batch = jnp.expand_dims(dummy_state, axis=0)
    dummy_action_batch = jnp.expand_dims(dummy_action, axis=0)

    zerozero_model = ZeroZeroModel(state_embedder, action_embedder, all_actions)
    key = jax.random.PRNGKey(0)

    # Initialize model parameters
    zerozero_model_params = zerozero_model.init(
        key, dummy_state_batch, dummy_action_batch
    )
    old_params = zerozero_model_params

    # Load checkpoint if it exists
    if args.checkpoint_dir:
        absolute_checkpoint_dir = os.path.abspath(args.checkpoint_dir)
    else:
        absolute_checkpoint_dir = os.path.abspath(
            os.path.join("data", "checkpoints", args.game)
        )

    zerozero_model_params = checkpoints.restore_checkpoint(
        absolute_checkpoint_dir, target=zerozero_model_params
    )
    new_params = zerozero_model_params

    loaded_params = checkpoints.restore_checkpoint(absolute_checkpoint_dir, target=None)

    dummy_tx = optax.adam(learning_rate=0.0001)  # Any trainer will do. Just use
    _train_state = train_state.TrainState.create(
        apply_fn=zerozero_model.apply, params=zerozero_model_params, tx=dummy_tx
    )

    _train_state_2 = train_state.TrainState.create(
        apply_fn=zerozero_model.apply, params=old_params, tx=dummy_tx
    )
    latest_checkpoint = checkpoints.latest_checkpoint(absolute_checkpoint_dir)
    _train_state_2_old = _train_state_2
    if latest_checkpoint is not None:
        _train_state_2 = checkpoints.restore_checkpoint(
            latest_checkpoint, _train_state_2
        )
    else:
        print("No checkpoint found, starting from scratch.")
    _train_state_2_new = _train_state_2

    print(
        f'default params: {_train_state_2_old.params["params"]["action_embedder"]["action_embeddings"][0][:5]}'
    )
    print(
        f'update params : {_train_state_2_new.params["params"]["action_embedder"]["action_embeddings"][0][:5]}'
    )
    # Load the checkpoint into the TrainState
    # state = checkpoints.restore_checkpoint(absolute_checkpoint_dir, target=state)

    # Extract the parameters
    zerozero_model_params = _train_state_2_new.params

    print(
        f"Loaded checkpoint from {absolute_checkpoint_dir}, was loaded: {old_params is not new_params} updated: {old_params != new_params}"
    )

    for k1 in set(old_params) | set(new_params):
        for k2 in set(old_params[k1]) | set(new_params[k1]):
            for k3 in set(old_params[k1][k2]) | set(new_params[k1][k2]):
                print(
                    f"Checking[{k1}][{k2}][{k3}] -> same = {old_params[k1][k2][k3] is new_params[k1][k2][k3]}, equal = {old_params[k1][k2][k3] ==  new_params[k1][k2][k3]}"
                )

    return ZeroZeroPlayer(zerozero_model, zerozero_model_params, game, serializer)  # type: ignore


PLAYER_REGISTRY: dict[
    str,
    Callable[
        [
            argparse.Namespace,
            Game[Any, Any, Any],
            RegisteredGame[Any, Any, Any],
            int,
            dict[str, Any],
        ],
        Player[Any, Any, Any],
    ],
] = {
    "human": lambda args, game, registered_game, player_id, params: HumanPlayer(game),
    "random": lambda args, game, registered_game, player_id, params: RandomPlayer(),
    "minimax": lambda args, game, registered_game, player_id, params: MinimaxPlayer(
        game, player_id
    ),
    "zerozero": lambda args, game, registered_game, player_id, params: load_zerozero_player(
        args, game, registered_game, player_id
    ),
}

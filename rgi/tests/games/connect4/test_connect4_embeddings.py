import jax
import jax.numpy as jnp
import pytest
from rgi.games.connect4 import (
    Connect4CNN,
    Connect4StateEmbedder,
    Connect4ActionEmbedder,
)
from rgi.games.connect4 import Connect4Game

# pylint: disable=redefined-outer-name  # pytest fixtures trigger this false positive


@pytest.fixture
def game() -> Connect4Game:
    return Connect4Game()


def test_connect4_state_embedder(game: Connect4Game) -> None:
    cnn_model = Connect4CNN()
    embedder = Connect4StateEmbedder(cnn_model)
    state = game.initial_state()
    params = {"cnn_model": cnn_model.init(jax.random.PRNGKey(0), jnp.zeros((1, 6, 7, 1)))}

    embedding = embedder.embed_state(params, state)
    assert embedding.shape == (64,)


def test_connect4_action_embedder() -> None:
    embedder = Connect4ActionEmbedder()
    params = embedder.init(jax.random.PRNGKey(0), jnp.array(0))

    for action in range(1, 8):
        embedding = embedder.embed_action(params, action)
        assert embedding.shape == (64,)


def test_invalid_inputs() -> None:
    cnn_model = Connect4CNN()
    state_embedder = Connect4StateEmbedder(cnn_model)
    action_embedder = Connect4ActionEmbedder()
    action_params = action_embedder.init(jax.random.PRNGKey(0), jnp.array(0))

    with pytest.raises(ValueError):
        state_embedder.embed_state({}, "not a Connect4State")  # type: ignore

    with pytest.raises(ValueError):
        action_embedder.embed_action(action_params, 0)

    with pytest.raises(ValueError):
        action_embedder.embed_action(action_params, 8)

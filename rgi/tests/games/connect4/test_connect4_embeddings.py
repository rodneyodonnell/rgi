from typing import Any
import jax
import jax.numpy as jnp
import pytest
from flax.typing import FrozenVariableDict
from rgi.games.connect4.connect4_embeddings import Connect4StateEmbedder, Connect4ActionEmbedder
from rgi.games.connect4 import Connect4Game

# pylint: disable=redefined-outer-name  # pytest fixtures trigger this false positive

TJaxParams = dict[str, Any]


@pytest.fixture
def game() -> Connect4Game:
    return Connect4Game()


def test_connect4_state_embedder(game: Connect4Game) -> None:
    init_state = game.initial_state()

    state_embedder = Connect4StateEmbedder()
    params = state_embedder.init(jax.random.PRNGKey(0), init_state)

    state = game.next_state(init_state, 1)
    embedding = state_embedder.apply(params, state)
    assert isinstance(embedding, jax.Array)
    assert embedding.shape == (64,)


def test_connect4_action_embedder() -> None:
    init_action = 1

    action_embedder = Connect4ActionEmbedder()
    params = action_embedder.init(jax.random.PRNGKey(0), init_action)

    for action in range(1, 8):
        embedding = action_embedder.apply(params, action)
        assert isinstance(embedding, jax.Array)
        assert embedding.shape == (64,)


def test_invalid_inputs(game: Connect4Game) -> None:
    init_state = game.initial_state()
    init_action = 1

    state_embedder = Connect4StateEmbedder()
    state_params = state_embedder.init(jax.random.PRNGKey(0), init_state)

    action_embedder = Connect4ActionEmbedder()
    action_params = action_embedder.init(jax.random.PRNGKey(0), init_action)

    with pytest.raises(ValueError):
        state_embedder.apply(state_params, "not a Connect4State")  # type: ignore

    with pytest.raises(ValueError):
        action_embedder.apply(action_params, 0)

    with pytest.raises(ValueError):
        action_embedder.apply(action_params, 8)

from typing import Any, cast

import jax
import jax.numpy as jnp
import pytest
from pytest_mock import MockerFixture

from rgi.players.zerozero.zerozero_player import ZeroZeroPlayer
from rgi.players.zerozero.zerozero_model import ZeroZeroModel
from rgi.core.base import Game
from rgi.tests.players.test_zerozero_model import DummyStateEmbedder, DummyActionEmbedder

# pylint: disable=redefined-outer-name  # pytest fixtures trigger this false positive

TAction = int
TGameState = tuple[int, int]

GAME_STATE = (0, 0)


@pytest.fixture
def model() -> ZeroZeroModel[Any, Any, TAction]:
    return ZeroZeroModel(
        state_embedder=DummyStateEmbedder(),
        action_embedder=DummyActionEmbedder(),
        possible_actions=[0, 1, 2, 3, 4, 5, 6],
        embedding_dim=64,
        hidden_dim=128,
    )


@pytest.fixture
def params(model: ZeroZeroModel[Any, Any, TAction]) -> dict[str, Any]:
    key = jax.random.PRNGKey(0)
    dummy_state: TGameState = (1, 5)
    dummy_action: TAction = 2
    init_params = model.init(key, dummy_state, dummy_action)
    return dict(init_params)


@pytest.fixture
def player(model: ZeroZeroModel[Any, Any, int], params: dict) -> ZeroZeroPlayer[Any, Any, int]:
    return ZeroZeroPlayer(model, params, temperature=1.0, rng_key=jax.random.PRNGKey(0))


def test_select_action_with_all_legal_actions(player: ZeroZeroPlayer[Any, Any, int]):
    action = player.select_action(GAME_STATE, legal_actions=[1, 2, 3, 4, 5, 6, 7])
    assert action in [1, 2, 3, 4, 5, 6, 7]


def test_select_action_with_some_illegal_actions(player: ZeroZeroPlayer[Any, Any, int]):
    action = player.select_action(GAME_STATE, [1, 2, 4, 5, 6, 7])  # 3 is illegal
    assert action in [1, 2, 4, 5, 6, 7]
    assert action != 3


def test_update_state(player: ZeroZeroPlayer[Any, Any, int]):
    # Currently, update_state does nothing, so we just check it doesn't raise an error
    player.update_state(None, 1)


@pytest.mark.parametrize(
    "legal_actions",
    [
        [1, 2, 3],
        [4, 5, 6, 7],
        [1, 7],
    ],
)
def test_select_action_respects_legal_actions(player: ZeroZeroPlayer[Any, Any, int], legal_actions: list[int]):
    action = player.select_action(GAME_STATE, legal_actions)
    assert action in legal_actions


def test_select_action_with_single_legal_action(player: ZeroZeroPlayer[Any, Any, int]):
    action = player.select_action(GAME_STATE, [4])
    assert action == 4

from typing import Any, cast

import jax
import jax.numpy as jnp
import pytest
from pytest_mock import MockerFixture

from rgi.players.zerozero.zerozero_player import ZeroZeroPlayer
from rgi.players.zerozero.zerozero_model import ZeroZeroModel
from rgi.core.base import Game

# pylint: disable=redefined-outer-name  # pytest fixtures trigger this false positive


@pytest.fixture
def mock_zerozero_model(mocker: MockerFixture) -> ZeroZeroModel[Any, Any, int]:
    mock_model = mocker.Mock(spec=ZeroZeroModel)
    mock_model.possible_actions = [1, 2, 3, 4, 5, 6, 7]
    probabilities = jnp.array([0.1, 0.2, 0.3, 0.1, 0.1, 0.1, 0.1])
    mock_model.apply.return_value = (None, None, jnp.log(probabilities))
    return cast(ZeroZeroModel[Any, Any, int], mock_model)


@pytest.fixture
def player(mock_zerozero_model: ZeroZeroModel[Any, Any, int]) -> ZeroZeroPlayer[Any, Any, int]:
    return ZeroZeroPlayer(mock_zerozero_model, params={}, temperature=1.0, rng_key=jax.random.PRNGKey(0))


def test_select_action_with_all_legal_actions(player: ZeroZeroPlayer[Any, Any, int]):
    action = player.select_action(None, [1, 2, 3, 4, 5, 6, 7])
    assert action in [1, 2, 3, 4, 5, 6, 7]


def test_select_action_with_some_illegal_actions(player: ZeroZeroPlayer[Any, Any, int]):
    action = player.select_action(None, [1, 2, 4, 5, 6, 7])  # 3 is illegal
    assert action in [1, 2, 4, 5, 6, 7]
    assert action != 3


def test_temperature_effect(mock_zerozero_model: ZeroZeroModel[Any, Any, int]):
    cold_player = ZeroZeroPlayer(mock_zerozero_model, params={}, temperature=0.1)
    hot_player = ZeroZeroPlayer(mock_zerozero_model, params={}, temperature=10.0)

    # With low temperature, should almost always choose the highest probability action
    cold_actions = [cold_player.select_action(None, [1, 2, 3, 4, 5, 6, 7]) for _ in range(100)]
    assert cold_actions.count(3) > 90  # Action 3 has the highest probability

    # With high temperature, should choose more randomly
    hot_actions = [hot_player.select_action(None, [1, 2, 3, 4, 5, 6, 7]) for _ in range(100)]
    assert len(set(hot_actions)) > 1  # Should choose multiple different actions


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
    action = player.select_action(None, legal_actions)
    assert action in legal_actions


def test_select_action_with_single_legal_action(player: ZeroZeroPlayer[Any, Any, int]):
    action = player.select_action(None, [4])
    assert action == 4


def test_select_action_randomness(mock_zerozero_model: ZeroZeroModel[Any, Any, int]):
    player = ZeroZeroPlayer(mock_zerozero_model, params={}, temperature=1.0, rng_key=jax.random.PRNGKey(0))
    actions = [player.select_action(None, [1, 2, 3, 4, 5, 6, 7]) for _ in range(1000)]

    # Check that we're getting different actions
    assert len(set(actions)) > 1, "Expected multiple different actions to be selected"

    # Check that the distribution roughly matches the expected probabilities
    action_counts = {a: actions.count(a) for a in set(actions)}
    assert (
        action_counts[3] > action_counts[2] > action_counts[1]
    ), "Expected action 3 to be chosen most often, then 2, then 1"


def test_reproducibility(mock_zerozero_model: ZeroZeroModel[Any, Any, int]):
    player1 = ZeroZeroPlayer(mock_zerozero_model, params={}, temperature=1.0, rng_key=jax.random.PRNGKey(42))
    player2 = ZeroZeroPlayer(mock_zerozero_model, params={}, temperature=1.0, rng_key=jax.random.PRNGKey(42))

    actions1 = [player1.select_action(None, [1, 2, 3, 4, 5, 6, 7]) for _ in range(10)]
    actions2 = [player2.select_action(None, [1, 2, 3, 4, 5, 6, 7]) for _ in range(10)]

    assert actions1 == actions2, "Expected the same sequence of actions with the same initial RNG key"

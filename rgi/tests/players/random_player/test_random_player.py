from typing import Any, cast

from pytest_mock import MockerFixture
import pytest
from rgi.players.random_player.random_player import RandomPlayer
from rgi.core.base import Game

# pylint: disable=redefined-outer-name  # pytest fixtures trigger this false positive


@pytest.fixture
def game(mocker: MockerFixture) -> Game[Any, int]:
    mock_game = mocker.Mock(spec=Game)
    mock_game.legal_actions.return_value = [1, 2, 7]
    mock_game.pretty_str.return_value = "Mock pretty state"
    return cast(Game[Any, int], mock_game)


@pytest.fixture
def player() -> RandomPlayer[Any, Any]:
    return RandomPlayer()


def test_select_action(game: Game[Any, int], player: RandomPlayer[Any, Any]) -> None:
    state = None  # RandomPlayer doesn't use the state
    actions = game.legal_actions(state)

    # Run multiple times to ensure we're getting different actions
    selected_actions = set(player.select_action(state, actions) for _ in range(50))

    assert len(selected_actions) > 1, "RandomPlayer should select different actions over multiple calls"
    assert all(action in actions for action in selected_actions), "All selected actions should be legal"


def test_update_state(player: RandomPlayer[Any, Any]) -> None:
    # RandomPlayer's update_state should do nothing
    player.update_player_state(None, 0, 0)
    # If we reach here without error, the test passes

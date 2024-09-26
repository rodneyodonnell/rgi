from typing import Any, cast

from unittest.mock import patch
import pytest
from pytest_mock import MockerFixture

from rgi.players.human_player import HumanPlayer
from rgi.core.base import Game

# pylint: disable=redefined-outer-name  # pytest fixtures trigger this false positive


@pytest.fixture
def game(mocker: MockerFixture) -> Game[Any, Any, int]:
    mock_game = mocker.Mock(spec=Game)
    mock_game.legal_actions.return_value = [1, 2, 3]
    mock_game.pretty_str.return_value = "Mock pretty state"
    return cast(Game[Any, Any, int], mock_game)


@pytest.fixture
def player(game: Game[Any, Any, int]) -> HumanPlayer[Game[Any, Any, int], Any, int]:
    return HumanPlayer(game)


@pytest.mark.parametrize(
    "user_input, expected_action",
    [
        ("1", 1),
        ("2", 2),
        ("3", 3),
        ("i:1", 1),
        ("i:2", 2),
        ("i:3", 3),
    ],
)
def test_select_action(
    game: Game[Any, Any, int], player: HumanPlayer[Game[Any, Any, int], Any, int], user_input: str, expected_action: int
) -> None:
    with patch("builtins.input", return_value=user_input):
        action = player.select_action(None, game.legal_actions(None))
        assert action == expected_action


def test_invalid_input(game: Game[Any, Any, int], player: HumanPlayer[Game[Any, Any, int], Any, int]) -> None:
    with patch("builtins.input", side_effect=["invalid", "1"]):
        action = player.select_action(None, game.legal_actions(None))
        assert action == 1


def test_update_state(player: HumanPlayer[Game[Any, Any, int], Any, int]) -> None:
    # HumanPlayer's update_state should do nothing
    player.update_state(None, 0)
    # If we reach here without error, the test passes

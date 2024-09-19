import pytest
from unittest.mock import patch
from rgi.players.human_player import HumanPlayer
from rgi.core.base import Game


@pytest.fixture
def game(mocker):
    game = mocker.Mock(spec=Game)
    game.legal_actions.return_value = [1, 2, 3]
    game.pretty_str.return_value = "Mock pretty state"
    return game


@pytest.fixture
def player(game):
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
def test_select_action(game, player, user_input, expected_action):
    with patch("builtins.input", return_value=user_input):
        action = player.select_action(None, game.legal_actions(None))
        assert action == expected_action


def test_invalid_input(game, player):
    with patch("builtins.input", side_effect=["invalid", "1"]):
        action = player.select_action(None, game.legal_actions(None))
        assert action == 1


def test_update_state(player):
    # HumanPlayer's update_state should do nothing
    player.update_state(None, None)
    # If we reach here without error, the test passes

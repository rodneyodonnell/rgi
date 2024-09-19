import pytest
from rgi.players.random_player import RandomPlayer
from rgi.core.base import Game


@pytest.fixture
def game(mocker):
    game = mocker.Mock(spec=Game)
    game.legal_actions.return_value = [1, 2, 3]
    return game


@pytest.fixture
def player():
    return RandomPlayer()


def test_select_action(game, player):
    state = None  # RandomPlayer doesn't use the state
    actions = game.legal_actions(state)

    # Run multiple times to ensure we're getting different actions
    selected_actions = set(player.select_action(state, actions) for _ in range(50))

    assert len(selected_actions) > 1, "RandomPlayer should select different actions over multiple calls"
    assert all(action in actions for action in selected_actions), "All selected actions should be legal"


def test_update_state(player):
    # RandomPlayer's update_state should do nothing
    player.update_state(None, None)
    # If we reach here without error, the test passes

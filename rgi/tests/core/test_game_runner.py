from typing import Any

import pytest

from rgi.players.random_player.random_player import RandomPlayer
from rgi.core import base
from rgi.core.game_runner import GameRunner
from rgi.games.connect4 import connect4
from rgi.games.othello import othello
from rgi.games.count21 import count21
from rgi.tests import test_utils

# pylint: disable=redefined-outer-name  # pytest fixtures trigger this false positive


@pytest.fixture
def connect4_game() -> connect4.Connect4Game:
    return connect4.Connect4Game()


@pytest.fixture
def connect4_players() -> list[test_utils.PresetPlayer[connect4.GameState, connect4.Action]]:
    return [
        test_utils.PresetPlayer[connect4.GameState, connect4.Action](actions=[2, 2, 2, 2]),
        test_utils.PresetPlayer[connect4.GameState, connect4.Action](actions=[1, 3, 4, 5]),
    ]


@pytest.fixture
def random_players() -> list[RandomPlayer[Any, Any]]:
    return [RandomPlayer(), RandomPlayer()]


def test_game_runner_basic(connect4_game: connect4.Connect4Game, random_players: list[RandomPlayer[Any, Any]]) -> None:
    """Test that GameRunner can run a basic game."""
    runner = GameRunner(connect4_game, random_players)
    trajectory = runner.run()

    # Basic validation
    assert len(trajectory.actions) > 0
    assert len(trajectory.game_states) == len(trajectory.actions) + 1
    assert len(trajectory.action_player_ids) == len(trajectory.actions)
    assert len(trajectory.incremental_rewards) == len(trajectory.actions)
    assert len(trajectory.final_reward) == trajectory.num_players

    equality_checker = test_utils.EqualityChecker()

    # Sanity check that final state is terminal and equal to the runner's state.
    assert connect4_game.is_terminal(runner.game_state)
    assert equality_checker.check_equality(trajectory.game_states[-1], runner.game_state)

    # Sanity check that second last state is not terminal and differes from final state.
    assert not connect4_game.is_terminal(trajectory.game_states[-2])
    assert not equality_checker.check_equality(trajectory.game_states[-2], runner.game_state)


@pytest.mark.parametrize(
    "player1_actions,player2_actions,expected_rewards",
    [
        ([2, 2, 2, 2], [1, 3, 4, 5], [1.0, -1.0]),  # Player 1 wins vertically
        ([2, 2, 1, 1, 2], [4, 4, 4, 7, 4], [-1.0, 1.0]),  # Player 2 wins vertically
        ([2, 3, 5, 2, 4], [1, 1, 1, 2, 2], [1.0, -1.0]),  # Player 1 wins horizontally
    ],
)
def test_game_runner_outcomes(
    connect4_game: connect4.Connect4Game,
    player1_actions: list[int],
    player2_actions: list[int],
    expected_rewards: list[float],
) -> None:
    """Test that GameRunner produces expected outcomes for different action sequences."""
    players = [
        test_utils.PresetPlayer[connect4.GameState, connect4.Action](actions=player1_actions),
        test_utils.PresetPlayer[connect4.GameState, connect4.Action](actions=player2_actions),
    ]

    runner = GameRunner(connect4_game, players, verbose=False)
    trajectory = runner.run()

    assert trajectory.final_reward == expected_rewards


def test_game_runner_verbose(
    connect4_game: connect4.Connect4Game,
    connect4_players: list[test_utils.PresetPlayer[connect4.Connect4State, connect4.Action]],
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Test that verbose mode prints game information."""
    runner = GameRunner(connect4_game, connect4_players, verbose=True)
    runner.run_step()

    captured = capsys.readouterr()
    assert "player_id=" in captured.out
    assert "action=" in captured.out
    assert "next state:" in captured.out
    assert "result=" in captured.out


# Validation of random player on various games.
@pytest.mark.parametrize(
    "game",
    [
        pytest.param(connect4.Connect4Game(), id="connect4"),
        pytest.param(othello.OthelloGame(), id="othello"),
        pytest.param(count21.Count21Game(), id="count21"),
    ],
)
def test_random_games(game: base.Game[Any, Any]) -> None:
    players = [RandomPlayer[Any, Any](), RandomPlayer[Any, Any]()]
    runner = GameRunner(game, players, verbose=False)
    runner.run()
    assert game.is_terminal(runner.game_state)

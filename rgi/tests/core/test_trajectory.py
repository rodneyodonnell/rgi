from pathlib import Path
from typing import Any, Type, Sequence

import pytest

from rgi.core.trajectory import GameTrajectory
from rgi.core import base
from rgi.core.game_runner import GameRunner
from rgi.core.archive import RowFileArchiver
from rgi.players.random_player.random_player import RandomPlayer
from rgi.games.connect4 import connect4
from rgi.games.othello import othello
from rgi.games.count21 import count21
from rgi.tests import test_utils

# pylint: disable=redefined-outer-name  # pytest fixtures trigger this false positive


@pytest.fixture
def connect4_game() -> connect4.Connect4Game:
    return connect4.Connect4Game()


def test_fixed_trajectory_save_load(
    connect4_game: connect4.Connect4Game,
    tmp_path: Path,
) -> None:
    """Test that trajectories can be saved and loaded correctly."""

    player1 = test_utils.PresetPlayer[connect4.GameState, connect4.Action](actions=[2, 2, 2, 2])
    player2 = test_utils.PresetPlayer[connect4.GameState, connect4.Action](actions=[1, 3, 4, 5])

    # Create and save trajectory
    runner = GameRunner(connect4_game, [player1, player2])
    original_trajectory = runner.run()
    save_path = tmp_path / "trajectory.npz"
    archiver = RowFileArchiver()
    archiver.write_items([original_trajectory], str(save_path))

    # Load trajectory
    reloaded_trajectories: Sequence[GameTrajectory[connect4.GameState, connect4.Action]] = archiver.read_items(
        str(save_path), GameTrajectory
    )
    reloaded_trajectory = reloaded_trajectories[0]

    # Check equality
    equality_checker = test_utils.EqualityChecker()
    assert equality_checker.check_equality(original_trajectory, reloaded_trajectory)
    assert not equality_checker.errors


def test_trajectory_validation(connect4_game: connect4.Connect4Game) -> None:
    """Test that GameTrajectory validates its inputs correctly."""

    # Create a valid state
    valid_state = connect4_game.initial_state()

    # Test with empty states
    with pytest.raises(ValueError, match="must contain at least one state"):
        GameTrajectory(
            game_states=[],
            actions=[],
            action_player_ids=[],
            incremental_rewards=[],
            num_players=2,
            final_reward=[1.0, -1.0],
        )

    # Test with mismatched lengths
    with pytest.raises(ValueError, match="must be one more than the number of actions"):
        GameTrajectory(
            game_states=[valid_state],
            actions=[1],
            action_player_ids=[1],
            incremental_rewards=[0.0],
            num_players=2,
            final_reward=[1.0, -1.0],
        )

    # Test with mismatched number of final rewards
    with pytest.raises(ValueError, match="must be the same as the number of players"):
        GameTrajectory(
            game_states=[valid_state, valid_state],
            actions=[1],
            action_player_ids=[1],
            incremental_rewards=[0.0],
            num_players=2,
            final_reward=[1.0],
        )


# Validation of random player on various games.
@pytest.mark.parametrize(
    "game,state_type,action_type, allow_pickle",
    [
        pytest.param(connect4.Connect4Game(), connect4.GameState, connect4.Action, False, id="connect4"),
        pytest.param(othello.OthelloGame(), othello.GameState, othello.Action, False, id="othello"),
        pytest.param(count21.Count21Game(), count21.TGameState, count21.TAction, False, id="count21"),
    ],
)
def test_random_games(
    game: base.Game[Any, Any], state_type: Type[Any], action_type: Type[Any], allow_pickle: bool, tmp_path: Path
) -> None:
    players = [RandomPlayer[Any, Any](), RandomPlayer[Any, Any]()]
    runner = GameRunner(game, players, verbose=False)
    trajectory = runner.run()
    assert game.is_terminal(runner.game_state)

    # save & reload
    save_path = tmp_path / "trajectory.npz"
    archiver = RowFileArchiver()
    archiver.write_items([trajectory], str(save_path))
    reloaded_trajectories: Sequence[GameTrajectory[Any, Any]] = archiver.read_items(str(save_path), GameTrajectory)
    reloaded_trajectory = reloaded_trajectories[0]

    equality_checker = test_utils.EqualityChecker()
    assert equality_checker.check_equality(trajectory, reloaded_trajectory)
    assert not equality_checker.errors

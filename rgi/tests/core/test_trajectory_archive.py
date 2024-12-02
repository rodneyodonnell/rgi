from pathlib import Path
from typing import Any, Type

import pytest

from rgi.core.trajectory_archive import AppendOnlyArchive, ReadOnlyArchive
from rgi.core.trajectory import GameTrajectory
from rgi.core.game_runner import GameRunner
from rgi.games.connect4 import connect4
from rgi.games.othello import othello
from rgi.games.count21 import count21
from rgi.tests import test_utils


@pytest.fixture
def connect4_trajectory() -> GameTrajectory[connect4.GameState, connect4.Action]:
    """Create a fixed Connect4 trajectory for testing."""
    game = connect4.Connect4Game()
    player1 = test_utils.PresetPlayer[connect4.GameState, connect4.Action](actions=[2, 2, 2, 2])
    player2 = test_utils.PresetPlayer[connect4.GameState, connect4.Action](actions=[1, 3, 4, 5])
    runner = GameRunner(game, [player1, player2])
    return runner.run()


def test_archive_basic(connect4_trajectory: GameTrajectory[Any, Any], tmp_path: Path) -> None:
    """Test basic archive operations."""
    archive_path = tmp_path / "archive"

    # Write trajectories
    with AppendOnlyArchive(archive_path, connect4.GameState, connect4.Action) as archive:
        archive.add_trajectory(connect4_trajectory)
        assert len(archive) == 1

    # Read and verify
    reader = ReadOnlyArchive(archive_path, connect4.GameState, connect4.Action)
    assert len(reader) == 1

    # Verify trajectory contents
    equality_checker = test_utils.EqualityChecker()
    retrieved_trajectory = reader[0]
    assert equality_checker.check_equality(connect4_trajectory, retrieved_trajectory)
    assert not equality_checker.errors

    reader.close()


def test_archive_multiple_trajectories(connect4_trajectory: GameTrajectory[Any, Any], tmp_path: Path) -> None:
    """Test archive with multiple trajectories."""
    archive_path = tmp_path / "archive"

    # Write multiple trajectories
    num_trajectories = 5
    with AppendOnlyArchive(archive_path, connect4.GameState, connect4.Action) as archive:
        for _ in range(num_trajectories):
            archive.add_trajectory(connect4_trajectory)
        assert len(archive) == num_trajectories

    # Read and verify each trajectory
    reader = ReadOnlyArchive(archive_path, connect4.GameState, connect4.Action)
    assert len(reader) == num_trajectories

    for trajectory in reader:
        equality_checker = test_utils.EqualityChecker()
        assert equality_checker.check_equality(connect4_trajectory, trajectory)
        assert not equality_checker.errors

    reader.close()


@pytest.mark.parametrize(
    "game,state_type,action_type",
    [
        pytest.param(connect4.Connect4Game(), connect4.GameState, connect4.Action, id="connect4"),
        pytest.param(othello.OthelloGame(), othello.GameState, othello.Action, id="othello"),
        pytest.param(count21.Count21Game(), count21.TGameState, count21.TAction, id="count21"),
    ],
)
def test_archive_different_games(
    game: Any,
    state_type: Type[Any],
    action_type: Type[Any],
    tmp_path: Path,
) -> None:
    """Test archive with different game types."""
    archive_path = tmp_path / "archive"

    # Create a trajectory
    players = [
        test_utils.PresetPlayer[Any, Any](actions=[1, 2, 3]),
        test_utils.PresetPlayer[Any, Any](actions=[1, 2, 3]),
    ]
    runner = GameRunner(game, players)
    trajectory = runner.run()

    # Write trajectory
    with AppendOnlyArchive(archive_path, state_type, action_type) as archive:
        archive.add_trajectory(trajectory)

    # Read and verify
    reader = ReadOnlyArchive(archive_path, state_type, action_type)
    retrieved_trajectory = reader[0]

    equality_checker = test_utils.EqualityChecker()
    assert equality_checker.check_equality(trajectory, retrieved_trajectory)
    assert not equality_checker.errors

    reader.close()

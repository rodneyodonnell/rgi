from pathlib import Path
from typing import Any, Type

import pytest
import numpy as np

from rgi.core.archive import (
    Archive,
    ListBasedArchive,
    PickleBasedArchive,
    CombinedArchive,
)
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


def test_list_archive_basic(connect4_trajectory: GameTrajectory[Any, Any]) -> None:
    """Test basic ListBasedArchive operations."""
    archive = ListBasedArchive(connect4.GameState, connect4.Action)
    archive.add_trajectory(connect4_trajectory)

    assert len(archive) == 1
    retrieved = archive[0]

    equality_checker = test_utils.EqualityChecker()
    assert equality_checker.check_equality(connect4_trajectory, retrieved)
    assert not equality_checker.errors


def test_pickle_archive_basic(connect4_trajectory: GameTrajectory[Any, Any], tmp_path: Path) -> None:
    """Test basic PickleBasedArchive operations."""
    archive_path = tmp_path / "archive.npz"

    # Create list archive and save to file
    list_archive = ListBasedArchive(connect4.GameState, connect4.Action)
    list_archive.add_trajectory(connect4_trajectory)
    PickleBasedArchive.save(list_archive, archive_path)

    # Load and verify
    pickle_archive = PickleBasedArchive(archive_path, connect4.GameState, connect4.Action)
    assert len(pickle_archive) == 1

    retrieved = pickle_archive[0]
    equality_checker = test_utils.EqualityChecker()
    assert equality_checker.check_equality(connect4_trajectory, retrieved)
    assert not equality_checker.errors

    pickle_archive.close()


def test_combined_archive(connect4_trajectory: GameTrajectory[Any, Any], tmp_path: Path) -> None:
    """Test CombinedArchive operations."""
    # Create list archive
    list_archive = ListBasedArchive(connect4.GameState, connect4.Action)
    list_archive.add_trajectory(connect4_trajectory)

    # Create pickle archive
    archive_path = tmp_path / "archive.npz"
    PickleBasedArchive.save(list_archive, archive_path)
    pickle_archive = PickleBasedArchive(archive_path, connect4.GameState, connect4.Action)

    # Create combined archive
    combined = CombinedArchive([list_archive, pickle_archive])
    assert len(combined) == 2

    # Verify both trajectories
    for i in range(2):
        retrieved = combined[i]
        equality_checker = test_utils.EqualityChecker()
        assert equality_checker.check_equality(connect4_trajectory, retrieved)
        assert not equality_checker.errors

    pickle_archive.close()


@pytest.mark.parametrize(
    "game,state_type,action_type,test_actions",
    [
        pytest.param(
            connect4.Connect4Game(),
            connect4.GameState,
            connect4.Action,
            [[1, 2, 3], [2, 3, 4]],
            id="connect4",
        ),
        pytest.param(
            othello.OthelloGame(),
            othello.GameState,
            othello.Action,
            [[(2, 3), (2, 4), (2, 5)], [(3, 2), (4, 2), (5, 2)]],
            id="othello",
        ),
        pytest.param(
            count21.Count21Game(),
            count21.TGameState,
            count21.TAction,
            [[1, 1, 1], [2, 2, 2]],
            id="count21",
        ),
    ],
)
def test_archive_different_games(
    game: Any,
    state_type: Type[Any],
    action_type: Type[Any],
    test_actions: list[list[Any]],
    tmp_path: Path,
) -> None:
    """Test archives with different game types."""
    # Create a trajectory
    players = [
        test_utils.PresetPlayer[Any, Any](actions=test_actions[0]),
        test_utils.PresetPlayer[Any, Any](actions=test_actions[1]),
    ]
    runner = GameRunner(game, players)
    trajectory = runner.run()

    # Test list archive
    list_archive = ListBasedArchive(state_type, action_type)
    list_archive.add_trajectory(trajectory)
    retrieved = list_archive[0]

    equality_checker = test_utils.EqualityChecker()
    assert equality_checker.check_equality(trajectory, retrieved)
    assert not equality_checker.errors

    # Test pickle archive
    archive_path = tmp_path / "archive.npz"
    PickleBasedArchive.save(list_archive, archive_path)

    pickle_archive = PickleBasedArchive(archive_path, state_type, action_type)
    retrieved = pickle_archive[0]

    equality_checker = test_utils.EqualityChecker()
    assert equality_checker.check_equality(trajectory, retrieved)
    assert not equality_checker.errors

    pickle_archive.close()

import time
import psutil
import pytest
from pytest_benchmark.fixture import BenchmarkFixture

from pathlib import Path
import numpy as np
from typing import Type, Any
import random

from rgi.games.connect4 import connect4
from rgi.games.count21 import count21

from rgi.tests import test_utils

from rgi.core.trajectory_archive import (
    BaseArchive,
    SingleFileArchive,
    # SplitFileArchive,
    MemoryMappedArchive,
    ArchiveStats,
    AppendOnlyArchive,
    ReadOnlyArchive,
)
from rgi.core.trajectory import GameTrajectory, TrajectoryBuilder
from rgi.core.base import Game, TGameState, TAction


def get_memory_usage() -> float:
    """Get current memory usage in MB"""
    process = psutil.Process()
    return process.memory_info().rss / (1024 * 1024)  # type: ignore


def random_trajectory(
    game: Game[TGameState, TAction], trajectory_length: int = 20
) -> GameTrajectory[TGameState, TAction]:
    """Generate a random trajectory for testing"""
    state = game.initial_state()

    builder = TrajectoryBuilder(game, state)
    for _ in range(trajectory_length):
        # Cycle though games until we hit the required length.
        if game.is_terminal(state):
            state = game.initial_state()

        action_player_id = game.current_player_id(state)
        legal_actions = game.legal_actions(state)
        action = random.choice(legal_actions)
        state = game.next_state(state, action)

        builder.record_step(
            action_player_id=action_player_id,
            action=action,
            updated_state=state,
            incremental_reward=0.0,
        )

    return builder.build()


def benchmark_archive(
    trajectories: list[GameTrajectory[TGameState, TAction]],
    game_state_type: Type[TGameState],
    action_type: Type[TAction],
    tmp_path: Path,
) -> ArchiveStats:
    """Benchmark archive implementations"""
    archive_path = tmp_path / "test_archive"

    # Measure write performance
    initial_memory = get_memory_usage()
    peak_memory = initial_memory

    write_start = time.perf_counter()
    with AppendOnlyArchive(archive_path, game_state_type, action_type) as archive:
        for trajectory in trajectories:
            archive.add_trajectory(trajectory)
            peak_memory = max(peak_memory, get_memory_usage())
    write_time = (time.perf_counter() - write_start) * 1000

    # Measure read performance
    read_start = time.perf_counter()
    reader = ReadOnlyArchive(archive_path, game_state_type, action_type)

    # Test both sequential and random access
    sequential_trajectories = list(reader)
    random_indices = [random.randrange(len(reader)) for _ in range(min(10, len(reader)))]
    random_trajectories = [reader[i] for i in random_indices]

    reader.close()
    read_time = (time.perf_counter() - read_start) * 1000

    # Get file size
    file_size = sum(p.stat().st_size for p in [archive_path.with_suffix(".rgi"), archive_path.with_suffix(".idx")])

    return ArchiveStats(
        write_time_ms=write_time,
        read_time_ms=read_time,
        file_size_bytes=file_size,
        peak_memory_mb=peak_memory - initial_memory,
    )


@pytest.mark.benchmark(
    group="archive",
    min_rounds=5,
)
@pytest.mark.parametrize(
    "game,game_state_type,action_type",
    [
        (connect4.Connect4Game(), connect4.GameState, connect4.Action),
        (count21.Count21Game(), count21.TGameState, count21.TAction),
    ],
)
@pytest.mark.parametrize("num_trajectories", [10])
@pytest.mark.parametrize("trajectory_length", [10, 50])
def test_archive_performance(
    benchmark: BenchmarkFixture,
    game: Game[TGameState, TAction],
    game_state_type: Type[TGameState],
    action_type: Type[TAction],
    num_trajectories: int,
    trajectory_length: int,
    tmp_path: Path,
) -> None:
    def run_benchmark() -> ArchiveStats:
        test_utils.delete_directory_contents(tmp_path)
        return benchmark_archive(
            trajectories,
            game_state_type,
            action_type,
            tmp_path,
        )

    # Create sample trajectories
    trajectories = []
    for _ in range(num_trajectories):
        trajectories.append(random_trajectory(game, trajectory_length=trajectory_length))

    stats = benchmark.pedantic(run_benchmark, iterations=1, rounds=5)

    print(f"\nArchive Performance:")
    print(f"Write time: {stats.write_time_ms:.2f}ms")
    print(f"Read time: {stats.read_time_ms:.2f}ms")
    print(f"File size: {stats.file_size_bytes / 1024:.2f}KB")
    print(f"Peak memory: {stats.peak_memory_mb:.2f}MB")

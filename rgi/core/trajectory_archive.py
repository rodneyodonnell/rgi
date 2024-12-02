import io
import os
import mmap
import struct
from typing import override
from typing import Generic, Iterator, Type
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from abc import ABC, abstractmethod

from rgi.core.base import TGameState, TAction
from rgi.core.trajectory import GameTrajectory


class AppendOnlyArchive(Generic[TGameState, TAction]):
    """Archive that only supports appending new trajectories.

    Example:
    >>> from rgi.games.connect4 import connect4
    >>> with AppendOnlyArchive('archive', connect4.GameState, connect4.Action) as archive:
    ...     archive.add_trajectory(trajectory1)
    ...     archive.add_trajectory(trajectory2)
    ... # Files automatically closed after with block
    """

    def __init__(self, filepath: Path | str, game_state_type: Type[TGameState], action_type: Type[TAction]):
        self.game_state_type = game_state_type
        self.action_type = action_type

        self.index_path = Path(filepath).with_suffix(".idx")
        self.data_path = Path(filepath).with_suffix(".rgi")

        # Open in append mode
        self._index_file = open(self.index_path, "ab")
        self._data_file = open(self.data_path, "ab")

        # Track number of trajectories written
        self._index_file.seek(0, os.SEEK_END)
        self._num_written = self._index_file.tell() // 4  # 4 bytes per index

    def __enter__(self) -> "AppendOnlyArchive[TGameState, TAction]":
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self.save()

    def add_trajectory(self, trajectory: GameTrajectory[TGameState, TAction]) -> None:
        """Add a new trajectory to the archive."""
        pos = self._data_file.tell()
        trajectory.write(self._data_file)
        self._index_file.write(struct.pack("I", pos))
        self._num_written += 1

    def save(self) -> None:
        """Close the archive files."""
        self._index_file.close()
        self._data_file.close()

    def __len__(self) -> int:
        return self._num_written


class ReadOnlyArchive(Generic[TGameState, TAction]):
    """Archive that only supports reading existing trajectories."""

    def __init__(self, filepath: Path | str, game_state_type: Type[TGameState], action_type: Type[TAction]):
        self.game_state_type = game_state_type
        self.action_type = action_type

        self.index_path = Path(filepath).with_suffix(".idx")
        self.data_path = Path(filepath).with_suffix(".rgi")

        if not (self.index_path.exists() and self.data_path.exists()):
            raise ValueError(f"Archive files not found at {filepath}")

        # Open files and create memory maps
        self._index_file = open(self.index_path, "rb")
        self._data_file = open(self.data_path, "rb")

        self._index_mmap = mmap.mmap(self._index_file.fileno(), 0, access=mmap.ACCESS_READ)
        self._data_mmap = mmap.mmap(self._data_file.fileno(), 0, access=mmap.ACCESS_READ)

        self._num_trajectories = len(self._index_mmap) // 4  # 4 bytes per index

    def get_trajectory(self, index: int) -> GameTrajectory[TGameState, TAction]:
        """Get a trajectory by index."""
        if not 0 <= index < self._num_trajectories:
            raise IndexError("Trajectory index out of range")

        # Read start position
        start_bytes = self._index_mmap[index * 4 : (index + 1) * 4]
        start = struct.unpack("I", start_bytes)[0]

        # Read end position (next index or end of file)
        if index + 1 < self._num_trajectories:
            end_bytes = self._index_mmap[(index + 1) * 4 : (index + 2) * 4]
            end = struct.unpack("I", end_bytes)[0]
        else:
            end = len(self._data_mmap)

        # Extract trajectory data
        data = self._data_mmap[start:end]
        return GameTrajectory.read(io.BytesIO(data), self.game_state_type, self.action_type)

    def __getitem__(self, index: int) -> GameTrajectory[TGameState, TAction]:
        """Allow array-like access to trajectories."""
        return self.get_trajectory(index)

    def __len__(self) -> int:
        return self._num_trajectories

    def __iter__(self) -> Iterator[GameTrajectory[TGameState, TAction]]:
        """Iterate over all trajectories."""
        for i in range(len(self)):
            yield self.get_trajectory(i)

    def close(self) -> None:
        """Close all file handles."""
        self._index_mmap.close()
        self._data_mmap.close()
        self._index_file.close()
        self._data_file.close()

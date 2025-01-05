from abc import ABC, abstractmethod
from pathlib import Path
from typing import Generic, Iterator, List, TypeVar, cast

import numpy as np

T = TypeVar("T")


class Archive(Generic[T], ABC):
    """Base Archive class for storing lists of @dataclass objects."""

    @abstractmethod
    def __len__(self) -> int:
        """Return number of items in archive."""

    @abstractmethod
    def __getitem__(self, idx: int) -> T:
        """Get items at given index."""

    @abstractmethod
    def __iter__(self) -> Iterator[T]:
        """Iterate over items."""

    def close(self) -> None:
        """Close the archive and free resources."""


class ListBasedArchive(Archive[T]):
    """In-memory archive simply storing items in a list."""

    def __init__(self, item_type: type[T]):
        """Initialize empty archive."""
        self._item_type = item_type
        self._items: List[T] = []

    def add_item(self, item: T) -> None:
        """Add item to archive."""
        self._items.append(item)

    def __len__(self) -> int:
        return len(self._items)

    def __getitem__(self, idx: int) -> T:
        return self._items[idx]

    def __iter__(self) -> Iterator[T]:
        return iter(self._items)


class MMappedArchive(Archive[T]):
    """Read-only archive storing items in a mmaped numpy file."""

    def __init__(self, filepath: Path, item_type: type[T]):
        """Initialize archive from file.

        Args:
            filepath: Path to archive file
            item_type: Type of items stored in archive
        """
        self._filepath = filepath
        self._item_type = item_type
        self._data = np.load(filepath, mmap_mode="r", allow_pickle=True)
        self._length = len(self._data)  # TODO: fix.

    @staticmethod
    def save(archive: Archive[T], filepath: Path) -> None:
        """Save archive to file."""
        data_dict = {}
        for i, trajectory in enumerate(archive):
            prefix = f"traj_{i}_"
            # Access internal attributes through dict to avoid type issues
            traj_dict = trajectory.__dict__
            data_dict[f"{prefix}states"] = traj_dict["game_states"]
            data_dict[f"{prefix}actions"] = traj_dict["actions"]
            data_dict[f"{prefix}action_player_ids"] = traj_dict["action_player_ids"]
            data_dict[f"{prefix}incremental_rewards"] = traj_dict["incremental_rewards"]
            data_dict[f"{prefix}num_players"] = np.array([traj_dict["num_players"]], dtype=np.int64)
            data_dict[f"{prefix}final_reward"] = traj_dict["final_reward"]

        np.savez_compressed(filepath, **data_dict)

    def __len__(self) -> int:
        return self._length

    def __getitem__(self, idx: int) -> T:
        if not 0 <= idx < len(self):
            raise IndexError(f"Index {idx} out of range for archive with {len(self)} trajectories")

        prefix = f"traj_{idx}_"
        states = self._data[f"{prefix}states"]
        actions = self._data[f"{prefix}actions"]
        action_player_ids = self._data[f"{prefix}action_player_ids"]
        incremental_rewards = self._data[f"{prefix}incremental_rewards"]
        num_players = self._data[f"{prefix}num_players"].item()
        final_reward = self._data[f"{prefix}final_reward"]

        # # Cast to correct types since we're loading from file
        # return cast(
        #     GameTrajectory[TState, TMove],
        #     GameTrajectory(
        #         game_states=states,
        #         actions=actions,
        #         action_player_ids=action_player_ids,
        #         incremental_rewards=incremental_rewards,
        #         num_players=num_players,
        #         final_reward=final_reward,
        #     ),
        # )
        return self._item_type(
            states=states,
            actions=actions,
            action_player_ids=action_player_ids,
            incremental_rewards=incremental_rewards,
            num_players=num_players,
            final_reward=final_reward,
        )

    def __iter__(self) -> Iterator[T]:
        for i in range(len(self)):
            yield self[i]

    def close(self) -> None:
        """Close the archive and free resources."""
        self._data.close()


class CombinedArchive(Archive[T]):
    """Archive combining multiple archives into a single view."""

    def __init__(self, archives: List[Archive[T]]):
        """Initialize combined archive.

        Args:
            archives: List of archives to combine
        """
        self._archives = archives
        self._lengths = [len(archive) for archive in archives]
        self._cumsum = np.cumsum([0] + self._lengths)

    def _locate(self, idx: int) -> tuple[Archive[T], int]:
        """Find archive and local index for global index.

        Args:
            idx: Global index into combined archive

        Returns:
            Tuple of (archive, local_index)

        Raises:
            IndexError: If index is out of range
        """
        if not 0 <= idx < len(self):
            raise IndexError(f"Index {idx} out of range for archive with {len(self)} trajectories")

        archive_idx = np.searchsorted(self._cumsum[1:], idx, side="right")
        local_idx = idx - self._cumsum[archive_idx]
        return self._archives[archive_idx], local_idx

    def __len__(self) -> int:
        return self._cumsum[-1]

    def __getitem__(self, idx: int) -> T:
        archive, local_idx = self._locate(idx)
        return archive[local_idx]

    def __iter__(self) -> Iterator[T]:
        for archive in self._archives:
            yield from archive

    def close(self) -> None:
        """Close all archives."""
        for archive in self._archives:
            archive.close()

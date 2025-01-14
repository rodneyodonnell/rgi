import abc
import dataclasses
import typing
import types

from typing import TypeVar, Any, Sequence, cast
from types import GenericAlias


import numpy as np

from rgi.core.types import FileOrPath, PrimitiveType, DataclassProtocol, is_primitive_type, is_dataclass_type

T = typing.TypeVar("T")


ArchiveColumn = np.ndarray[Any, np.dtype[Any]]
ArchiveColumns = dict[str, ArchiveColumn]


class Archive(Sequence[T], abc.ABC):
    pass


class AppendableArchive(Archive[T]):
    @abc.abstractmethod
    def append(self, item: T) -> None:
        """Add item to archive."""


class ListBasedArchive(AppendableArchive[T]):
    """In-memory archive simply storing items in a list."""

    def __init__(self, item_type: type[T]):
        """Initialize empty archive."""
        self._item_type = item_type
        self._items: list[T] = []

    @typing.override
    def append(self, item: T) -> None:
        """Add item to archive."""
        self._items.append(item)

    @typing.override
    def __len__(self) -> int:
        return len(self._items)

    @typing.overload
    def __getitem__(self, idx: int) -> T: ...

    @typing.overload
    def __getitem__(self, idx: slice) -> Sequence[T]: ...

    @typing.override
    def __getitem__(self, idx: int | slice) -> T | Sequence[T]:
        return self._items[idx]

    @typing.override
    def __repr__(self) -> str:
        return f"ListBasedArchive(item_type={self._item_type}, len={len(self)}, items[:1]={self._items[:1]})"


class ArchiveSerializer(typing.Generic[T]):
    def __init__(self, item_type: type[T] | types.GenericAlias):
        self._item_type = item_type

    def serialize_to_dict(self, items: Sequence[T]) -> ArchiveColumns:
        return self._serialize_to_dict("", self._item_type, items)

    def serialize_to_file(self, items: Sequence[T], file: FileOrPath) -> None:
        d = self.serialize_to_dict(items)
        np.savez_compressed(file, **d)

    _U = TypeVar("_U")

    def _serialize_to_dict(
        self, field_path: str, item_type: type[_U] | GenericAlias, items: Sequence[_U]
    ) -> ArchiveColumns:

        if is_primitive_type(item_type):
            return self._serialize_primitive(field_path, item_type, cast(Sequence[PrimitiveType], items))

        if is_dataclass_type(item_type):
            return self._serialize_dataclass(field_path, item_type, cast(Sequence[DataclassProtocol], items))

        if item_type is np.ndarray:
            return self._serialize_ndarray(field_path, cast(Sequence[np.ndarray[Any, Any]], items))

        if (base_type := typing.get_origin(item_type)) is not None:
            base_type_args = typing.get_args(item_type)

            if base_type is list:
                return self._serialize_generic_list(field_path, base_type_args[0], cast(Sequence[Sequence[Any]], items))
            if base_type is tuple:
                return self._serialize_generic_tuple(field_path, base_type_args, cast(Sequence[Sequence[Any]], items))
            if base_type is np.ndarray:
                return self._serialize_ndarray(field_path, cast(Sequence[np.ndarray[Any, Any]], items))

        raise NotImplementedError(f"Cannot add fields for field `{field_path}` with unhandled type {item_type}")

    def _serialize_primitive(
        self, field_path: str, item_type: type[PrimitiveType], items: Sequence[PrimitiveType]
    ) -> ArchiveColumns:
        """Serialize primitive types to ndarray."""
        # TODO: Check returned array is not of type 'o' if serialization is strict.
        return {field_path: np.array(items, dtype=item_type)}

    def _serialize_dataclass(
        self, field_path: str, item_type: type[DataclassProtocol], items: Sequence[DataclassProtocol]
    ) -> ArchiveColumns:
        """For dataclass types, which will recursively handle fields of various types."""
        d: ArchiveColumns = {}
        for field in dataclasses.fields(item_type):
            field_type = field.type
            if not isinstance(field_type, (type, GenericAlias)):
                raise ValueError(f"Field {field.name} with field_type {field_type} is not a Type. Unable to serialize.")

            field_key = f"{field_path}.{field.name}"
            field_items = [getattr(item, field.name) for item in items]

            field_dict = self._serialize_to_dict(field_key, field_type, field_items)
            d.update(field_dict)
        return d

    def _serialize_ndarray(self, field_path: str, items: Sequence[np.ndarray[Any, Any]]) -> ArchiveColumns:
        flat_values = np.concatenate([arr.flatten() for arr in items])
        shapes = [arr.shape for arr in items]

        values_dict = {f"{field_path}.*": flat_values}
        shape_dict = self._serialize_to_dict(f"{field_path}.#", tuple[int, ...], shapes)
        return values_dict | shape_dict

    def _serialize_generic_list(
        self, field_path: str, item_type: type[_U], items: Sequence[Sequence[_U]]
    ) -> ArchiveColumns:
        unrolled_items = [item for item_list in items for item in item_list]
        unrolled_lengths = [len(item_list) for item_list in items]
        values_dict = self._serialize_to_dict(f"{field_path}.*", item_type, unrolled_items)
        length_dict = self._serialize_to_dict(f"{field_path}.#", int, unrolled_lengths)
        return values_dict | length_dict

    def _serialize_generic_tuple(
        self, field_path: str, base_type_args: tuple[type, ...], items: Sequence[Sequence[_U]]
    ) -> ArchiveColumns:
        if base_type_args[-1] is Ellipsis:  # type: ignore
            return self._serialize_generic_list(field_path, base_type_args[0], items)

        d = {}
        for i, t in enumerate(base_type_args):
            tuple_field_path = f"{field_path}.{i}"
            tuple_field_items = [item[i] for item in items]
            tuple_serialized = self._serialize_to_dict(tuple_field_path, t, tuple_field_items)
            d.update(tuple_serialized)
        return d


# class MMappedArchive(Archive[T]):
#     """Read-only archive storing items in a mmaped numpy file."""

#     def __init__(self, filepath: Path, item_type: type[T]):
#         """Initialize archive from file.

#         Args:
#             filepath: Path to archive file
#             item_type: Type of items stored in archive
#         """
#         self._filepath = filepath
#         self._item_type = item_type
#         self._data = np.load(filepath, mmap_mode="r", allow_pickle=True)
#         self._length = len(self._data)  # TODO: fix.

#     @staticmethod
#     def save(archive: Archive[T], filepath: Path) -> None:
#         """Save archive to file."""
#         data_dict = {}
#         for i, trajectory in enumerate(archive):
#             prefix = f"traj_{i}_"
#             # Access internal attributes through dict to avoid type issues
#             traj_dict = trajectory.__dict__
#             data_dict[f"{prefix}states"] = traj_dict["game_states"]
#             data_dict[f"{prefix}actions"] = traj_dict["actions"]
#             data_dict[f"{prefix}action_player_ids"] = traj_dict["action_player_ids"]
#             data_dict[f"{prefix}incremental_rewards"] = traj_dict["incremental_rewards"]
#             data_dict[f"{prefix}num_players"] = np.array([traj_dict["num_players"]], dtype=np.int64)
#             data_dict[f"{prefix}final_reward"] = traj_dict["final_reward"]

#         np.savez_compressed(filepath, **data_dict)

#     def __len__(self) -> int:
#         return self._length

#     def __getitem__(self, idx: int) -> T:
#         if not 0 <= idx < len(self):
#             raise IndexError(f"Index {idx} out of range for archive with {len(self)} trajectories")

#         prefix = f"traj_{idx}_"
#         states = self._data[f"{prefix}states"]
#         actions = self._data[f"{prefix}actions"]
#         action_player_ids = self._data[f"{prefix}action_player_ids"]
#         incremental_rewards = self._data[f"{prefix}incremental_rewards"]
#         num_players = self._data[f"{prefix}num_players"].item()
#         final_reward = self._data[f"{prefix}final_reward"]

#         # # Cast to correct types since we're loading from file
#         # return cast(
#         #     GameTrajectory[TState, TMove],
#         #     GameTrajectory(
#         #         game_states=states,
#         #         actions=actions,
#         #         action_player_ids=action_player_ids,
#         #         incremental_rewards=incremental_rewards,
#         #         num_players=num_players,
#         #         final_reward=final_reward,
#         #     ),
#         # )
#         return self._item_type(
#             states=states,
#             actions=actions,
#             action_player_ids=action_player_ids,
#             incremental_rewards=incremental_rewards,
#             num_players=num_players,
#             final_reward=final_reward,
#         )

#     def __iter__(self) -> Iterator[T]:
#         for i in range(len(self)):
#             yield self[i]

#     def close(self) -> None:
#         """Close the archive and free resources."""
#         self._data.close()


# class CombinedArchive(Archive[T]):
#     """Archive combining multiple archives into a single view."""

#     def __init__(self, archives: list[Archive[T]]):
#         """Initialize combined archive.

#         Args:
#             archives: List of archives to combine
#         """
#         self._archives = archives
#         self._lengths = [len(archive) for archive in archives]
#         self._cumsum = np.cumsum([0] + self._lengths)

#     def _locate(self, idx: int) -> tuple[Archive[T], int]:
#         """Find archive and local index for global index.

#         Args:
#             idx: Global index into combined archive

#         Returns:
#             Tuple of (archive, local_index)

#         Raises:
#             IndexError: If index is out of range
#         """
#         if not 0 <= idx < len(self):
#             raise IndexError(f"Index {idx} out of range for archive with {len(self)} trajectories")

#         archive_idx = np.searchsorted(self._cumsum[1:], idx, side="right")
#         local_idx = idx - self._cumsum[archive_idx]
#         return self._archives[archive_idx], local_idx

#     def __len__(self) -> int:
#         return self._cumsum[-1]

#     def __getitem__(self, idx: int) -> T:
#         archive, local_idx = self._locate(idx)
#         return archive[local_idx]

#     def __iter__(self) -> Iterator[T]:
#         for archive in self._archives:
#             yield from archive

#     def close(self) -> None:
#         """Close all archives."""
#         for archive in self._archives:
#             archive.close()

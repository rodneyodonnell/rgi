import abc
import dataclasses
import typing
import types
import pathlib
from typing import Any, Sequence, cast
from types import GenericAlias


import numpy as np

from rgi.core.types import (
    FileOrPath,
    PrimitiveType,
    DataclassProtocol,
    is_primitive_type,
    is_dataclass_type,
)

T = typing.TypeVar("T")
_U = typing.TypeVar("_U")


ArchiveColumn = np.ndarray[Any, np.dtype[Any]]
ArchiveColumns = dict[str, ArchiveColumn]
ArchiveMemmap = np.memmap[Any, Any]


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


class MMappedArchive(Archive[T]):
    """Read-only archive storing items in a mmaped numpy file."""

    def __init__(
        self,
        file: FileOrPath,
        item_type: type[T],
        serializer: "ArchiveSerializer[T]",
        allow_pickle: bool = True,
    ):
        """Initialize archive from file.

        Args:
            file: File to load archive from
            item_type: Type of items stored in archive
        """
        self._file: FileOrPath = file
        self._item_type: type[T] = item_type
        self._serializer: ArchiveSerializer[T] = serializer
        self._data: np.memmap[Any, Any] = np.load(file, mmap_mode="r", allow_pickle=allow_pickle)

    def __enter__(self) -> "MMappedArchive[T]":
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: types.TracebackType | None,
    ) -> None:
        self._data.close()  # type: ignore

    @typing.override
    def __len__(self) -> int:
        return len(self._data)

    @typing.overload
    def __getitem__(self, idx: int) -> T: ...

    @typing.overload
    def __getitem__(self, idx: slice) -> Sequence[T]: ...

    @typing.override
    def __getitem__(self, idx: int | slice) -> T | Sequence[T]:
        if isinstance(idx, slice):
            raise NotImplementedError("Cannot get slice from mmaped archive")
        return self._serializer._get_item("", self._item_type, idx, self._data)

    @typing.override
    def __iter__(self) -> typing.Iterator[T]:
        for i in range(len(self)):
            yield self[i]


class ArchiveSerializer(typing.Generic[T]):
    def __init__(self, item_type: type[T] | types.GenericAlias):
        self._item_type = item_type

    def save(self, items: Sequence[T], file: FileOrPath) -> None:
        columns = self.to_columns(items)
        return self.to_file(columns, file)

    def load_sequence(self, file: FileOrPath, slice_: slice | None = None) -> Sequence[T]:
        columns = np.load(file)
        return self.from_columns(columns, slice_)

    def load_mmap(self, path: FileOrPath) -> MMappedArchive[T]:
        return MMappedArchive(path, self._item_type, self)

    def to_columns(self, items: Sequence[T]) -> ArchiveColumns:
        return self._to_columns("", self._item_type, items)

    def from_columns(self, columns: ArchiveColumns, slice_: slice | None = None) -> Sequence[T]:
        slice_ = slice_ or slice(None, None, 1)
        return self._from_columns("", self._item_type, columns, slice_)

    def to_file(self, columns: ArchiveColumns, file: FileOrPath) -> None:
        np.savez_compressed(file, **columns)

    def from_file(self, file: FileOrPath) -> ArchiveColumns:
        columns: ArchiveColumns = np.load(file)
        return columns

    def _to_columns(self, field_path: str, item_type: type[_U] | GenericAlias, items: Sequence[_U]) -> ArchiveColumns:

        if is_primitive_type(item_type):
            return self._to_primitive_columns(field_path, item_type, cast(Sequence[PrimitiveType], items))

        if is_dataclass_type(item_type):
            return self._to_dataclass_columns(field_path, item_type, cast(Sequence[DataclassProtocol], items))

        if item_type is np.ndarray:
            return self._to_ndarray_columns(field_path, cast(Sequence[np.ndarray[Any, Any]], items))

        if (base_type := typing.get_origin(item_type)) is not None:
            base_type_args = typing.get_args(item_type)

            if base_type is list:
                return self._to_generic_list_columns(
                    field_path, base_type_args[0], cast(Sequence[Sequence[Any]], items)
                )
            if base_type is tuple:
                return self._to_generic_tuple_columns(field_path, base_type_args, cast(Sequence[Sequence[Any]], items))
            if base_type is np.ndarray:
                return self._to_ndarray_columns(field_path, cast(Sequence[np.ndarray[Any, Any]], items))

        raise NotImplementedError(f"Cannot add fields for field `{field_path}` with unhandled type {item_type}")

    def _to_primitive_columns(
        self,
        field_path: str,
        item_type: type[PrimitiveType],
        items: Sequence[PrimitiveType],
    ) -> ArchiveColumns:
        """Serialize primitive types to ndarray."""
        # TODO: Check returned array is not of type 'o' if serialization is strict.
        return {field_path: np.array(items, dtype=item_type)}

    def _to_dataclass_columns(
        self,
        field_path: str,
        item_type: type[DataclassProtocol],
        items: Sequence[DataclassProtocol],
    ) -> ArchiveColumns:
        """For dataclass types, which will recursively handle fields of various types."""
        d: ArchiveColumns = {}
        for field in dataclasses.fields(item_type):
            field_type = field.type
            assert isinstance(field_type, (type, GenericAlias))

            field_key = f"{field_path}.{field.name}"
            field_items = [getattr(item, field.name) for item in items]

            field_dict = self._to_columns(field_key, field_type, field_items)
            d.update(field_dict)
        return d

    def _to_ndarray_columns(self, field_path: str, items: Sequence[np.ndarray[Any, Any]]) -> ArchiveColumns:
        flat_values = np.concatenate([arr.flatten() for arr in items])
        shapes = [arr.shape for arr in items]

        values_dict = {f"{field_path}.*": flat_values}
        shape_dict = self._to_columns(f"{field_path}.#", tuple[int, ...], shapes)
        return values_dict | shape_dict

    def _to_generic_list_columns(
        self, field_path: str, item_type: type[_U], items: Sequence[Sequence[_U]]
    ) -> ArchiveColumns:
        unrolled_items = [item for item_list in items for item in item_list]
        length_cumsum = np.cumsum([0] + [len(item_list) for item_list in items])
        values_dict = self._to_columns(f"{field_path}.*", item_type, unrolled_items)
        length_dict = self._to_columns(f"{field_path}.#", int, length_cumsum)
        return values_dict | length_dict

    def _to_generic_tuple_columns(
        self,
        field_path: str,
        base_type_args: tuple[type, ...],
        items: Sequence[Sequence[_U]],
    ) -> ArchiveColumns:
        if base_type_args[-1] is Ellipsis:  # type: ignore
            return self._to_generic_list_columns(field_path, base_type_args[0], items)

        d = {}
        for i, t in enumerate(base_type_args):
            tuple_field_path = f"{field_path}.{i}"
            tuple_field_items = [item[i] for item in items]
            tuple_serialized = self._to_columns(tuple_field_path, t, tuple_field_items)
            d.update(tuple_serialized)
        return d

    def _from_columns(
        self,
        field_path: str,
        item_type: type[_U] | GenericAlias,
        columns: ArchiveColumns,
        slice_: slice | None = None,
    ) -> Sequence[_U]:
        assert slice_ is not None
        if is_primitive_type(item_type):
            return cast(
                Sequence[_U],
                self._from_primitive_columns(field_path, item_type, columns, slice_),
            )
        if is_dataclass_type(item_type):
            return cast(
                Sequence[_U],
                self._from_dataclass_columns(field_path, item_type, columns),
            )
        if item_type is np.ndarray:
            return cast(Sequence[_U], self._from_ndarray_columns(field_path, columns))
        if (base_type := typing.get_origin(item_type)) is not None:
            base_type_args = typing.get_args(item_type)
            if base_type is list:
                return cast(
                    Sequence[_U],
                    self._from_generic_list_columns(field_path, base_type_args[0], columns),
                )
            if base_type is tuple:
                return cast(
                    Sequence[_U],
                    self._from_generic_tuple_columns(field_path, base_type_args, columns),
                )
            if base_type is np.ndarray:
                return cast(Sequence[_U], self._from_ndarray_columns(field_path, columns))

        raise NotImplementedError(f"Cannot deserialize columns for field `{field_path}` with type {item_type}")

    def _from_primitive_columns(
        self,
        field_path: str,
        item_type: type[PrimitiveType],
        columns: ArchiveColumns,
        slice_: slice,
    ) -> Sequence[PrimitiveType]:
        return cast(
            Sequence[PrimitiveType],
            [item_type(item) for item in columns[field_path][slice_]],
        )

    def _from_dataclass_columns(
        self,
        field_path: str,
        item_type: type[DataclassProtocol],
        columns: ArchiveColumns,
    ) -> Sequence[DataclassProtocol]:
        deserialized_fields: list[Any] = []
        for field in dataclasses.fields(item_type):
            field_type = field.type
            assert isinstance(field_type, (type, GenericAlias))

            field_key = f"{field_path}.{field.name}"
            field_items = self._from_columns(field_key, field_type, columns)
            deserialized_fields.append(field_items)

        items = [item_type(*fields) for fields in zip(*deserialized_fields)]
        return cast(Sequence[DataclassProtocol], items)

    def _from_generic_list_columns(
        self, field_path: str, item_type: type[_U], columns: ArchiveColumns
    ) -> Sequence[Sequence[_U]]:
        unrolled_items = self._from_columns(f"{field_path}.*", item_type, columns)
        length_cumsum = columns[f"{field_path}.#"]

        ret: list[Sequence[_U]] = []
        for start, end in zip(length_cumsum[:-1], length_cumsum[1:]):
            ret.append(unrolled_items[start:end])
        return ret

    def _from_generic_tuple_columns(
        self, field_path: str, base_type_args: tuple[type, ...], columns: ArchiveColumns
    ) -> Sequence[tuple[_U]]:
        if base_type_args[-1] is Ellipsis:  # type: ignore
            return self._from_generic_list_columns(field_path, base_type_args[0], columns)  # type: ignore

        deserialized_fields: list[Any] = []
        for i, t in enumerate(base_type_args):
            tuple_field_path = f"{field_path}.{i}"
            tuple_field_items = self._from_columns(tuple_field_path, t, columns)  # type: ignore
            deserialized_fields.append(tuple_field_items)

        items = [tuple(fields) for fields in zip(*deserialized_fields)]
        return items  # type: ignore

    def _from_ndarray_columns(self, field_path: str, columns: ArchiveColumns) -> Sequence[np.ndarray[Any, Any]]:
        flat_values = columns[f"{field_path}.*"]
        shapes = self._from_columns(f"{field_path}.#", tuple[int, ...], columns)

        start = 0
        ret: list[np.ndarray[Any, Any]] = []
        for shape in shapes:
            size = np.prod(shape)
            end = start + size
            ret.append(np.reshape(flat_values[start:end], shape))
            start = end
        return ret

    def _get_item(
        self,
        field_path: str,
        item_type: type[T],
        idx: int,
        data: ArchiveMemmap,
        allow_slow: bool = False,
    ) -> T:
        if is_primitive_type(item_type):
            return self._get_primitive_item(field_path, item_type, idx, data)  # type: ignore

        if (base_type := typing.get_origin(item_type)) is not None:
            base_type_args = typing.get_args(item_type)
            if base_type is list:
                return self._get_generic_list_item(field_path, base_type_args[0], columns)
            # if base_type is tuple:
            #     return cast(Sequence[_U], self._from_generic_tuple_columns(field_path, base_type_args, columns))
            # if base_type is np.ndarray:
            #     return cast(Sequence[_U], self._from_ndarray_columns(field_path, columns))

        if allow_slow:
            print("WARNING: Falling back to slow lookup for {fielf_path} with fype {item_type}")
            sequence = self._from_columns(field_path, item_type, data)  # type: ignore
            return sequence[idx]

        raise NotImplementedError(f"Cannot deserialize columns for field `{field_path}` with type {item_type}")

    def _get_primitive_item(
        self,
        field_path: str,
        item_type: type[PrimitiveType],
        idx: int,
        data: ArchiveMemmap,
    ) -> PrimitiveType:
        return item_type(data[field_path][idx])  # type: ignore

    def _get_generic_list_item(
        self, field_path: str, item_type: type[_U], idx: int, data: ArchiveMemmap
    ) -> Sequence[_U]:
        unrolled_items = data[f"{field_path}.*"]
        length_cumsum = data[f"{field_path}.#"]

        return data[field_path][idx]

    # def _from_generic_list_columns(
    #     self, field_path: str, item_type: type[_U], columns: ArchiveColumns
    # ) -> Sequence[Sequence[_U]]:
    #     unrolled_items = self._from_columns(f"{field_path}.*", item_type, columns)
    #     cumsum = columns[f"{field_path}.#"]
    #
    #     ret: list[Sequence[_U]] = []
    #     for start, end in zip(cumsum[:-1], cumsum[1:]):
    #         ret.append(unrolled_items[start:end])
    #     return ret

    # def _from_columns(
    #     self, field_path: str, item_type: type[_U] | GenericAlias, columns: ArchiveColumns
    # ) -> Sequence[_U]:
    #     if is_primitive_type(item_type):
    #         return cast(Sequence[_U], self._from_primitive_columns(field_path, item_type, columns))
    #     if is_dataclass_type(item_type):
    #         return cast(Sequence[_U], self._from_dataclass_columns(field_path, item_type, columns))
    #     if item_type is np.ndarray:
    #         return cast(Sequence[_U], self._from_ndarray_columns(field_path, columns))
    #     if (base_type := typing.get_origin(item_type)) is not None:
    #         base_type_args = typing.get_args(item_type)
    #         if base_type is list:
    #             return cast(Sequence[_U], self._from_generic_list_columns(field_path, base_type_args[0], columns))
    #         if base_type is tuple:
    #             return cast(Sequence[_U], self._from_generic_tuple_columns(field_path, base_type_args, columns))
    #         if base_type is np.ndarray:
    #             return cast(Sequence[_U], self._from_ndarray_columns(field_path, columns))

    #     raise NotImplementedError(f"Cannot deserialize columns for field `{field_path}` with type {item_type}")

    # def _from_primitive_columns(
    #     self, field_path: str, item_type: type[PrimitiveType], columns: ArchiveColumns
    # ) -> Sequence[PrimitiveType]:
    #     return cast(Sequence[PrimitiveType], [item_type(item) for item in columns[field_path]])

    # def _from_dataclass_columns(
    #     self, field_path: str, item_type: type[DataclassProtocol], columns: ArchiveColumns
    # ) -> Sequence[DataclassProtocol]:
    #     deserialized_fields: list[Any] = []
    #     for field in dataclasses.fields(item_type):
    #         field_type = field.type
    #         assert isinstance(field_type, (type, GenericAlias))

    #         field_key = f"{field_path}.{field.name}"
    #         field_items = self._from_columns(field_key, field_type, columns)
    #         deserialized_fields.append(field_items)

    #     items = [item_type(*fields) for fields in zip(*deserialized_fields)]
    #     return cast(Sequence[DataclassProtocol], items)

    # def _from_generic_list_columns(
    #     self, field_path: str, item_type: type[_U], columns: ArchiveColumns
    # ) -> Sequence[Sequence[_U]]:
    #     unrolled_items = self._from_columns(f"{field_path}.*", item_type, columns)
    #     cumsum = columns[f"{field_path}.#"]
    #
    #     ret: list[Sequence[_U]] = []
    #     for start, end in zip(cumsum[:-1], cumsum[1:]):
    #         ret.append(unrolled_items[start:end])
    #     return ret

    # def _from_generic_tuple_columns(
    #     self, field_path: str, base_type_args: tuple[type, ...], columns: ArchiveColumns
    # ) -> Sequence[tuple[_U]]:
    #     if base_type_args[-1] is Ellipsis:  # type: ignore
    #         return self._from_generic_list_columns(field_path, base_type_args[0], columns)  # type: ignore

    #     deserialized_fields: list[Any] = []
    #     for i, t in enumerate(base_type_args):
    #         tuple_field_path = f"{field_path}.{i}"
    #         tuple_field_items = self._from_columns(tuple_field_path, t, columns)  # type: ignore
    #         deserialized_fields.append(tuple_field_items)

    #     items = [tuple(fields) for fields in zip(*deserialized_fields)]
    #     return items  # type: ignore

    # def _from_ndarray_columns(self, field_path: str, columns: ArchiveColumns) -> Sequence[np.ndarray[Any, Any]]:
    #     flat_values = columns[f"{field_path}.*"]
    #     shapes = self._from_columns(f"{field_path}.#", tuple[int, ...], columns)

    #     start = 0
    #     ret: list[np.ndarray[Any, Any]] = []
    #     for shape in shapes:
    #         size = np.prod(shape)
    #         end = start + size
    #         ret.append(np.reshape(flat_values[start:end], shape))
    #         start = end
    #     return ret


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

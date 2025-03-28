"""Archive implementations for storing and loading sequences of data."""

import abc
import collections
import dataclasses
import json
import pathlib
import pickle
import types
import typing
from typing import Any, Iterator, Sequence, cast

import numpy as np

from rgi.core import types as rgi_types
from rgi.core.types import DataclassProtocol, PrimitiveType, TypeOrGeneric

T = typing.TypeVar("T")


ArchiveColumn = np.ndarray[Any, np.dtype[Any]]
ArchiveColumnDict = dict[str, ArchiveColumn]


@dataclasses.dataclass
class NamedColumn:
    """Column of data in an archive."""

    name: str
    data: ArchiveColumn


class Archive(Sequence[T], abc.ABC):
    """Base class for all archives - serializable sequence of items."""


class AppendableArchive(Archive[T]):
    """Archive that supports appending items."""

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

    # overloads to make mypy happy. Actual implemenation in @override method.
    @typing.overload
    def __getitem__(self, idx: int) -> T: ...

    @typing.overload
    def __getitem__(self, idx: slice) -> Sequence[T]: ...

    @typing.override
    def __getitem__(self, idx: int | slice) -> T | Sequence[T]:
        return self._items[idx]

    @typing.override
    def __repr__(self) -> str:
        return f"ListBasedArchive(item_type={self._item_type}, len={len(self)}, items[:1]={self._items[:1]})"  # pylint: disable=line-too-long


class MMapRowArchive(Archive[T]):
    """Archive for reading items from a mmaped numpy file."""

    def __init__(self, path: pathlib.Path | str, item_type: TypeOrGeneric[T]):
        self._item_type = item_type
        self._path = path

        # 1) Load the offsets array
        self._offsets = np.load(f"{path}.index")  # shape = (count+1,)
        self._sequence_length = len(self._offsets) - 1

        # 2) Memory-map the data file
        self._data = np.memmap(path, mode="r")

    def __len__(self) -> int:
        return self._sequence_length

    def _get_item(self, idx: int) -> T:
        start = self._offsets[idx]
        end = self._offsets[idx + 1]
        return typing.cast(T, pickle.loads(self._data[start:end].tobytes()))

    def _get_slice(self, idx: slice) -> Sequence[T]:
        return [self._get_item(i) for i in range(*idx.indices(len(self)))]

    def __iter__(self) -> Iterator[T]:
        for i in range(len(self)):
            yield self._get_item(i)

    @typing.overload
    def __getitem__(self, idx: int) -> T: ...

    @typing.overload
    def __getitem__(self, idx: slice) -> Sequence[T]: ...

    @typing.override
    def __getitem__(self, idx: int | slice) -> T | Sequence[T]:
        if isinstance(idx, slice):
            return self._get_slice(idx)
        # Handle negative indices
        if idx < 0:
            idx += len(self)
        if not 0 <= idx < len(self):
            raise IndexError(f"Index {idx} out of range for archive with {len(self)} items")
        return self._get_slice(slice(idx, idx + 1))[0]


class RowFileArchiver:
    """Class for reading/writing sequences/rows to disk in row based format."""

    def write_items(self, items: typing.Iterable[T], path: pathlib.Path | str) -> None:
        """Save sequence of items to file in row format.

        Write 'objects' to data_filename as raw pickled bytes (one after another).
        Then write offsets into a .npy array (index_filename), where:
        offsets[i] = byte offset in data_filename for the i-th object
        offsets[-1] = final size of the data file
        """

        # 1) Write data
        offsets = [0]
        with open(path, "wb") as fdata:
            for item in items:
                payload = pickle.dumps(item, protocol=5)
                fdata.write(payload)
                offsets.append(fdata.tell())

        # 2) Convert offsets to a NumPy array and save
        #    We'll use uint64 for large file support
        offsets_array = np.array(offsets, dtype=np.uint64)

        # Manually open the file, otherwise np.save appends .npy to filename.
        with open(f"{path}.index", "wb") as f:
            np.save(f, offsets_array)

    def read_items(self, path: pathlib.Path | str, item_type: TypeOrGeneric[T]) -> MMapRowArchive[T]:
        """Read sequence of items from file in row format."""
        return MMapRowArchive(path, item_type)


class SequenceToColumnConverter:
    """Class for converting item Sequences to columns of data."""

    def to_columns(
        self,
        field_path: str,
        item_type: TypeOrGeneric[T],
        items: Sequence[T],
    ) -> Iterator[NamedColumn]:
        """Convert sequence of items to columns of data."""

        if rgi_types.is_primitive_type(item_type):
            return self.to_primitive_columns(field_path, item_type, cast(Sequence[PrimitiveType], items))

        if rgi_types.is_dataclass_type(item_type):
            return self.to_dataclass_columns(field_path, item_type, cast(Sequence[DataclassProtocol], items))

        if item_type is np.ndarray:
            return self.to_ndarray_columns(field_path, cast(Sequence[np.ndarray[Any, Any]], items))

        if (base_type := typing.get_origin(item_type)) is not None:
            base_type_args = typing.get_args(item_type)

            if base_type in (list, collections.abc.Sequence):
                return self.to_generic_list_columns(field_path, base_type_args[0], cast(Sequence[Sequence[Any]], items))
            if base_type is tuple:
                return self.to_generic_tuple_columns(field_path, base_type_args, cast(Sequence[Sequence[Any]], items))
            if base_type is np.ndarray:
                return self.to_ndarray_columns(field_path, cast(Sequence[np.ndarray[Any, Any]], items))
            if rgi_types.is_dataclass_type(base_type):
                return self.to_generic_dataclass_columns(
                    field_path, base_type, base_type_args, cast(Sequence[Any], items)
                )

        raise NotImplementedError(f"Cannot add fields for field `{field_path}` with unhandled type {item_type}")

    def to_primitive_columns(
        self,
        field_path: str,
        item_type: type[PrimitiveType],
        items: Sequence[PrimitiveType],
    ) -> Iterator[NamedColumn]:
        """Serialize primitive types to ndarray."""
        array = np.array(items, dtype=item_type)
        assert array.dtype != np.dtype("O")
        yield NamedColumn(field_path, array)

    def to_dataclass_columns(
        self,
        field_path: str,
        item_type: type[DataclassProtocol],
        items: Sequence[DataclassProtocol],
    ) -> Iterator[NamedColumn]:
        """For dataclass types, which will recursively handle fields of various types."""
        for field in dataclasses.fields(item_type):
            field_type = field.type
            if not rgi_types.is_type_or_generic(field_type):
                raise ValueError(f"Field {field.name} of type {field_type} is not a valid field type in {field_path}")

            field_key = f"{field_path}.{field.name}"
            field_items = [getattr(item, field.name) for item in items]

            yield from self.to_columns(field_key, field_type, field_items)

    def to_ndarray_columns(self, field_path: str, items: Sequence[np.ndarray[Any, Any]]) -> Iterator[NamedColumn]:
        """Convert list of ndarrays to columns of data."""
        flat_values = np.concatenate([arr.flatten() for arr in items])
        shapes = [arr.shape for arr in items]
        size_cumsum = np.cumsum([0] + [np.prod(shape) for shape in shapes])  # type: ignore

        yield NamedColumn(f"{field_path}.*", flat_values)
        yield from self.to_columns(f"{field_path}.#", int, size_cumsum)  # type: ignore
        yield from self.to_columns(f"{field_path}.shape", tuple[int, ...], shapes)

    def to_generic_list_columns(
        self, field_path: str, item_type: type[T], items: Sequence[Sequence[T]]
    ) -> Iterator[NamedColumn]:
        """Convert list of lists to columns of data."""
        unrolled_items = [item for item_list in items for item in item_list]
        length_cumsum = np.cumsum([0] + [len(item_list) for item_list in items])
        yield from self.to_columns(f"{field_path}.*", item_type, unrolled_items)
        yield from self.to_columns(f"{field_path}.#", int, length_cumsum)  # type: ignore

    def to_generic_tuple_columns(
        self,
        field_path: str,
        base_type_args: tuple[type, ...],
        items: Sequence[Sequence[T]],
    ) -> Iterator[NamedColumn]:
        """Convert list of tuples to columns of data."""
        if base_type_args[-1] is Ellipsis:  # type: ignore
            yield from self.to_generic_list_columns(field_path, base_type_args[0], items)
            return

        for i, t in enumerate(base_type_args):
            tuple_field_path = f"{field_path}.{i}"
            tuple_field_items = [item[i] for item in items]
            yield from self.to_columns(tuple_field_path, t, tuple_field_items)

    def to_generic_dataclass_columns(
        self,
        field_path: str,
        base_type: type[DataclassProtocol],
        base_type_args: tuple[type, ...],
        items: Sequence[Any],
    ) -> Iterator[NamedColumn]:
        """For generic dataclass types, recursively handle fields with type parameter substitution."""

        for field in dataclasses.fields(base_type):
            field_type = rgi_types.resolve_type_vars(field.type, base_type, base_type_args)

            field_key = f"{field_path}.{field.name}"
            field_items = [getattr(item, field.name) for item in items]

            yield from self.to_columns(field_key, field_type, field_items)


class ColumnToSequenceConverter:
    """Class for converting columns of data to Sequences of items."""

    def from_columns(
        self,
        field_path: str,
        item_type: TypeOrGeneric[T],
        columns: ArchiveColumnDict,
        slice_: slice,
    ) -> Sequence[T]:
        """Convert columns of data to sequence of items."""
        if slice_ is None:
            raise ValueError("Slice must be provided")

        if rgi_types.is_primitive_type(item_type):
            return cast(
                Sequence[T],
                self.from_primitive_columns(field_path, item_type, columns, slice_),
            )
        if rgi_types.is_dataclass_type(item_type):
            return cast(
                Sequence[T],
                self.from_dataclass_columns(field_path, item_type, columns, slice_),
            )
        if item_type is np.ndarray:
            return cast(Sequence[T], self.from_ndarray_columns(field_path, columns, slice_))
        if (base_type := typing.get_origin(item_type)) is not None:
            base_type_args = typing.get_args(item_type)
            if base_type in (list, collections.abc.Sequence):
                return cast(
                    Sequence[T],
                    self.from_generic_list_columns(field_path, base_type_args[0], columns, slice_),
                )
            if base_type is tuple:
                return cast(
                    Sequence[T],
                    self.from_generic_tuple_columns(field_path, base_type_args, columns, slice_),
                )
            if base_type is np.ndarray:
                return cast(
                    Sequence[T],
                    self.from_ndarray_columns(field_path, columns, slice_),
                )
            if rgi_types.is_dataclass_type(base_type):
                return cast(
                    Sequence[T],
                    self.from_generic_dataclass_columns(field_path, base_type, base_type_args, columns, slice_),
                )

        raise NotImplementedError(f"Cannot deserialize columns for field `{field_path}` with type {item_type}")

    def from_primitive_columns(
        self,
        field_path: str,
        item_type: type[PrimitiveType],
        columns: ArchiveColumnDict,
        slice_: slice,
    ) -> Sequence[PrimitiveType]:
        """Deserialize primitive types from columns."""
        return cast(
            Sequence[PrimitiveType],
            [item_type(item) for item in columns[field_path][slice_]],
        )

    def from_dataclass_columns(
        self,
        field_path: str,
        item_type: type[DataclassProtocol],
        columns: ArchiveColumnDict,
        slice_: slice,
    ) -> Sequence[DataclassProtocol]:
        """Deserialize dataclass types from columns."""
        deserialized_fields: list[Any] = []
        for field in dataclasses.fields(item_type):
            field_type = field.type
            if not rgi_types.is_type_or_generic(field_type):
                raise ValueError(f"Field {field.name} of type {field_type} is not a valid field type in {field_path}")

            field_key = f"{field_path}.{field.name}"
            field_items = self.from_columns(field_key, field_type, columns, slice_)
            deserialized_fields.append(field_items)

        items = [item_type(*fields) for fields in zip(*deserialized_fields)]
        return cast(Sequence[DataclassProtocol], items)

    def from_generic_list_columns(
        self,
        field_path: str,
        item_type: type[T],
        columns: ArchiveColumnDict,
        slice_: slice,
    ) -> Sequence[Sequence[T]]:
        """Deserialize list of lists from columns."""
        if slice_.step is not None and slice_.step != 1:
            raise ValueError("Step must be None or 1 to decode lists")

        length_cumsum = columns[f"{field_path}.#"][slice_.start : slice_.stop + 1]

        unrolled_items: Sequence[T] = self.from_columns(
            f"{field_path}.*",
            item_type,
            columns,
            slice(length_cumsum[0], length_cumsum[-1]),
        )

        offset = length_cumsum[0]
        ret: list[Sequence[T]] = []
        for start, end in zip(length_cumsum[:-1], length_cumsum[1:]):
            ret.append(unrolled_items[start - offset : end - offset])
        return ret

    def from_generic_tuple_columns(
        self,
        field_path: str,
        base_type_args: tuple[type, ...],
        columns: ArchiveColumnDict,
        slice_: slice,
    ) -> Sequence[Any]:
        """Deserialize list of tuples from columns."""
        if base_type_args[-1] is Ellipsis:  # type: ignore
            return self.from_generic_list_columns(field_path, base_type_args[0], columns, slice_)

        deserialized_fields: list[Any] = []
        for i, t in enumerate(base_type_args):
            tuple_field_path = f"{field_path}.{i}"
            tuple_field_items: Sequence[Any] = self.from_columns(tuple_field_path, t, columns, slice_)
            deserialized_fields.append(tuple_field_items)

        items = [tuple(fields) for fields in zip(*deserialized_fields)]
        return items

    def from_ndarray_columns(
        self, field_path: str, columns: ArchiveColumnDict, slice_: slice
    ) -> Sequence[np.ndarray[Any, Any]]:
        """Deserialize list of ndarrays from columns."""
        flat_values = columns[f"{field_path}.*"]
        size_cumsum = columns[f"{field_path}.#"]
        shapes: Sequence[Any] = self.from_columns(f"{field_path}.shape", tuple[int, ...], columns, slice_)

        ret: list[np.ndarray[Any, Any]] = []
        for i, shape in enumerate(shapes):
            start = size_cumsum[slice_.start + i]
            end = size_cumsum[slice_.start + i + 1]
            ret.append(np.reshape(flat_values[start:end], shape))
        return ret

    def from_generic_dataclass_columns(
        self,
        field_path: str,
        base_type: type[DataclassProtocol],
        base_type_args: tuple[type, ...],
        columns: ArchiveColumnDict,
        slice_: slice,
    ) -> Sequence[DataclassProtocol]:
        """Deserialize generic dataclass types from columns."""

        deserialized_fields: list[Any] = []
        for field in dataclasses.fields(base_type):
            field_type = rgi_types.resolve_type_vars(field.type, base_type, base_type_args)

            field_key = f"{field_path}.{field.name}"
            field_items = self.from_columns(field_key, field_type, columns, slice_)
            deserialized_fields.append(field_items)

        items = [base_type(*fields) for fields in zip(*deserialized_fields)]
        return cast(Sequence[DataclassProtocol], items)


class ColumnFileArchiver:
    """Class for readining/writing sequences/columns to disk in column based format."""

    MAGIC = b"RGF"  # 3-byte magic string
    VERSION = b"\x01"  # 1-byte version

    def __init__(self) -> None:
        self._sequence_to_columns_converter = SequenceToColumnConverter()
        self._columns_to_sequence_converter = ColumnToSequenceConverter()

    def write_items(
        self,
        item_type: type[T] | types.GenericAlias,
        items: Sequence[T],
        path: pathlib.Path | str,
    ) -> None:
        """Save sequence of items to file in column format."""
        named_columns = self._sequence_to_columns_converter.to_columns("", item_type, items)
        return self.write_columns(path, named_columns, sequence_length=len(items))

    def write_columns(
        self,
        path: pathlib.Path | str,
        named_columns: Iterator[NamedColumn],
        sequence_length: int,
    ) -> None:
        """Write columns to file."""
        # Open file for writing (fail if exists)
        with open(path, "wb") as f:
            f.write(self.MAGIC + self.VERSION)

            metadata_dict = {
                "column": [],
                "sequence_length": sequence_length,
            }
            # Write columns & create metadata_dict
            for column in named_columns:
                data_offset = f.tell()
                f.write(column.data.tobytes(order="C"))  # Write in C-contiguous order

                # Store info about this array (for the metadata later)
                column_metadata = {
                    "name": column.name,
                    "dtype": str(column.data.dtype),  # e.g. 'float64'
                    "shape": column.data.shape,  # e.g. (10,)
                    "offset": data_offset,  # byte offset
                }
                metadata_dict["column"].append(column_metadata)  # type: ignore

        # Write medatata_dict to a separate index file
        with open(f"{path}.index", "w", encoding="utf-8") as f:
            json.dump(metadata_dict, f, indent=2)

    def read_sequence(self, item_type: type[T] | types.GenericAlias, path: pathlib.Path | str) -> Sequence[T]:
        """Read sequence of items from file."""
        column_dict, sequence_length = self.read_columns(path)
        return self._columns_to_sequence_converter.from_columns(
            "",
            item_type,
            column_dict,
            slice(0, sequence_length),
        )

    def read_columns(self, path: pathlib.Path | str) -> tuple[ArchiveColumnDict, int]:
        """Read columns from file, returns dict & sequence_length"""

        with open(f"{path}.index", "r", encoding="utf-8") as f:
            metadata_dict = json.load(f)

        # Validate header.
        with open(path, "rb") as f:
            header = f.read(4)  # first 4 bytes
            assert header[:3] == self.MAGIC and header[3:] == self.VERSION

        # MMap columns.
        archive_column_dict: ArchiveColumnDict = {}
        for column_metadata in metadata_dict["column"]:
            # Note: mm objects have no close() method, so we just rely on GC to close the file.
            mm = np.memmap(
                path,
                mode="r",  # read-only
                offset=column_metadata["offset"],
                shape=tuple(column_metadata["shape"]),
                dtype=np.dtype(column_metadata["dtype"]),
                order="C",
            )
            archive_column_dict[column_metadata["name"]] = mm

        return archive_column_dict, metadata_dict["sequence_length"]


class MMapColumnArchive(Archive[T]):
    """Read-only archive storing items in a mmaped numpy file."""

    def __init__(self, path: pathlib.Path | str, item_type: TypeOrGeneric[T]):
        """Initialize archive from file.

        Args:
            file: File to load archive from
            item_type: Type of items stored in archive
        """
        self._item_type = item_type
        self._path = path

        archiver = ColumnFileArchiver()
        column_dict, sequence_length = archiver.read_columns(path)
        self._column_to_sequence_converter = ColumnToSequenceConverter()

        self._column_dict = column_dict
        self._sequence_length = sequence_length

    @typing.override
    def __len__(self) -> int:
        return self._sequence_length

    def _get_slice(self, idx: slice) -> Sequence[T]:
        return self._column_to_sequence_converter.from_columns("", self._item_type, self._column_dict, idx)

    @typing.overload
    def __getitem__(self, idx: int) -> T: ...

    @typing.overload
    def __getitem__(self, idx: slice) -> Sequence[T]: ...

    @typing.override
    def __getitem__(self, idx: int | slice) -> T | Sequence[T]:
        if isinstance(idx, slice):
            return self._get_slice(idx)

        # get single item
        # Handle negative indices
        if idx < 0:
            idx += len(self)
        if not 0 <= idx < len(self):
            raise IndexError(f"Index {idx} out of range for archive with {len(self)} items")
        return self._get_slice(slice(idx, idx + 1))[0]

    @typing.override
    def __iter__(self) -> typing.Iterator[T]:
        for i in range(len(self)):
            yield self[i]


class CombinedArchive(Archive[T]):
    """Archive combining multiple archives into a single view."""

    def __init__(self, archives: list[Archive[T]]):
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
        return int(self._cumsum[-1])

    def _get_item(self, idx: int) -> T:
        archive, local_idx = self._locate(idx)
        return archive[local_idx]

    def _get_slice(self, _slice: slice) -> Sequence[T]:
        return [self._get_item(i) for i in range(*_slice.indices(len(self)))]

    @typing.overload
    def __getitem__(self, idx: int) -> T: ...

    @typing.overload
    def __getitem__(self, idx: slice) -> Sequence[T]: ...

    def __getitem__(self, idx: int | slice) -> T | Sequence[T]:
        if isinstance(idx, slice):
            return self._get_slice(idx)

        # get single item
        # Handle negative indices
        if idx < 0:
            idx += len(self)
        if not 0 <= idx < len(self):
            raise IndexError(f"Index {idx} out of range for archive with {len(self)} items")

        return self._get_item(idx)

    def __iter__(self) -> Iterator[T]:
        for archive in self._archives:
            yield from archive

"""Tests for archive implementations."""

import pathlib
import tempfile
from dataclasses import dataclass, fields
from typing import Type, TypeVar, Any, Callable
import typing
import numpy as np
import pytest

from numpy.testing import assert_equal

from rgi.core.archive import (
    ListBasedArchive,
    ColumnFileArchiver,
    MMapColumnArchive,
    RowFileArchiver,
    MMapRowArchive,
    CombinedArchive,
)

T = TypeVar("T")


@typing.dataclass_transform()
def dataclass_with_np_eq(*args: Any, **kwargs: Any) -> Callable[[Type[T]], Type[T]]:
    """
    Decorator that defines a class as a dataclass with numpy-aware equality.
    """
    if args and isinstance(args[0], type):
        raise TypeError(
            "dataclass_with_np_eq must be called with parentheses. "
            "Use @dataclass_with_np_eq() instead of @dataclass_with_np_eq"
        )

    def wrapper(cls: Type[T]) -> Type[T]:
        kwargs_copy = {**kwargs, "eq": False}
        cls = dataclass(**kwargs_copy)(cls)

        def __eq__(self: T, other: object) -> bool:
            if not isinstance(other, type(self)):
                return False
            for field in fields(cls):  # type: ignore
                self_val = getattr(self, field.name)
                other_val = getattr(other, field.name)
                if isinstance(self_val, np.ndarray):
                    if not np.array_equal(self_val, other_val):
                        return False
                elif self_val != other_val:
                    return False
            return True

        setattr(cls, "__eq__", __eq__)
        return cls

    return wrapper


@dataclass
class SimpleData:
    x: int
    y: float
    name: str


@dataclass_with_np_eq()
class NestedData:
    simple: SimpleData
    values: list[int]
    points: tuple[float, float, float]
    matrix: np.ndarray[Any, np.dtype[np.float64]]


_SAMPLE_SIMPLE_DATA = [
    SimpleData(x=1, y=2.0, name="test"),
    SimpleData(x=3, y=4.0, name="test2"),
    SimpleData(x=5, y=6.0, name="test3"),
    SimpleData(x=7, y=8.0, name="test4"),
    SimpleData(x=9, y=10.0, name="test5"),
    SimpleData(x=11, y=12.0, name="test6"),
    SimpleData(x=13, y=14.0, name="test7"),
    SimpleData(x=15, y=16.0, name="test8"),
    SimpleData(x=17, y=18.0, name="test9"),
    SimpleData(x=19, y=20.0, name="test10"),
]

_SAMPLE_NESTED_DATA = [
    NestedData(
        simple=SimpleData(x=1, y=2.0, name="test"),
        values=[1, 2, 3],
        points=(1.0, 2.0, 3.0),
        matrix=np.array([[1, 2], [3, 4]]),
    ),
    NestedData(
        simple=SimpleData(x=5, y=6.0, name="test2"),
        values=[4, 5, 6, 8, 9, 10, 11],
        points=(4.0, 5.0, 6.0),
        matrix=np.array([[5, 6, 7], [8, 9, 10], [11, 12, 13], [14, 15, 16]]),
    ),
    NestedData(
        simple=SimpleData(x=9, y=10.0, name="test3"),
        values=[8, 9, 10, 12, 13, 14, 15, 16, 17, 18],
        points=(8.0, 9.0, 10.0),
        matrix=np.array([[9, 10, 11], [12, 13, 14], [15, 16, 17], [18, 19, 20]]),
    ),
]

_SAMPLE_NDARRAY_DATA = [
    np.array([[1, 2], [3, 4]]),
    np.array([[5, 6, 7], [8, 9, 10], [11, 12, 13], [14, 15, 16]]),
    np.array([[9, 10, 11], [12, 13, 14], [15, 16, 17], [18, 19, 20]]),
]


def test_list_based_archive() -> None:
    archive = ListBasedArchive(SimpleData)
    items = [
        SimpleData(x=1, y=2.0, name="test"),
        SimpleData(x=3, y=4.0, name="test2"),
        SimpleData(x=5, y=6.0, name="test3"),
    ]

    for i, item in enumerate(items):
        archive.append(item)

        assert len(archive) == i + 1
        assert archive[0] == items[0]
        assert archive[: i + 1] == items[: i + 1]


@pytest.mark.parametrize(
    "item_type, items",
    [
        pytest.param(int, [10, 20, 30, 40, 50, 60, 70, 80, 90, 100], id="int"),
        pytest.param(list[int], [[10, 20], [30, 40, 50], [60, 70, 80, 90, 100]], id="list[int]"),
        pytest.param(list[list[int]], [[[10, 20], [30, 40, 50]], [[60, 70, 80, 90, 100]]], id="list[list[int]]"),
        pytest.param(tuple[int, float, str], [(1, 2.0, "a"), (3, 4.0, "b")], id="tuple[int, float, str]"),
        pytest.param(str, ["foo", "bar", "baz", "", "", "longer_string", "x"], id="str"),
        pytest.param(np.ndarray, _SAMPLE_NDARRAY_DATA, id="ndarray"),
        pytest.param(SimpleData, _SAMPLE_SIMPLE_DATA, id="simple"),
        pytest.param(NestedData, _SAMPLE_NESTED_DATA, id="nested"),
    ],
)
def test_row_based_archive(tmp_path: pathlib.Path, item_type: Type[T], items: list[T]) -> None:
    archiver = RowFileArchiver()
    path = tmp_path / "test.rgr"
    archiver.write_items(items, path)
    row_archive = archiver.read_items(path, item_type)

    assert len(row_archive) == len(items)
    assert list(row_archive) == items
    assert row_archive[0] == items[0]
    assert row_archive[0:2] == items[0:2]
    assert row_archive[-1] == items[-1]
    assert row_archive[-1:3:-2] == items[-1:3:-2]

    # Test out of bounds
    with pytest.raises(IndexError):
        _ = row_archive[len(items)]
    with pytest.raises(IndexError):
        _ = row_archive[-len(items) - 1]


@pytest.mark.parametrize(
    "item_type, items",
    [
        pytest.param(int, [10, 20, 30, 40, 50, 60, 70, 80, 90, 100], id="int"),
        pytest.param(list[int], [[10, 20], [30, 40, 50], [60, 70, 80, 90, 100]], id="list[int]"),
        pytest.param(list[list[int]], [[[10, 20], [30, 40, 50]], [[60, 70, 80, 90, 100]]], id="list[list[int]]"),
        pytest.param(tuple[int, float, str], [(1, 2.0, "a"), (3, 4.0, "b")], id="tuple[int, float, str]"),
        pytest.param(str, ["foo", "bar", "baz", "", "", "longer_string", "x"], id="str"),
        pytest.param(np.ndarray, _SAMPLE_NDARRAY_DATA, id="ndarray"),
        pytest.param(SimpleData, _SAMPLE_SIMPLE_DATA, id="simple"),
        pytest.param(NestedData, _SAMPLE_NESTED_DATA, id="nested"),
    ],
)
def test_column_based_archive_sequence(tmp_path: pathlib.Path, item_type: Type[T], items: list[T]) -> None:
    archiver = ColumnFileArchiver()
    path = pathlib.Path(tmp_path) / "test.col"
    archiver.write_sequence(item_type, items, path)
    reloaded = archiver.read_sequence(item_type, path)

    assert len(reloaded) == len(items)
    assert list(reloaded) == items
    assert reloaded[0] == items[0]
    assert reloaded[0:2] == items[0:2]
    assert reloaded[-1] == items[-1]
    assert reloaded[-1:3:-2] == items[-1:3:-2]


@pytest.mark.parametrize(
    "item_type, items",
    [
        pytest.param(int, [10, 20, 30, 40, 50, 60, 70, 80, 90, 100], id="int"),
        pytest.param(list[int], [[10, 20], [30, 40, 50], [60, 70, 80, 90, 100]], id="list[int]"),
        pytest.param(list[list[int]], [[[10, 20], [30, 40, 50]], [[60, 70, 80, 90, 100]]], id="list[list[int]]"),
        pytest.param(tuple[int, float, str], [(1, 2.0, "a"), (3, 4.0, "b")], id="tuple[int, float, str]"),
        pytest.param(str, ["foo", "bar", "baz", "", "", "longer_string", "x"], id="str"),
        pytest.param(np.ndarray, _SAMPLE_NDARRAY_DATA, id="ndarray"),
        pytest.param(SimpleData, _SAMPLE_SIMPLE_DATA, id="simple"),
        pytest.param(NestedData, _SAMPLE_NESTED_DATA, id="nested"),
    ],
)
def test_mmap_column(tmp_path: pathlib.Path, item_type: Type[T], items: list[T]) -> None:
    archiver = ColumnFileArchiver()
    path = pathlib.Path(tmp_path) / "test.col"
    archiver.write_sequence(item_type, items, path)
    column_archive = MMapColumnArchive(path, item_type)

    assert len(column_archive) == len(items)
    assert list(column_archive) == items
    assert column_archive[0] == items[0]
    assert column_archive[0:2] == items[0:2]
    assert column_archive[-1] == items[-1]
    ## Not supported for lists, etc.
    if item_type not in [list[int], list[list[int]], np.ndarray, NestedData]:
        assert column_archive[-1:3:-2] == items[-1:3:-2]


# def test_invalid_type() -> None:
#     class NonDataclass:
#         pass

#     serializer = ArchiveSerializer(NonDataclass)
#     with pytest.raises(NotImplementedError):
#         serializer.to_columns([NonDataclass()])


# def test_mmap_primitive_type() -> None:
#     serializer = ArchiveSerializer(int)
#     original = [10, 20, 30]
#     serializer.save(original, "test.npz")
#     with serializer.load_mmap("test.npz") as archive:
#         for idx, item in enumerate(original):
#             assert archive[idx] == item


# def test_mmap_primitive_list_type() -> None:
#     serializer = ArchiveSerializer(list[int])
#     original = [[10, 20], [30, 40, 50]]
#     serializer.save(original, "test.npz")
#     with serializer.load_mmap("test.npz") as archive:
#         for idx, item in enumerate(original):
#             assert archive[idx] == item


# def test_mmap_archive_file_not_found(tmp_path: pathlib.Path) -> None:
#     """Test MMapColumnArchive handles missing files."""
#     serializer = ArchiveSerializer(SimpleData)
#     file_path = tmp_path / "nonexistent.npz"

#     with pytest.raises(FileNotFoundError):
#         serializer.load_mmap(file_path)


# def test_column_file_archiver(tmp_path: pathlib.Path) -> None:
#     """Test ColumnFileArchiver write and read."""
#     archiver = ColumnFileArchiver()
#     data = [SimpleData(x=1, y=2.0, name="test"), SimpleData(x=3, y=4.0, name="test2")]
#     path = tmp_path / "test.col"

#     archiver.write_sequence(SimpleData, data, path)
#     result = archiver.read_sequence(SimpleData, path)

#     assert len(result) == len(data)
#     assert list(result) == data


# def test_column_file_archiver_nested(tmp_path: pathlib.Path) -> None:
#     """Test ColumnFileArchiver with nested data."""
#     archiver = ColumnFileArchiver()
#     data = [
#         NestedData(
#             simple=SimpleData(x=1, y=2.0, name="test"),
#             values=[1, 2, 3],
#             points=(1.0, 2.0, 3.0),
#             matrix=np.array([[1, 2], [3, 4]]),
#         )
#     ]
#     path = tmp_path / "test_nested.col"

#     archiver.write_sequence(NestedData, data, path)
#     result = archiver.read_sequence(NestedData, path)

#     assert len(result) == len(data)
#     assert_equal(list(result), data)


# def test_mmap_column_archive(tmp_path: pathlib.Path) -> None:
#     """Test MMapColumnArchive functionality."""
#     # First write data using ColumnFileArchiver
#     archiver = ColumnFileArchiver()
#     data = [SimpleData(x=1, y=2.0, name="test"), SimpleData(x=3, y=4.0, name="test2")]
#     path = tmp_path / "test.col"
#     archiver.write_sequence(SimpleData, data, path)

#     # Now read using MMapColumnArchive
#     archive = MMapColumnArchive(path, SimpleData)
#     assert len(archive) == len(data)
#     assert list(archive) == data
#     assert archive[1] == data[1]
#     assert archive[0:2] == data[0:2]


# def test_row_file_archiver(tmp_path: pathlib.Path) -> None:
#     """Test RowFileArchiver write and read."""
#     archiver = RowFileArchiver()
#     data = [SimpleData(x=1, y=2.0, name="test"), SimpleData(x=3, y=4.0, name="test2")]
#     path = tmp_path / "test.row"

#     archiver.write_items(data, path)
#     archive = archiver.read_items(path, SimpleData)

#     assert len(archive) == len(data)
#     assert list(archive) == data


# def test_mmap_row_archive(tmp_path: pathlib.Path) -> None:
#     """Test MMapRowArchive functionality."""
#     # First write data using RowFileArchiver
#     archiver = RowFileArchiver()
#     data = [SimpleData(x=1, y=2.0, name="test"), SimpleData(x=3, y=4.0, name="test2")]
#     path = tmp_path / "test.row"
#     archiver.write_items(data, path)

#     # Now read using MMapRowArchive
#     archive = MMapRowArchive(path, SimpleData)
#     assert len(archive) == len(data)
#     assert list(archive) == data
#     assert archive[1] == data[1]
#     assert archive[0:2] == data[0:2]


# def test_combined_archive() -> None:
#     """Test CombinedArchive functionality."""
#     archive1 = ListBasedArchive(SimpleData)
#     archive2 = ListBasedArchive(SimpleData)

#     data1 = [SimpleData(x=1, y=2.0, name="test1")]
#     data2 = [SimpleData(x=2, y=3.0, name="test2")]

#     archive1.append(data1[0])
#     archive2.append(data2[0])

#     combined = CombinedArchive([archive1, archive2])

#     assert len(combined) == 2
#     assert list(combined) == data1 + data2
#     assert combined[1] == data2[0]
#     assert combined[0:2] == data1 + data2


# def test_combined_archive_empty() -> None:
#     """Test CombinedArchive with empty archives."""
#     archive1 = ListBasedArchive(SimpleData)
#     archive2 = ListBasedArchive(SimpleData)

#     combined = CombinedArchive([archive1, archive2])

#     assert len(combined) == 0
#     assert list(combined) == []


# def test_combined_archive_index_error() -> None:
#     """Test CombinedArchive index error handling."""
#     archive = ListBasedArchive(SimpleData)
#     archive.append(SimpleData(x=1, y=2.0, name="test"))

#     combined = CombinedArchive([archive])

#     with pytest.raises(IndexError):
#         _ = combined[1]

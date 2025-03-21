"""Tests for archive implementations."""

import pathlib
import typing
from dataclasses import dataclass
from typing import Any, Type, TypeVar

import numpy as np
import pytest

from rgi.core.archive import (
    ColumnFileArchiver,
    CombinedArchive,
    ListBasedArchive,
    MMapColumnArchive,
    MMapRowArchive,
    RowFileArchiver,
)
from rgi.core.utils import dataclass_with_np_eq

T = TypeVar("T")


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


T1 = typing.TypeVar("T1")
T2 = typing.TypeVar("T2")


@dataclass_with_np_eq()
class GenericData(typing.Generic[T1, T2]):
    x: T1
    y: T2


SAMPLE_SIMPLE_DATA = [
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

SAMPLE_NESTED_DATA = [
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

SAMPLE_NDARRAY_DATA = [
    np.array([[1, 2], [3, 4]]),
    np.array([[5, 6, 7], [8, 9, 10], [11, 12, 13], [14, 15, 16]]),
    np.array([[9, 10, 11], [12, 13, 14], [15, 16, 17], [18, 19, 20]]),
]


SAMPLE_GENERIC_INT_FLOAT_DATA: list[GenericData[int, float]] = [
    GenericData(x=1, y=2.0),
    GenericData(x=3, y=4.0),
    GenericData(x=5, y=6.0),
]

SAMPLE_GENERIC_INT_LIST_DATA: list[GenericData[int, list[int]]] = [
    GenericData(x=1, y=[1, 2, 3]),
    GenericData(x=3, y=[4, 5, 6, 7, 8, 9, 10]),
    GenericData(x=5, y=[11, 12, 13, 14, 15, 16, 17, 18, 19, 20]),
]


def assert_item_equal(item_type: Type[T], item: T, expected: T) -> None:
    if item_type == np.ndarray:
        assert np.array_equal(item, expected)  # type: ignore
    else:
        assert item == expected


def assert_sequence_equal(item_type: type[T], items: typing.Sequence[T], expected: typing.Sequence[T]) -> None:
    if item_type == np.ndarray:
        assert all(np.array_equal(a, b) for a, b in zip(items, expected, strict=True))  # type: ignore
    else:
        assert list(items) == expected


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
        pytest.param(np.ndarray, SAMPLE_NDARRAY_DATA, id="ndarray"),
        pytest.param(SimpleData, SAMPLE_SIMPLE_DATA, id="simple"),
        pytest.param(NestedData, SAMPLE_NESTED_DATA, id="nested"),
        pytest.param(GenericData[int, float], SAMPLE_GENERIC_INT_FLOAT_DATA, id="generic_int_float"),
        pytest.param(GenericData[int, list[int]], SAMPLE_GENERIC_INT_LIST_DATA, id="generic_int_list"),
    ],
)
def test_row_based_archive(tmp_path: pathlib.Path, item_type: Type[T], items: list[T]) -> None:
    archiver = RowFileArchiver()
    path = tmp_path / "test.rgr"
    archiver.write_items(items, path)
    row_archive: MMapRowArchive[T] = archiver.read_items(path, item_type)

    assert len(row_archive) == len(items)
    assert_sequence_equal(item_type, list(row_archive), items)
    assert_item_equal(item_type, row_archive[0], items[0])
    assert_sequence_equal(item_type, row_archive[0:2], items[0:2])
    assert_item_equal(item_type, row_archive[-1], items[-1])
    assert_sequence_equal(item_type, row_archive[-1:3:-2], items[-1:3:-2])

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
        pytest.param(np.ndarray, SAMPLE_NDARRAY_DATA, id="ndarray"),
        pytest.param(SimpleData, SAMPLE_SIMPLE_DATA, id="simple"),
        pytest.param(NestedData, SAMPLE_NESTED_DATA, id="nested"),
        pytest.param(GenericData[int, float], SAMPLE_GENERIC_INT_FLOAT_DATA, id="generic_int_float"),
        pytest.param(GenericData[int, list[int]], SAMPLE_GENERIC_INT_LIST_DATA, id="generic_int_list"),
    ],
)
def test_column_based_archive_sequence(tmp_path: pathlib.Path, item_type: Type[T], items: list[T]) -> None:
    archiver = ColumnFileArchiver()
    path = pathlib.Path(tmp_path) / "test.col"
    archiver.write_items(item_type, items, path)
    reloaded = archiver.read_sequence(item_type, path)

    assert len(reloaded) == len(items)
    assert_sequence_equal(item_type, list(reloaded), items)
    assert_item_equal(item_type, reloaded[0], items[0])
    assert_sequence_equal(item_type, reloaded[0:2], items[0:2])
    assert_item_equal(item_type, reloaded[-1], items[-1])
    assert_sequence_equal(item_type, reloaded[-1:3:-2], items[-1:3:-2])


@pytest.mark.parametrize(
    "item_type, items",
    [
        pytest.param(int, [10, 20, 30, 40, 50, 60, 70, 80, 90, 100], id="int"),
        pytest.param(list[int], [[10, 20], [30, 40, 50], [60, 70, 80, 90, 100]], id="list[int]"),
        pytest.param(list[list[int]], [[[10, 20], [30, 40, 50]], [[60, 70, 80, 90, 100]]], id="list[list[int]]"),
        pytest.param(tuple[int, float, str], [(1, 2.0, "a"), (3, 4.0, "b")], id="tuple[int, float, str]"),
        pytest.param(str, ["foo", "bar", "baz", "", "", "longer_string", "x"], id="str"),
        pytest.param(np.ndarray, SAMPLE_NDARRAY_DATA, id="ndarray"),
        pytest.param(SimpleData, SAMPLE_SIMPLE_DATA, id="simple"),
        pytest.param(NestedData, SAMPLE_NESTED_DATA, id="nested"),
        pytest.param(GenericData[int, float], SAMPLE_GENERIC_INT_FLOAT_DATA, id="generic_int_float"),
        pytest.param(GenericData[int, list[int]], SAMPLE_GENERIC_INT_LIST_DATA, id="generic_int_list"),
    ],
)
def test_mmap_column(tmp_path: pathlib.Path, item_type: Type[T], items: list[T]) -> None:
    archiver = ColumnFileArchiver()
    path = pathlib.Path(tmp_path) / "test.col"
    archiver.write_items(item_type, items, path)
    column_archive: MMapColumnArchive[T] = MMapColumnArchive(path, item_type)

    assert len(column_archive) == len(items)
    assert_sequence_equal(item_type, list(column_archive), items)
    assert_item_equal(item_type, column_archive[0], items[0])
    assert_sequence_equal(item_type, column_archive[0:2], items[0:2])
    assert_item_equal(item_type, column_archive[-1], items[-1])
    ## Not supported for lists, etc.
    if item_type not in [list[int], list[list[int]], np.ndarray, NestedData, GenericData[int, list[int]]]:
        assert column_archive[-1:3:-2] == items[-1:3:-2]


def test_invalid_column_type(tmp_path: pathlib.Path) -> None:
    class NonDataclass:
        pass

    archiver = ColumnFileArchiver()
    with pytest.raises(NotImplementedError):
        archiver.write_items(NonDataclass, [], tmp_path / "test.col")


@pytest.mark.parametrize(
    "item_type, items",
    [
        pytest.param(int, [10, 20, 30, 40, 50, 60, 70, 80, 90, 100], id="int"),
        pytest.param(list[int], [[10, 20], [30, 40, 50], [60, 70, 80, 90, 100]], id="list[int]"),
        pytest.param(list[list[int]], [[[10, 20], [30, 40, 50]], [[60, 70, 80, 90, 100]]], id="list[list[int]]"),
        pytest.param(tuple[int, float, str], [(1, 2.0, "a"), (3, 4.0, "b")], id="tuple[int, float, str]"),
        pytest.param(str, ["foo", "bar", "baz", "", "", "longer_string", "x"], id="str"),
        pytest.param(np.ndarray, SAMPLE_NDARRAY_DATA, id="ndarray"),
        pytest.param(SimpleData, SAMPLE_SIMPLE_DATA, id="simple"),
        pytest.param(NestedData, SAMPLE_NESTED_DATA, id="nested"),
        pytest.param(GenericData[int, float], SAMPLE_GENERIC_INT_FLOAT_DATA, id="generic_int_float"),
        pytest.param(GenericData[int, list[int]], SAMPLE_GENERIC_INT_LIST_DATA, id="generic_int_list"),
    ],
)
def test_combined_archive(tmp_path: pathlib.Path, item_type: Type[T], items: list[T]) -> None:
    """Test CombinedArchive functionality."""

    list_items = items[::]
    list_archive = ListBasedArchive(item_type)
    for item in list_items:
        list_archive.append(item)

    row_items = items[::-1]
    row_archiver = RowFileArchiver()
    row_path = tmp_path / "test.rgr"
    row_archiver.write_items(row_items, row_path)
    row_archive: MMapRowArchive[T] = row_archiver.read_items(row_path, item_type)

    column_items = items[1::]
    column_archiver = ColumnFileArchiver()
    column_path = tmp_path / "test.col"
    column_archiver.write_items(item_type, column_items, column_path)
    column_archive: MMapColumnArchive[T] = MMapColumnArchive(column_path, item_type)

    combined_items = list_items + row_items + column_items
    combined_archive = CombinedArchive([list_archive, row_archive, column_archive])

    assert len(combined_archive) == len(combined_items)
    assert_sequence_equal(item_type, list(combined_archive), combined_items)
    assert_item_equal(item_type, combined_archive[0], combined_items[0])
    assert_sequence_equal(item_type, combined_archive[0:2], combined_items[0:2])
    assert_item_equal(item_type, combined_archive[-1], combined_items[-1])
    ## Not supported for lists, etc.
    if item_type not in [list[int], list[list[int]], np.ndarray, NestedData]:
        assert combined_archive[-1:3:-2] == combined_items[-1:3:-2]

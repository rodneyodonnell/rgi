"""Tests for archive implementations."""

import pathlib
from dataclasses import dataclass
from typing import Any
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


@dataclass
class SimpleData:
    x: int
    y: float
    name: str


@dataclass
class NestedData:
    simple: SimpleData
    values: list[int]
    points: tuple[float, float, float]
    matrix: np.ndarray[Any, np.dtype[np.float64]]

    def __eq__(self, other: Any) -> bool:
        if not isinstance(other, NestedData):
            return False
        return (
            self.simple == other.simple
            and self.values == other.values
            and self.points == other.points
            and np.array_equal(self.matrix, other.matrix)
        )


def test_primitive_serialization() -> None:
    serializer = ArchiveSerializer(int)
    result = serializer.to_columns([1, 2, 3])
    assert isinstance(result[""], np.ndarray)
    np.testing.assert_array_equal(result[""], np.array([1, 2, 3]))


def test_list_serialization() -> None:
    serializer = ArchiveSerializer(list[int])
    original = [[1, 2], [3, 4, 5]]
    serialized = serializer.to_columns(original)

    np.testing.assert_array_equal(serialized[".*"], np.array([1, 2, 3, 4, 5]))
    np.testing.assert_array_equal(serialized[".#"], np.array([0, 2, 5]))

    deserialized = serializer.from_columns(serialized)
    assert deserialized == original


def test_dataclass_serialization() -> None:
    data = [SimpleData(x=1, y=2.0, name="test"), SimpleData(x=3, y=4.0, name="test2")]
    serializer = ArchiveSerializer(SimpleData)
    result = serializer.to_columns(data)

    np.testing.assert_array_equal(result[".x"], np.array([1, 3]))
    np.testing.assert_array_equal(result[".y"], np.array([2.0, 4.0]))
    np.testing.assert_array_equal(result[".name"], np.array(["test", "test2"]))


def test_nested_dataclass_serialization() -> None:
    original = [
        NestedData(
            simple=SimpleData(x=1, y=2.0, name="test"),
            values=[1, 2, 3],
            points=(1.0, 2.0, 3.0),
            matrix=np.array([[1, 2], [3, 4]]),
        )
    ]
    serializer = ArchiveSerializer(NestedData)
    serialized = serializer.to_columns(original)

    np.testing.assert_array_equal(serialized[".simple.x"], np.array([1]))
    np.testing.assert_array_equal(serialized[".values.*"], np.array([1, 2, 3]))
    np.testing.assert_array_equal(serialized[".values.#"], np.array([0, 3]))
    np.testing.assert_array_equal(serialized[".points.0"], np.array([1.0]))
    np.testing.assert_array_equal(serialized[".matrix.*"], np.array([1, 2, 3, 4]))
    np.testing.assert_array_equal(serialized[".matrix.#.*"], np.array([2, 2]))
    np.testing.assert_array_equal(serialized[".matrix.#.#"], np.array([0, 2]))

    deserialized = serializer.from_columns(serialized)
    assert deserialized == original


def test_tuple_serialization() -> None:
    serializer = ArchiveSerializer(tuple[int, float, str])
    result = serializer.to_columns([(1, 2.0, "a"), (3, 4.0, "b")])

    np.testing.assert_array_equal(result[".0"], np.array([1, 3]))
    np.testing.assert_array_equal(result[".1"], np.array([2.0, 4.0]))
    np.testing.assert_array_equal(result[".2"], np.array(["a", "b"]))


def test_ndarray_serialization() -> None:
    serializer = ArchiveSerializer(np.ndarray)
    original = [np.array([[1, 2], [3, 4]]), np.array([[5, 6], [7, 8]])]
    serialized = serializer.to_columns(original)

    np.testing.assert_array_equal(serialized[".*"], np.array([1, 2, 3, 4, 5, 6, 7, 8]))
    np.testing.assert_array_equal(serialized[".#.*"], np.array([2, 2, 2, 2]))
    np.testing.assert_array_equal(serialized[".#.#"], np.array([0, 2, 4]))

    deserialized = serializer.from_columns(serialized)
    np.testing.assert_array_equal(deserialized, original)


def test_invalid_type() -> None:
    class NonDataclass:
        pass

    serializer = ArchiveSerializer(NonDataclass)
    with pytest.raises(NotImplementedError):
        serializer.to_columns([NonDataclass()])


def test_file_io(tmp_path: pathlib.Path) -> None:
    """Test to_file and from_file methods."""
    # Create test data and serialize to columns
    data = [SimpleData(x=1, y=2.0, name="test"), SimpleData(x=3, y=4.0, name="test2")]
    serializer = ArchiveSerializer(SimpleData)
    columns = serializer.to_columns(data)

    # Save columns to file
    file_path = tmp_path / "test_archive.npz"
    serializer.to_file(columns, file_path)

    # Load columns from file
    loaded_columns = serializer.from_file(file_path)

    # Verify columns match
    assert set(columns.keys()) == set(loaded_columns.keys())
    for key in columns:
        np.testing.assert_array_equal(columns[key], loaded_columns[key])


def test_file_io_with_empty_columns(tmp_path: pathlib.Path) -> None:
    """Test handling of empty columns during file I/O."""
    serializer = ArchiveSerializer(SimpleData)
    empty_columns: dict[str, np.ndarray[Any, Any]] = {}

    file_path = tmp_path / "test_empty_archive.npz"

    serializer.to_file(empty_columns, file_path)
    loaded_columns = serializer.from_file(file_path)

    assert len(empty_columns) == 0
    assert len(loaded_columns) == 0


def test_simple_save_and_load_archive() -> None:
    data = [SimpleData(x=1, y=2.0, name="test"), SimpleData(x=3, y=4.0, name="test2")]
    serializer = ArchiveSerializer(SimpleData)
    serializer.save(data, "test.npz")
    loaded_data = serializer.load_sequence("test.npz")
    assert len(loaded_data) == len(data)
    assert all(isinstance(item, SimpleData) for item in loaded_data)
    assert all(
        item.x == data[i].x and item.y == data[i].y and item.name == data[i].name for i, item in enumerate(loaded_data)
    )


def test_list_save_and_load_archive() -> None:
    data = [[10, 20, 30], [40, 50, 60, 70]]
    serializer = ArchiveSerializer(list[int])
    serializer.save(data, "test.npz")
    loaded_data = serializer.load_sequence("test.npz")
    assert len(loaded_data) == len(data)
    assert data == loaded_data


def test_nested_list_save_and_load_archive() -> None:
    data = [[[10], [20, 30]], [[40], [50], [60, 70, 80]]]
    serializer = ArchiveSerializer(list[list[int]])
    serializer.save(data, "test.npz")
    loaded_data = serializer.load_sequence("test.npz")
    assert len(loaded_data) == len(data)
    assert data == loaded_data


def test_tuple_save_and_load_archive() -> None:
    data = [(1, [2, 3], "a"), (4, [5, 6, 7], "b")]
    serializer = ArchiveSerializer(tuple[int, list[int], str])
    serializer.save(data, "test.npz")
    loaded_data = serializer.load_sequence("test.npz")
    assert len(loaded_data) == len(data)
    assert data == loaded_data


def test_nested_data_save_and_load_archive() -> None:
    data = [
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
    ]
    serializer = ArchiveSerializer(NestedData)
    serializer.save(data, "test.npz")
    loaded_data = serializer.load_sequence("test.npz")
    assert len(loaded_data) == len(data)
    assert_equal(data, loaded_data)


def test_mmap_archive_context_manager(tmp_path: pathlib.Path) -> None:
    """Test MMapColumnArchive context manager behavior."""
    # Create test data and save to file
    data = [SimpleData(x=1, y=2.0, name="test"), SimpleData(x=3, y=4.0, name="test2")]
    serializer = ArchiveSerializer(SimpleData)
    file_path = tmp_path / "test_mmap.npz"
    serializer.save(data, file_path)

    # Test context manager
    with serializer.load_mmap(file_path) as archive:
        item = serializer._get_item("", SimpleData, 0, archive._data)
        assert item == data[0]


def test_mmap_archive_multiple_opens(tmp_path: pathlib.Path) -> None:
    """Test MMapColumnArchive can be opened multiple times."""
    data = [SimpleData(x=1, y=2.0, name="test"), SimpleData(x=3, y=4.0, name="test2")]
    serializer = ArchiveSerializer(SimpleData)
    file_path = tmp_path / "test_mmap.npz"
    serializer.save(data, file_path)

    # Open archive multiple times
    with serializer.load_mmap(file_path) as archive1, serializer.load_mmap(file_path) as archive2:
        # We can access the '.x' column while file files are open.
        assert archive1._data[".x"] is not None
        assert archive2._data[".x"] is not None
        assert len(archive1) == len(archive2)

    # Verify files are closed by checking we can't access data anymore
    with pytest.raises(AttributeError):
        _ = archive1._data[".x"]
    with pytest.raises(AttributeError):
        _ = archive2._data[".x"]


def test_mmap_primitive_type() -> None:
    serializer = ArchiveSerializer(int)
    original = [10, 20, 30]
    serializer.save(original, "test.npz")
    with serializer.load_mmap("test.npz") as archive:
        for idx, item in enumerate(original):
            assert archive[idx] == item


def test_mmap_primitive_list_type() -> None:
    serializer = ArchiveSerializer(list[int])
    original = [[10, 20], [30, 40, 50]]
    serializer.save(original, "test.npz")
    with serializer.load_mmap("test.npz") as archive:
        for idx, item in enumerate(original):
            assert archive[idx] == item


def test_mmap_archive_file_not_found(tmp_path: pathlib.Path) -> None:
    """Test MMapColumnArchive handles missing files."""
    serializer = ArchiveSerializer(SimpleData)
    file_path = tmp_path / "nonexistent.npz"

    with pytest.raises(FileNotFoundError):
        serializer.load_mmap(file_path)


def test_list_based_archive() -> None:
    """Test basic ListBasedArchive functionality."""
    archive = ListBasedArchive(SimpleData)
    data = SimpleData(x=1, y=2.0, name="test")
    archive.append(data)

    assert len(archive) == 1
    assert archive[0] == data
    assert list(archive) == [data]


def test_list_based_archive_slice() -> None:
    """Test ListBasedArchive slicing."""
    archive = ListBasedArchive(SimpleData)
    data = [SimpleData(x=i, y=float(i), name=f"test{i}") for i in range(3)]
    for item in data:
        archive.append(item)

    assert archive[1:] == data[1:]
    assert archive[:2] == data[:2]


def test_column_file_archiver(tmp_path: pathlib.Path) -> None:
    """Test ColumnFileArchiver write and read."""
    archiver = ColumnFileArchiver()
    data = [SimpleData(x=1, y=2.0, name="test"), SimpleData(x=3, y=4.0, name="test2")]
    path = tmp_path / "test.col"

    archiver.write_sequence(SimpleData, data, path)
    result = archiver.read_sequence(SimpleData, path)

    assert len(result) == len(data)
    assert list(result) == data


def test_column_file_archiver_nested(tmp_path: pathlib.Path) -> None:
    """Test ColumnFileArchiver with nested data."""
    archiver = ColumnFileArchiver()
    data = [
        NestedData(
            simple=SimpleData(x=1, y=2.0, name="test"),
            values=[1, 2, 3],
            points=(1.0, 2.0, 3.0),
            matrix=np.array([[1, 2], [3, 4]]),
        )
    ]
    path = tmp_path / "test_nested.col"

    archiver.write_sequence(NestedData, data, path)
    result = archiver.read_sequence(NestedData, path)

    assert len(result) == len(data)
    assert_equal(list(result), data)


def test_mmap_column_archive(tmp_path: pathlib.Path) -> None:
    """Test MMapColumnArchive functionality."""
    # First write data using ColumnFileArchiver
    archiver = ColumnFileArchiver()
    data = [SimpleData(x=1, y=2.0, name="test"), SimpleData(x=3, y=4.0, name="test2")]
    path = tmp_path / "test.col"
    archiver.write_sequence(SimpleData, data, path)

    # Now read using MMapColumnArchive
    archive = MMapColumnArchive(path, SimpleData)
    assert len(archive) == len(data)
    assert list(archive) == data
    assert archive[1] == data[1]
    assert archive[0:2] == data[0:2]


def test_row_file_archiver(tmp_path: pathlib.Path) -> None:
    """Test RowFileArchiver write and read."""
    archiver = RowFileArchiver()
    data = [SimpleData(x=1, y=2.0, name="test"), SimpleData(x=3, y=4.0, name="test2")]
    path = tmp_path / "test.row"

    archiver.write_items(data, path)
    archive = archiver.read_items(path, SimpleData)

    assert len(archive) == len(data)
    assert list(archive) == data


def test_mmap_row_archive(tmp_path: pathlib.Path) -> None:
    """Test MMapRowArchive functionality."""
    # First write data using RowFileArchiver
    archiver = RowFileArchiver()
    data = [SimpleData(x=1, y=2.0, name="test"), SimpleData(x=3, y=4.0, name="test2")]
    path = tmp_path / "test.row"
    archiver.write_items(data, path)

    # Now read using MMapRowArchive
    archive = MMapRowArchive(path, SimpleData)
    assert len(archive) == len(data)
    assert list(archive) == data
    assert archive[1] == data[1]
    assert archive[0:2] == data[0:2]


def test_combined_archive() -> None:
    """Test CombinedArchive functionality."""
    archive1 = ListBasedArchive(SimpleData)
    archive2 = ListBasedArchive(SimpleData)

    data1 = [SimpleData(x=1, y=2.0, name="test1")]
    data2 = [SimpleData(x=2, y=3.0, name="test2")]

    archive1.append(data1[0])
    archive2.append(data2[0])

    combined = CombinedArchive([archive1, archive2])

    assert len(combined) == 2
    assert list(combined) == data1 + data2
    assert combined[1] == data2[0]
    assert combined[0:2] == data1 + data2


def test_combined_archive_empty() -> None:
    """Test CombinedArchive with empty archives."""
    archive1 = ListBasedArchive(SimpleData)
    archive2 = ListBasedArchive(SimpleData)

    combined = CombinedArchive([archive1, archive2])

    assert len(combined) == 0
    assert list(combined) == []


def test_combined_archive_index_error() -> None:
    """Test CombinedArchive index error handling."""
    archive = ListBasedArchive(SimpleData)
    archive.append(SimpleData(x=1, y=2.0, name="test"))

    combined = CombinedArchive([archive])

    with pytest.raises(IndexError):
        _ = combined[1]

import pathlib
from dataclasses import dataclass
from typing import Any
import numpy as np
import pytest
import contextlib

from numpy.testing import assert_equal

from rgi.core.archive import ArchiveSerializer


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


def test_dataclass_serialization() -> None:
    data = [SimpleData(x=1, y=2.0, name="test"), SimpleData(x=3, y=4.0, name="test2")]
    serializer = ArchiveSerializer(SimpleData)
    result = serializer.to_columns(data)

    np.testing.assert_array_equal(result[".x"], np.array([1, 3]))
    np.testing.assert_array_equal(result[".y"], np.array([2.0, 4.0]))
    np.testing.assert_array_equal(result[".name"], np.array(["test", "test2"]))


def test_nested_dataclass_serialization() -> None:
    data = [
        NestedData(
            simple=SimpleData(x=1, y=2.0, name="test"),
            values=[1, 2, 3],
            points=(1.0, 2.0, 3.0),
            matrix=np.array([[1, 2], [3, 4]]),
        )
    ]
    serializer = ArchiveSerializer(NestedData)
    result = serializer.to_columns(data)

    np.testing.assert_array_equal(result[".simple.x"], np.array([1]))
    np.testing.assert_array_equal(result[".values.*"], np.array([1, 2, 3]))
    np.testing.assert_array_equal(result[".values.#"], np.array([3]))
    np.testing.assert_array_equal(result[".points.0"], np.array([1.0]))
    np.testing.assert_array_equal(result[".matrix.*"], np.array([1, 2, 3, 4]))
    np.testing.assert_array_equal(result[".matrix.#.*"], np.array([2, 2]))
    np.testing.assert_array_equal(result[".matrix.#.#"], np.array([2]))


def test_list_serialization() -> None:
    serializer = ArchiveSerializer(list[int])
    result = serializer.to_columns([[1, 2], [3, 4, 5]])

    np.testing.assert_array_equal(result[".*"], np.array([1, 2, 3, 4, 5]))
    np.testing.assert_array_equal(result[".#"], np.array([2, 3]))


def test_tuple_serialization() -> None:
    serializer = ArchiveSerializer(tuple[int, float, str])
    result = serializer.to_columns([(1, 2.0, "a"), (3, 4.0, "b")])

    np.testing.assert_array_equal(result[".0"], np.array([1, 3]))
    np.testing.assert_array_equal(result[".1"], np.array([2.0, 4.0]))
    np.testing.assert_array_equal(result[".2"], np.array(["a", "b"]))


def test_ndarray_serialization() -> None:
    serializer = ArchiveSerializer(np.ndarray)
    data = [np.array([[1, 2], [3, 4]]), np.array([[5, 6], [7, 8]])]
    result = serializer.to_columns(data)

    np.testing.assert_array_equal(result[".*"], np.array([1, 2, 3, 4, 5, 6, 7, 8]))
    np.testing.assert_array_equal(result[".#.*"], np.array([2, 2, 2, 2]))
    np.testing.assert_array_equal(result[".#.#"], np.array([2, 2]))


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
    """Test MMappedArchive context manager behavior."""
    # Create test data and save to file
    data = [SimpleData(x=1, y=2.0, name="test"), SimpleData(x=3, y=4.0, name="test2")]
    serializer = ArchiveSerializer(SimpleData)
    file_path = tmp_path / "test_mmap.npz"
    serializer.save(data, file_path)

    # Test context manager
    with serializer.load_mmap(file_path) as archive:
        # assert len(archive) == len(data)
        # Note: __getitem__ is not implemented yet, so we can't check contents
        assert data[0] == archive[0]


def test_mmap_archive_multiple_opens(tmp_path: pathlib.Path) -> None:
    """Test MMappedArchive can be opened multiple times."""
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


def test_mmap_archive_file_not_found(tmp_path: pathlib.Path) -> None:
    """Test MMappedArchive handles missing files."""
    serializer = ArchiveSerializer(SimpleData)
    file_path = tmp_path / "nonexistent.npz"

    with pytest.raises(FileNotFoundError):
        serializer.load_mmap(file_path)

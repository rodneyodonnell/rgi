from dataclasses import dataclass
from typing import Any
import numpy as np
import pytest

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


def test_primitive_serialization() -> None:
    serializer = ArchiveSerializer(int)
    result = serializer.serialize_to_dict([1, 2, 3])
    assert isinstance(result[""], np.ndarray)
    np.testing.assert_array_equal(result[""], np.array([1, 2, 3]))


def test_dataclass_serialization() -> None:
    data = [SimpleData(x=1, y=2.0, name="test"), SimpleData(x=3, y=4.0, name="test2")]
    serializer = ArchiveSerializer(SimpleData)
    result = serializer.serialize_to_dict(data)

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
    result = serializer.serialize_to_dict(data)

    np.testing.assert_array_equal(result[".simple.x"], np.array([1]))
    np.testing.assert_array_equal(result[".values.*"], np.array([1, 2, 3]))
    np.testing.assert_array_equal(result[".values.#"], np.array([3]))
    np.testing.assert_array_equal(result[".points.0"], np.array([1.0]))
    np.testing.assert_array_equal(result[".matrix.*"], np.array([1, 2, 3, 4]))
    np.testing.assert_array_equal(result[".matrix.#.*"], np.array([2, 2]))
    np.testing.assert_array_equal(result[".matrix.#.#"], np.array([2]))


def test_list_serialization() -> None:
    serializer = ArchiveSerializer(list[int])
    result = serializer.serialize_to_dict([[1, 2], [3, 4, 5]])

    np.testing.assert_array_equal(result[".*"], np.array([1, 2, 3, 4, 5]))
    np.testing.assert_array_equal(result[".#"], np.array([2, 3]))


def test_tuple_serialization() -> None:
    serializer = ArchiveSerializer(tuple[int, float, str])
    result = serializer.serialize_to_dict([(1, 2.0, "a"), (3, 4.0, "b")])

    np.testing.assert_array_equal(result[".0"], np.array([1, 3]))
    np.testing.assert_array_equal(result[".1"], np.array([2.0, 4.0]))
    np.testing.assert_array_equal(result[".2"], np.array(["a", "b"]))


def test_ndarray_serialization() -> None:
    serializer = ArchiveSerializer(np.ndarray)
    data = [np.array([[1, 2], [3, 4]]), np.array([[5, 6], [7, 8]])]
    result = serializer.serialize_to_dict(data)

    np.testing.assert_array_equal(result[".*"], np.array([1, 2, 3, 4, 5, 6, 7, 8]))
    np.testing.assert_array_equal(result[".#.*"], np.array([2, 2, 2, 2]))
    np.testing.assert_array_equal(result[".#.#"], np.array([2, 2]))


def test_invalid_type() -> None:
    class NonDataclass:
        pass

    serializer = ArchiveSerializer(NonDataclass)
    with pytest.raises(NotImplementedError):
        serializer.serialize_to_dict([NonDataclass()])

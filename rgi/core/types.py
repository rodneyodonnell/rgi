import pathlib
import typing
import dataclasses

from typing import Any, TypeGuard, Protocol

FileOrPath = str | pathlib.Path | typing.IO[typing.Any]
PrimitiveType = int | float | str | bool


@dataclasses.dataclass
class DataclassProtocol(Protocol):
    """Dataclass Protocol to keep type checker and dataclasses.is_dataclass() fn happy."""


def is_primitive_type(t: Any) -> TypeGuard[type[PrimitiveType]]:
    return t in (int, float, str, bool)


def is_dataclass_type(t: Any) -> TypeGuard[type[DataclassProtocol]]:
    return isinstance(t, type) and dataclasses.is_dataclass(t)

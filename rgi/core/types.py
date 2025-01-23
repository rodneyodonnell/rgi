import pathlib
import typing
import dataclasses

from typing import Any, TypeGuard, Protocol, TypeVar

T = TypeVar("T")  # We want invariant T for most types
T_co = TypeVar("T_co", covariant=True)  # protocol definiton required covariant.

FileOrPath = str | pathlib.Path | typing.IO[typing.Any]
PrimitiveType = int | float | str | bool


@dataclasses.dataclass
class DataclassProtocol(Protocol):
    """Dataclass Protocol to keep type checker and dataclasses.is_dataclass() fn happy."""


class GenericType(Protocol[T_co]):
    """Protocol for generic types - used for static type checking only."""


def is_generic_type(t: Any) -> TypeGuard[GenericType[Any]]:
    """Runtime check for generic types."""
    return typing.get_origin(t) is not None


TypeOrGeneric = type[T] | GenericType[T]


def is_type_or_generic(t: Any) -> TypeGuard[TypeOrGeneric[Any]]:
    return isinstance(t, type) or is_generic_type(t)


def is_primitive_type(t: Any) -> TypeGuard[type[PrimitiveType]]:
    return t in (int, float, str, bool)


def is_dataclass_type(t: Any) -> TypeGuard[type[DataclassProtocol]]:
    return isinstance(t, type) and dataclasses.is_dataclass(t)

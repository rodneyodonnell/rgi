import dataclasses
import pathlib
import typing
from typing import Any, Protocol, TypeGuard, TypeVar

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


def resolve_type_vars(field_type: Any, base_type: type, type_args: tuple[type, ...]) -> Any:
    """Recursively resolve TypeVars in a type to their concrete types.

    Args:
        field_type: Type to resolve, potentially containing TypeVars
        base_type: Generic base type containing the TypeVar parameters
        type_args: Concrete types to substitute for the TypeVars

    Returns:
        Resolved type with all TypeVars replaced with concrete types

    Raises:
        ValueError: If a TypeVar cannot be matched to a concrete type
    """
    # Direct TypeVar
    if isinstance(field_type, typing.TypeVar):
        type_var_name = field_type.__name__
        for i, param in enumerate(base_type.__parameters__):  # type: ignore
            if param.__name__ == type_var_name:
                return type_args[i]
        raise ValueError(f"Could not find type argument for TypeVar {type_var_name}")

    # Generic type with potential TypeVar args
    if origin := typing.get_origin(field_type):
        resolved_args = tuple(resolve_type_vars(arg, base_type, type_args) for arg in typing.get_args(field_type))
        return origin[resolved_args]

    return field_type

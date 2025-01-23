import typing
from typing import Any, Callable, Type, TypeVar
import dataclasses

import numpy as np

T = TypeVar("T")


@typing.dataclass_transform()
def dataclass_with_np_eq(*args: Any, **kwargs: Any) -> Callable[[Type[T]], Type[T]]:
    """
    Decorator that defines a class as a dataclass with numpy-aware equality.

    Usage:
    @dataclass_with_np_eq()
    class MyClass:
        x: int
        y: np.ndarray[Any, np.dtype[np.float64]]
    """
    if args and isinstance(args[0], type):
        raise TypeError(
            "dataclass_with_np_eq must be called with parentheses. "
            "Use @dataclass_with_np_eq() instead of @dataclass_with_np_eq"
        )

    def wrapper(cls: Type[T]) -> Type[T]:
        cls = dataclasses.dataclass(**{**kwargs, "eq": False})(cls)

        def __eq__(self: T, other: object) -> bool:
            if not isinstance(other, type(self)):
                return False
            for field in dataclasses.fields(cls):  # type: ignore
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

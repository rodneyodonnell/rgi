import dataclasses
from typing import Any, Iterator, Literal, Protocol, Sequence, TypeGuard, override, runtime_checkable

import numpy as np

from rgi.core import base

PlayerState = Literal[None]


@runtime_checkable
class IndexableAndIterable(Protocol):
    """Utility protocol which matches lists, np.ndarrays, and other sequences.

    This is useful as `isinstance(np.ndarray([1,2,3]), Sequence) == False`.
    """

    def __getitem__(self, index: int) -> Any: ...
    def __len__(self) -> int: ...
    def __iter__(self) -> Iterator[Any]: ...


class PresetPlayer(base.Player[base.TGameState, PlayerState, base.TAction]):
    """Player that selects a pre-determined sequence of actions or action ids."""

    def __init__(
        self,
        *,
        actions: Sequence[base.TAction] | None = None,
        action_ids: Sequence[int] | None = None,
    ):
        self.actions = actions
        self.action_ids = action_ids
        self.idx = 0

    @override
    def select_action(self, game_state: base.TGameState, legal_actions: Sequence[base.TAction]) -> base.TAction:
        self.idx += 1
        if self.actions:
            return self.actions[self.idx - 1]
        if self.action_ids:
            return legal_actions[self.action_ids[self.idx - 1]]
        raise ValueError("No actions or action ids provided.")


class EqualityChecker:
    """Helper class to check equality of two objects containg dataclasses, lists, nparrays, etc."""

    def __init__(self, check_full_list: bool = False):
        self.check_full_list = check_full_list
        self.errors: list[tuple[str, str]] = []

    def print_errors(self) -> None:
        for path, message in self.errors:
            print(f"{path}: {message}")

    # Use Sized as ndarray isn't a Sequence.
    def is_sequence(self, obj: Any) -> TypeGuard[IndexableAndIterable]:
        return isinstance(obj, IndexableAndIterable)

    def is_primitive(self, obj: Any) -> bool:
        return isinstance(obj, (int, float, bool, np.number))

    def _check_equality_dataclass(self, obj1: Any, obj2: Any, path: str) -> bool:
        if not isinstance(obj1, type(obj2)):
            self.errors.append((path, f"dataclass: {type(obj1)} != {type(obj2)}"))
            return False
        all_ok = True
        for field_name in obj1.__dataclass_fields__:
            val1 = getattr(obj1, field_name)
            val2 = getattr(obj2, field_name)
            if not self.check_equality(val1, val2, path=f"{path}/{field_name}"):
                all_ok = False
        return all_ok

    def _check_equality_sequence(self, seq1: IndexableAndIterable, seq2: IndexableAndIterable, path: str) -> bool:
        if len(seq1) != len(seq2):
            self.errors.append((path, f"length mismatch: {len(seq1)} != {len(seq2)}"))
            return False
        all_ok = True
        for i, (val1, val2) in enumerate(zip(seq1, seq2)):
            if not self.check_equality(val1, val2, path=f"{path}/[{i}]"):
                all_ok = False
            if not self.check_full_list:
                break
        return all_ok

    def check_equality(self, obj1: Any, obj2: Any, path: str = "/") -> bool:
        if dataclasses.is_dataclass(obj1) and dataclasses.is_dataclass(obj2):
            return self._check_equality_dataclass(obj1, obj2, path)
        elif self.is_sequence(obj1) and self.is_sequence(obj2):
            return self._check_equality_sequence(obj1, obj2, path)
        else:
            if obj1 != obj2:
                self.errors.append((path, f"value mismatch: {obj1} != {obj2}"))
                return False
        return True

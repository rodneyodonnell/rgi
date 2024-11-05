# rgi/tests/core/test_trajectory.py

from dataclasses import dataclass

import pytest
import torch

from rgi.core.base import Batch, PrimitiveBatch


@dataclass
class DummyState:
    value: int
    flag: bool


@dataclass
class DummyBatch(Batch[DummyState]):
    value: torch.Tensor
    flag: torch.Tensor


def test_batch() -> None:
    states = [DummyState(1, True), DummyState(2, False), DummyState(3, True)]
    batch = DummyBatch.from_sequence(states)

    assert len(batch) == 3
    assert torch.equal(batch.value, torch.tensor([1, 2, 3]))
    assert torch.equal(batch.flag, torch.tensor([True, False, True]))
    assert batch[0] == DummyState(1, True)
    assert batch[1] == DummyState(2, False)
    assert batch[2] == DummyState(3, True)
    assert [batch[i] for i in range(len(batch))] == states


def test_batch_empty_sequence() -> None:
    with pytest.raises(ValueError, match="Cannot create a batch from an empty sequence"):
        DummyBatch.from_sequence([])


def test_primitive_batch() -> None:
    actions = [1, 2, 3]
    batch = PrimitiveBatch.from_sequence(actions)

    assert len(batch) == 3
    assert torch.equal(batch.values, torch.tensor([1, 2, 3]))
    assert batch[0] == 1
    assert batch[1] == 2
    assert batch[2] == 3
    assert [batch[i] for i in range(len(batch))] == actions


def test_primitive_batch_empty_sequence() -> None:
    with pytest.raises(ValueError, match="Cannot create a batch from an empty sequence"):
        PrimitiveBatch.from_sequence([])


def test_primitive_batch_different_types() -> None:
    mixed_data = [1, 2.0, "3"]
    with pytest.raises(TypeError):
        PrimitiveBatch.from_sequence(mixed_data)

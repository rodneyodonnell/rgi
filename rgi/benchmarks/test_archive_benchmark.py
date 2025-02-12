"""Benchmark tests for archive implementations."""

import dataclasses
import random
import statistics
import time
from pathlib import Path
from typing import Any, Callable, Sequence, TypeVar

import numpy as np
import pytest
from pytest_benchmark.fixture import BenchmarkFixture

from rgi.core.archive import Archive, ColumnFileArchiver, MMapColumnArchive, MMapRowArchive, RowFileArchiver
from rgi.tests.core.test_archive import SAMPLE_NESTED_DATA, NestedData

T = TypeVar("T")


def generate_benchmark_data(num_items: int) -> list[NestedData]:
    """Generate random benchmark data."""
    return [SAMPLE_NESTED_DATA[i % len(SAMPLE_NESTED_DATA)] for i in range(num_items)]


def write_data(items: list[NestedData], archive_type: str, path: Path) -> None:
    """Write data to a file."""
    if archive_type == "row":
        row_archiver = RowFileArchiver()
        row_archiver.write_items(items, path)
    else:
        col_archiver = ColumnFileArchiver()
        col_archiver.write_items(NestedData, items, path)


@pytest.mark.benchmark(group="archive_comparison")
@pytest.mark.parametrize("archive_type", ["row", "column"])
@pytest.mark.parametrize("num_items", [100, 1000])
def test_mmap_write_performance(
    benchmark: BenchmarkFixture,
    tmp_path: Path,
    archive_type: str,
    num_items: int,
) -> None:
    """Benchmark the write performance of MMapRowArchive and MMapColumnArchive."""

    # Generate test data
    print(f"\nGenerating {num_items} test items...")
    items = generate_benchmark_data(num_items)

    def write_fn() -> None:
        path = tmp_path / f"benchmark_{num_items}"
        write_data(items, archive_type, path)

    benchmark.pedantic(write_fn, iterations=10, rounds=5)


@pytest.mark.parametrize("archive_type", ["row", "column"])
@pytest.mark.parametrize("read_seek_type", ["sequential", "random", "field"])
@pytest.mark.parametrize("num_items", [100, 1000])
def test_mmap_read_performance(
    benchmark: BenchmarkFixture,
    tmp_path: Path,
    archive_type: str,
    read_seek_type: str,
    num_items: int,
) -> None:
    """Benchmark the write performance of MMapRowArchive and MMapColumnArchive."""

    # Generate test data
    print(f"\nGenerating {num_items} test items...")
    items = generate_benchmark_data(num_items)
    path = tmp_path / f"benchmark_{num_items}"
    write_data(items, archive_type, path)

    archive: Archive[NestedData]
    if archive_type == "row":
        archive = RowFileArchiver().read_items(path, NestedData)
    else:
        archive = MMapColumnArchive(path, NestedData)

    def benchmark_sequential_access() -> list[NestedData]:
        """Sequential access benchmark."""
        return [archive[i] for i in range(min(1000, len(archive)))]

    def benchmark_random_access() -> list[NestedData]:
        """Random access benchmark."""
        indices = list(range(len(archive)))
        random.shuffle(indices)
        return [archive[i] for i in indices[:1000]]  # Read 1000 random items

    def benchmark_field_access() -> list[float]:
        """Access specific fields benchmark."""
        return [archive[i].simple.y for i in range(min(1000, len(archive)))]

    bench_fn: Callable[[], list[Any]]
    if read_seek_type == "sequential":
        bench_fn = benchmark_sequential_access
    elif read_seek_type == "random":
        bench_fn = benchmark_random_access
    elif read_seek_type == "field":
        bench_fn = benchmark_field_access

    benchmark.pedantic(bench_fn, iterations=10, rounds=5)

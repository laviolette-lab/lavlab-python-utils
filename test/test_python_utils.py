# pylint: skip-file
# type: ignore
import numpy as np
import pytest
from lavlab.python_util import (
    chunkify,
    interlace_lists,
    rgba_to_uint,
    uint_to_rgba,
    is_memsafe_array,
    create_array,
    merge_async_iters,
    desync,
)

import lavlab


@pytest.fixture
def ctx():
    lavlab.ctx.resources.memory_usage = 1
    lavlab.ctx.resources.max_memory = 1024**2 * 3
    return lavlab.ctx


# Tests for Memory Safe Array Check
def test_is_memsafe_array(ctx):
    assert is_memsafe_array((100, 100), np.float64) == True  # Should be safe
    assert is_memsafe_array((10000, 10000), np.float64) == False  # Should exceed 1GB


# Tests for Array Creation
def test_create_array(ctx):
    array = create_array((10, 10), np.float64)
    assert array.shape == (10, 10)
    assert array.dtype == np.float64
    assert isinstance(array, np.ndarray)

    large_array = create_array((10000, 10000), np.float64)
    assert isinstance(
        large_array, np.memmap
    )  # Should create a disk-based array for large sizes


# Tests for Chunkify Function
def test_chunkify():
    lst = [1, 2, 3, 4, 5]
    chunks = chunkify(lst, 2)
    assert chunks == [[1, 2, 3], [4, 5]]  # Proper splitting into chunks

    chunks = chunkify(lst, 5)
    assert chunks == [[1], [2], [3], [4], [5]]  # One item per chunk


# Tests for Interlacing Lists
def test_interlace_lists():
    result = interlace_lists([1, 3], [2, 4])
    assert result == [1, 2, 3, 4]
    result = interlace_lists([1, 3, 5], [2, 4])
    assert result == [1, 2, 3, 4, 5]  # Handles uneven list lengths


# Asynchronous Tests
@pytest.mark.asyncio
async def test_merge_async_iters():
    async def async_gen(items):
        for item in items:
            yield item

    merged = merge_async_iters(async_gen([1, 3]), async_gen([2, 4]))
    results = [item async for item in merged]
    assert sorted(results) == [1, 2, 3, 4], "Should contain all items"


@pytest.mark.asyncio
async def test_desync():
    gen = desync([1, 2, 3])
    results = [item async for item in gen]
    assert results == [1, 2, 3]


# Tests for Color Utilities
def test_rgba_to_uint():
    assert rgba_to_uint(255, 0, 0, 255) == -16776961  # Test RGBA to uint conversion


def test_uint_to_rgba():
    red, green, blue, alpha = uint_to_rgba(-16776961)
    assert (red, green, blue, alpha) == (255, 0, 0, 255)  # Test uint to RGBA conversion

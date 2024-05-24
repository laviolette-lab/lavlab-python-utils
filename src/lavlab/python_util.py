"""General Python Utilities"""

from __future__ import annotations

import asyncio
import tempfile
from itertools import zip_longest
from math import ceil
from typing import AsyncGenerator

import numpy as np
import pyvips as pv  # type: ignore

import lavlab

LOGGER = lavlab.LOGGER.getChild("python_util")


def is_memsafe_array(shape: tuple[int, ...], dtype=np.float64) -> bool:
    """
    Checks if a desired array of given shape and datatype is too large for the memory constraints

    Parameters
    ----------
    shape : tuple[int,...]
        shape of the array
    dtype : np.dtype, optional
        datatype of given array, by default np.float64 for max safety

    Returns
    -------
    bool
        True if the array is safe to create in memory, False if it will blow your pc up.
    """
    size = np.prod(shape) * np.dtype(dtype).itemsize
    return size < lavlab.ctx.resources.max_memory


def is_memsafe_pvimg(pv_img: pv.Image) -> bool:
    """
    Checks if a given pyvips image is too large for the memory constraints

    Parameters
    ----------
    pv_img : pv.Image
        pyvips image to check

    Returns
    -------
    bool
        True if the image is safe to create in memory, False if it will blow your pc up.
    """
    # assume dtype is 64-bit float atm
    size = pv_img.width * pv_img.height * pv_img.bands * 64
    return size < lavlab.ctx.resources.max_memory


def is_storage_safe_img(shape: tuple[int, ...], dtype=np.float64) -> bool:
    """
    Checks if a desired array of given shape and datatype is too large for the storage constraints

    Parameters
    ----------
    shape : tuple[int,...]
        shape of the array
    dtype : np.dtype, optional
        datatype of given array, by default np.float64 for max safety

    Returns
    -------
    bool
        True if the array is safe to create in memory, False if it will blow your pc up.
    """
    size = np.prod(shape) * np.dtype(dtype).itemsize
    return size < lavlab.ctx.resources.max_temp_storage


def create_array(shape: tuple[int, ...], dtype=np.float64) -> np.ndarray:
    """
    Creates an in-memory array or a disk-based memmap array based on the available system memory.

    Parameters
    ----------
    shape : tuple
        Shape of the array.
    dtype : np.dtype, Default: np.float64
        Data-type of the array's elements.

    Returns
    -------
    array
        Numpy in-memory array or memmap array based on the available system memory.
    """
    if is_memsafe_array(shape, dtype):
        return np.zeros(shape, dtype)
    if not is_storage_safe_img(shape, dtype):
        raise MemoryError(
            f"Array of shape {shape} with dtype {dtype} is too large for storage."
        )
    path = tempfile.mkstemp(dir=lavlab.ctx.temp_dir)[1]
    return np.memmap(path, dtype=dtype, mode="w+", shape=shape)


def chunkify(lst: list, n: int) -> list[list]:
    """
    Breaks list into n chunks.

    Parameters
    ----------
    lst: list
        List to chunkify.
    n: int
        Number of lists to make

    Returns
    -------
    list[list*n]
        lst split into n chunks.
    """
    size = ceil(len(lst) / n)
    return list(map(lambda x: lst[x * size : x * size + size], list(range(n))))


def interlace_lists(*lists: list) -> list:
    """
    Interlaces a list of lists. Useful for combining tileLists of different channels.

    Parameters
    ----------
    *lists: list
        lists to merge.

    Returns
    -------
    list
        Merged list.

    Examples
    --------
    >>> interlace_lists([1,3],[2,4])
    [1,2,3,4]
    """
    return [
        val
        for tup in zip_longest(*lists, fillvalue=None)
        for val in tup
        if val is not None
    ]


def merge_async_iters(*a_iters) -> AsyncGenerator:
    """
    Merges async generators using a asyncio.Queue.

    Notes
    -----
    Code from: https://stackoverflow.com/a/55317623

    Parameters
    ----------
    *a_iters: AsyncGenerator
        AsyncGenerators to merge

    Returns
    -------
    AsyncGenerator
        Generator that calls all input generators
    """
    queue = asyncio.Queue(1)  # type: ignore
    run_count = len(a_iters)
    cancelling = False

    async def drain(a_iter):
        nonlocal run_count
        try:
            async for item in a_iter:
                await queue.put((False, item))
        except IOError as e:
            if not cancelling:
                await queue.put((True, e))
            else:
                raise
        finally:
            run_count -= 1

    async def merged():
        try:
            while run_count:
                raised, next_item = await queue.get()
                if raised:
                    cancel_tasks()
                    raise next_item
                yield next_item
        finally:
            cancel_tasks()

    def cancel_tasks():
        nonlocal cancelling
        cancelling = True
        for t in tasks:
            t.cancel()

    tasks = [asyncio.create_task(drain(a_iter)) for a_iter in a_iters]
    return merged()


async def desync(it) -> AsyncGenerator:
    """
    Turns sync iterable into an async iterable.

    Parameters
    ----------
    it: Iterable
        Synchronous iterable-like object (can be used in for loop)

    Returns
    -------
    AsyncGenerator
        asynchronously yields results from input iterable."""
    for x in it:
        yield x


def rgba_to_uint(red: int, green: int, blue: int, alpha=255) -> int:
    """
    Return the color as an Integer in RGBA encoding.

    Parameters
    ----------
    red: int
        Red color val (0-255)
    green: int
        Green color val (0-255)
    blue: int
        Blue color val (0-255)
    alpha: int
        Alpha opacity val (0-255)

    Returns
    -------
    int
        Integer encoding rgba value.
    """
    r = red << 24
    g = green << 16
    b = blue << 8
    a = alpha
    uint = r + g + b + a
    if uint > (2**31 - 1):  # convert to signed 32-bit int
        uint = uint - 2**32
    return int(uint)


def uint_to_rgba(uint: int) -> tuple[int, int, int, int]:
    """
    Return the color as an Integer in RGBA encoding.

    Parameters
    ----------
    int
        Integer encoding rgba value.

    Returns
    -------
    red: int
        Red color val (0-255)
    green: int
        Green color val (0-255)
    blue: int
        Blue color val (0-255)
    alpha: int
        Alpha opacity val (0-255)"""
    if uint < 0:  # convert from signed 32-bit int
        uint = uint + 2**32

    red = (uint >> 24) & 0xFF
    green = (uint >> 16) & 0xFF
    blue = (uint >> 8) & 0xFF
    alpha = uint & 0xFF

    return red, green, blue, alpha

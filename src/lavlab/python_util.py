from __future__ import annotations
import asyncio
from math import ceil
import numpy as np
import tempfile
from lavlab import ctx

#
## Python Utilities
#
def is_memsafe_array(shape, dtype=np.float64):
    size = np.prod(shape) * np.dtype(dtype).itemsize

    if size < ctx.max_memory:
        return True
    else:
        return False

def create_array(shape: tuple[int], dtype=np.float64):
    """
Creates an in-memory numpy array or a disk-based memmap array based on the available system memory.

Parameters
----------
shape : tuple
    Shape of the array.
dtype : np.dtype, Default: np.float64
    Data-type of the array’s elements.

Returns
-------
array
    Numpy in-memory array or memmap array based on the available system memory.
    """
    if is_memsafe_array(shape, dtype):
        return np.zeros(shape, dtype)
    else:
        _, path = tempfile.mkstemp()
        return np.memmap(path, dtype=dtype, mode='w+', shape=shape)


def chunkify(lst: list, n: int):
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
    return list(
        map(lambda x: lst[x * size:x * size + size],
        list(range(n)))
    )

def interlace_lists(*lists: list[list]) -> list:
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
    # get length of new arr
    length = 0
    for list in lists:
        length += len(list)

    # build new array
    arr = [None] * (length)
    for i, list in enumerate(lists):
        # slice index (put in every xth index)
        arr[i :: len(lists)] = list
    return arr


#
## Async Python Utilities
#
def merge_async_iters(*aiters):
    """
    Merges async generators using a asyncio.Queue.

    Notes
    -----
    Code from: https://stackoverflow.com/a/55317623

    Parameters
    ----------
    *aiters: AsyncGenerator
        AsyncGenerators to merge

    Returns
    -------
    AsyncGenerator
        Generator that calls all input generators
    """
    queue = asyncio.Queue(1)
    run_count = len(aiters)
    cancelling = False

    async def drain(aiter):
        nonlocal run_count
        try:
            async for item in aiter:
                await queue.put((False, item))
        except Exception as e:
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

    tasks = [asyncio.create_task(drain(aiter)) for aiter in aiters]
    return merged()


async def desync(it):
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

#
## Color Utilities
#
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


def uint_to_rgba(uint: int) -> int:
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


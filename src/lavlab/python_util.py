from __future__ import annotations
"""
Contains general utilities for lavlab's python scripts.
"""
import os
import asyncio

import numpy as np
from PIL import Image, ImageDraw
from skimage import measure

#
## Utility Dictionary
#
FILETYPE_DICTIONARY = {
    "SKIMAGE_FORMATS": {
        "JPEG": {"EXT": ".jpg", "MIME": "image/jpg"},
        "TIFF": {"EXT": ".tif", "MIME": "image/tiff"},
        "PNG": {"EXT": ".png", "MIME": "image/png"},
    },
    "MATLAB_FORMATS": {
        "M": {"EXT": ".m", "MIME": "text/plain", "MATLAB_MIME": "application/matlab-m"},
        "MAT": {
            "EXT": ".mat",
            "MIME": "application/octet-stream",
            "MATLAB_MIME": "application/matlab-mat",
        },
    },
    "GENERIC_FORMATS": {"TXT": {"EXT": ".txt", "MIME": "text/plain"}},
}
"""
Contains file extensions and mimetypes for commonly used files.

MATLAB_FORMATS has special key: MATLAB_MIME. MATLAB_MIME is a proprietary mimetype for MATLAB files. 
Clients will need to know how to handle MATLAB_MIME. Unless you know you need the MATLAB_MIME, use the normal mimetype.

```
FILETYPE_DICTIONARY={ 
    "SKIMAGE_FORMATS": {
        "JPEG": {
            "EXT": ".jpg",
            "MIME": "image/jpg"
        },
        "TIFF": {
            "EXT": ".tif",
            "MIME": "image/tiff"
        },
        "PNG": {
            "EXT": ".png",
            "MIME": "image/png"
        }
    },
    "MATLAB_FORMATS": {
        "M":{
            "EXT": ".m",
            "MIME": "text/plain",
            "MATLAB_MIME": "application/matlab-m"
        },
        "MAT":{
            "EXT": ".mat",
            "MIME": "application/octet-stream",
            "MATLAB_MIME": "application/matlab-mat"
        }
    },
    "GENERIC_FORMATS": {
        "TXT":{
            "EXT": ".txt",
            "MIME": "text/plain"
        }
    }
}
```

See Also
--------
lavlab.omero_utils.OMERO_DICTIONARY : Dictionary for converting omero info into python equivalents.
"""


#
## Utility Dictionary Utilities
#
def lookup_filetype_by_name(file: str) -> tuple[str, str]:
    """
    Searches dictionary for a matching filetype using the filename's extension.

    Parameters
    ----------
    file: str
        Filename to lookup type of.

    Returns
    -------
    tuple[str, str]
        Returns filetype set (SKIMAGE, MATLAB, etc) and the filetype key (JPEG, MAT, etc)
    """
    filename, f_ext = os.path.splitext(file)
    for set in FILETYPE_DICTIONARY:
        for format in FILETYPE_DICTIONARY[set]:
            ext = FILETYPE_DICTIONARY[set][format]["EXT"]
            if ext == f_ext:
                return set, format


#
## Python Utilities
#
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
    return [lst[i : i + n] for i in range(0, len(lst), n)]


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
## Image Array Utilities
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
        Integer encoding rgba value."""
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


def draw_shapes(
    input_img: Image.Image or np.ndarray,
    shape_points: tuple[int, tuple[int, int, int], tuple[np.ndarray, np.ndarray]],
) -> None:
    """
    Draws a list of shape points onto the input numpy array.

    Warns
    -------
    NO SAFETY CHECKS! MAKE SURE input_img AND shape_points ARE FOR THE SAME DOWNSAMPLE FACTOR!

    Parameters
    ----------
    input_img: np.ndarray
        3 channel numpy array
    shape_points: tuple(int, tuple(int,int,int), tuple(row, col))
        Expected to use output from lavlab.omero_util.getShapesAsPoints

    Returns
    -------
    ``None``
    """
    # need PIL image for processing
    arr = None
    if type(input_img) is np.ndarray:
        arr = input_img
        input_img = Image.fromarray(arr)

    # use pil imagedraw
    draw = ImageDraw.Draw(input_img)
    for id, rgb, xy in shape_points:
        draw.polygon(xy, fill=rgb)

    # but need to save changes to numpy if that's the input
    if arr is not None:
        new_arr = np.array(input_img)
        np.copyto(arr, new_arr, where=not None)


def apply_mask(img_bin: np.ndarray or Image, mask_bin: np.ndarray, where=None):
    """
    Essentially an alias for np.where()

    Notes
    -----
    DEPRECATED

    Parameters
    ----------
    img_bin: np.ndarray or PIL.Image
        Image as numpy array.
    mask_bin: np.ndarray
        Mask as numpy array.
    where: conditional, optional
        Passthrough for np.where conditional.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        Where and where not arrays"""
    if issubclass(type(img), Image.Image):
        img = np.array(img)
    if where is None:
        where = mask_bin != 0
    return np.where(where, mask_bin, img_bin)


def get_color_region_contours(
    img: np.ndarray or Image, rgb_val: tuple[int, int, int], axis=-1
) -> np.ndarray:
    """
    Finds the contours of all areas with a given rgb value. Useful for finding drawn ROIs.

    Parameters
    ----------
    img: np.ndarray or PIL.Image
        Image with ROIs. Converts PIL Image to np array for processing.
    rgb_val: tuple[int,int,int]
        Red, Green, and Blue values for the roi color.
    axis: int, Default: -1
        Which axis is the color channel. Default is the last axis [:,:,color]

    Returns
    -------
    list[ tuple[int(None), rgb_val, contours] ]
        Returns list of lavlab shapes.
    """
    if issubclass(type(img), Image.Image):
        img = np.array(img)
    mask_bin = np.all(img == rgb_val, axis=axis)
    contours = measure.find_contours(mask_bin, level=0.5)
    del mask_bin
    # wrap in lavlab shape convention
    for i, contour in enumerate(contours):
        contour = [(x, y) for y, x in contour]
        contours[i] = (None, rgb_val, contour)
    return contours

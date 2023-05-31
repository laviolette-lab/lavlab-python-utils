"""
Contains general utilities for lavlab's python scripts.
"""
import os
import asyncio

import numpy as np
from PIL import Image
from skimage import io, draw

#
## Utility Dictionary 
#
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
def lookup_filetype_by_name(file:str) -> tuple[str,str]:
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
            for ext in FILETYPE_DICTIONARY[set][format]["EXT"]:
                if ext == f_ext:
                    return set, format
#
## Python Utilities
#
def chunkify(lst:list,n:int):
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
    return [lst[i:i+n] for i in range(0, len(lst), n)]

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
    length=0
    for list in lists: length+=len(list)

    # build new array
    arr=[None]*(length)
    for i, list in enumerate(lists):
        # slice index (put in every xth index)
        arr[i::len(lists)] = list
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
    asynchronously yields results from input iterable.
"""
  for x in it: yield x  
  
        
#
## Image Array Utilities
#
def rgba_to_int(red: int, green: int, blue: int, alpha=255) -> int:
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
    return int.from_bytes([red, green, blue, alpha],
                      byteorder='big', signed=True)

def save_image_binary(path, bin, jpeg=None) -> str:
    """
Saves image binary to path using SciKit-Image. 

Notes
-----
Attempts to force Lossless JPEG compression.

Parameters
----------
path: str
    Path to save image at.
bin: np.ndarray
    Image as numpy array.
jpeg: bool, optional
    Whether or not to add quality=100 to skimage.io.imsave args. Will check for jpeg image.

Returns
-------
str
    Path of saved image.
    """
    # if not clarified, assume jpeg by filename
    if jpeg is None:
        if lookup_filetype_by_name(path) == "JPEG":
            jpeg=True
        else:
            jpeg=False

    # if jpeg make scikit image use lossless jpeg
    if jpeg is True: 
        io.imsave(path, bin, quality=100)
    else:
        io.imsave(path, bin)
    return path

def resize_image_array(input_array: np.ndarray, shape_yx: tuple[int,int], interpolation=Image.NEAREST) -> np.ndarray:
    """
Resizes input array to the given yx dimensions.

Notes
-----
no skimage or scipy versions as of scipy 1.3.0 :\n
https://docs.scipy.org/doc/scipy-1.2.1/reference/generated/scipy.misc.imresize.html


Parameters
----------
input_array: 2-3 dimensional np.ndarray
    Image array to resize. May support more than 3 channels but it is untested.
shape_yx: Desired height and width of output.
interpolation: PIL.Image.INTERPOLATION_TYPE, default: PIL.Image.NEAREST
    Which PIL interpolation type to use.

Returns
-------
np.ndarray
    Downsampled version of input_array.
    """
    return np.asarray(Image.fromarray(
        input_array).resize(shape_yx, interpolation), input_array.dtype)

def draw_shapes(input_img: np.ndarray, shape_points:tuple[int,tuple[int,int,int],tuple[np.ndarray, np.ndarray]]) -> None:
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
    for id, rgb, points in shape_points:
        rr,cc = draw.polygon(*points)
        input_img[rr,cc]=rgb


def apply_mask(img_bin: np.ndarray, mask_bin: np.ndarray, where=None):
    """
Essentially an alias for np.where()

Parameters
----------
img_bin: np.ndarray
    Image as numpy array.
mask_bin: np.ndarray
    Mask as numpy array.
where: conditional, optional
    Passthrough for np.where conditional.

Returns
-------
tuple[np.ndarray, np.ndarray]
    Where and where not arrays
"""
    if where is None:
        where=mask_bin!=0
    return np.where(where, mask_bin, img_bin)
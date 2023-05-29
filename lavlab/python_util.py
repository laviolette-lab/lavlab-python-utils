"""
Contains general utilities for lavlab's python scripts.
"""
import os
import asyncio
from skimage import io    
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
Contains mappings to filetype extensions and mimetypes.
SKIMAGE_FORMATS: JPEG, TIFF, PNG : Save formats supported by SciKit-Image
MATLAB_FORMATS: M, MAT : Proprietary Matlab filetypes, contains MATLAB_MIME a proprietary matlab mimetype
"""
def interlaceLists(lists: list) -> list:
    """
Interlaces a list of lists. Useful for combining tileLists of different channels.
Example: _interlaceLists([[1,3],[2,4]]) == [1,2,3,4] 
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

def chunkify(lst,n):
    return [lst[i:i+n] for i in range(0, len(lst), n)]

def merge_async_iters(*aiters):
    """
Merges async generators using a asyncio.Queue. From: https://stackoverflow.com/a/55317623\n
aiters...: AsyncGenerator[x]...\n
returns: AsyncGenerator[x]
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
  """Turns sync iterable into an async iterable."""
  for x in it: yield x  

def lookup_filetype_by_name(file):
    """Searches dictionary for a matching file type using the filename's extension"""
    filename, f_ext = os.path.splitext(file)
    for set in FILETYPE_DICTIONARY:
        for format in FILETYPE_DICTIONARY[set]:
            for ext in FILETYPE_DICTIONARY[set][format]["EXT"]:
                if ext == f_ext:
                    return format
        
def save_image_binary(path, bin, jpeg=None) -> str:
    """
Saves image binary to path using SciKit-Image. Forces Lossless JPEG compression.\n
path: path to save image at\n
bin: image as numpy array\n
jpeg: whether or not to add quality=100 to skimage.io.imsave args\n
returns: path of saved image
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
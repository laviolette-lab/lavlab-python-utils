from __future__ import annotations

import os
import logging
LOGGER = logging.getLogger(__name__)

import cv2
import numpy as np
import pyvips as pv
from skimage import measure
from lavlab import ctx
cv2.setNumThreads(ctx.num_threads)
#
## imsuite
#
def imread(image_path: os.PathLike ):
    """
Reads an image from a given filepath. (VIPS)
    """
    return pv.Image.new_from_file(str(image_path)).numpy()

def imwrite(img: np.ndarray or pv.Image, path: os.PathLike, fast=True):
    """
Writes an image array to disk as a tiff image
    """
    if not issubclass(type(img), pv.Image):
        assert issubclass(type(img), np.ndarray)
        img = pv.Image.new_from_array(img)
    opts = {}
    if img.width > 4000 or img.height > 4000:
        opts += ctx.TILING_OPTIONS
    if fast:
        opts += ctx.FAST_COMPRESSION_OPTIONS
    else:
        opts += ctx.SLOW_COMPRESSION_OPTIONS


    return



def imscale(img_arr: np.ndarray or pv.Image, factor: int):
    """
Resizes an input image. (VIPS)

Parameters
----------
img_arr: np.ndarray
    Input image as numpy array.
factor: int or tuple[int]
    Scale factor or desired dimensions

Returns
-------
np.ndarray
    Numpy array with new dimensions
    """
    if not issubclass(type(img_arr), pv.Image):
        assert issubclass(type(img_arr), np.ndarray)
        img_arr = pv.Image.new_from_array(img_arr)
    return img_arr.resize(factor).numpy()

def imresize(img_arr: np.ndarray or pv.Image, shape: tuple[int]):
    """
Resize an image to a target shape using pyvips. Allows deformation.

Parameters:
----------
img_arr: np.ndarray
    Input image to resize.
target_shape: tuple
    Target shape as a (height, width) tuple.

Returns:
-------
resized_image: np.ndarray
    Resized image.
    """
    if not issubclass(type(img_arr), pv.Image):
        assert issubclass(type(img_arr), np.ndarray)
        img_arr = pv.Image.new_from_array(img_arr)

    # Calculate the scaling factors for height and width
    scale_height = shape[0] / img_arr.height
    scale_width = shape[1] / img_arr.width

    # Apply and return the affine transformation to the image
    matrix = [scale_width, 0, 0, scale_height]
    return img_arr.affine(matrix).numpy()


#
## Drawing and Masking
#
def draw_shapes(
    img_arr: np.ndarray,
    shape_points: tuple[int, tuple[int, int, int], tuple[np.ndarray, np.ndarray]],
) -> None:
    """
    Draws a list of shape points onto the input numpy array.

    Warns
    -------
    NO SAFETY CHECKS! MAKE SURE img_arr AND shape_points ARE FOR THE SAME DOWNSAMPLE FACTOR!

    Parameters
    ----------
    img_arr: np.ndarray
        3 channel numpy array
    shape_points: tuple(int, tuple(int,int,int), tuple(row, col))
        Expected to use output from lavlab.omero_util.getShapesAsPoints

    Returns
    -------
    ``None``
    """
    assert issubclass(type(img_arr), np.ndarray)
    # Convert to BGR
    img_arr = cv2.cvtColor(img_arr, cv2.COLOR_RGB2BGR)
    # draw shapes
    for id, rgb, xy in shape_points:
        yx = [(y,x) for x,y in xy]
        bgr = tuple(rgb[::-1])
        cv2.fillPoly(img_arr, [yx], color=bgr) # TODO combine polys of same color for speed
    # Convert back to RGB
    img_arr = cv2.cvtColor(img_arr, cv2.COLOR_BGR2RGB)


def apply_mask(img_arr: np.ndarray, mask_arr: np.ndarray, where=None):
    """
    Essentially an alias for np.where()

    Notes
    -----
    DEPRECATED

    Parameters
    ----------
    img_arr: np.ndarray
        Image as numpy array.
    mask_bin: np.ndarray
        Mask as numpy array.
    where: conditional, optional
        Passthrough for np.where conditional.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        Where and where not arrays"""
    assert issubclass(type(img_arr), np.ndarray)
    assert issubclass(type(mask_arr), np.ndarray)

    if where is None:
        where = mask_arr != 0
    return np.where(where, mask_arr, img_arr)


def get_color_region_contours(
    img_arr: np.ndarray, rgb_val: tuple[int, int, int], axis=-1
) -> np.ndarray:
    """
    Finds the contours of all areas with a given rgb value. Useful for finding drawn ROIs.

    Parameters
    ----------
    img_arr: np.ndarray or PIL.Image
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
    assert issubclass(type(img_arr), np.ndarray)
    mask_bin = np.all(img_arr == rgb_val, axis=axis)
    contours = measure.find_contours(mask_bin, level=0.5)
    del mask_bin
    # wrap in lavlab shape convention
    for i, contour in enumerate(contours):
        contour = [(x, y) for y, x in contour]
        contours[i] = (None, rgb_val, contour)
    return contours

#
## helpers
#

def getDownsampleFromDimensions(base_shape:tuple[int,...], sample_shape:tuple[int,...]) -> tuple[float,...]:
    """
Essentially an alias for np.divide().

Finds the ratio between a base array shape and a sample array shape by dividing each axis.

Parameters
----------
base_shape: tuple(int)*x
    Shape of the larger image. (Image.size / base_nparray.shape)
sample_shape: tuple(int)*x
    Shape of the smaller image. (Image.size / sample_nparray.shape)

Raises
------
AssertionError
    Asserts that the input shapes have the same amount of axes

Returns
-------
tuple(int)*x
    Returns a tuple containing the downsample factor of each axis for the sample array.

"""
    assert len(base_shape) == len(sample_shape)
    return np.divide(base_shape, sample_shape)

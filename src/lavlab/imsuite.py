"""This module emulates MATLAB functions using our chosen library suite"""

from __future__ import annotations

import bisect

# import logging
import os
from enum import Enum
from typing import BinaryIO, Union

import matplotlib.pyplot
import nibabel as nib
import numpy as np
import pydicom
import pydicom.filereader
import pyvips as pv  # type: ignore
import scipy  # type: ignore
import scipy.ndimage  # type: ignore
import skimage

import lavlab
from lavlab.python_util import is_memsafe_pvimg

LOGGER = lavlab.LOGGER.getChild("imsuite")


class EdgeDetectionMethods(Enum):
    """List of available edge detection methods"""

    SOBEL = skimage.filters.sobel
    PREWITT = skimage.filters.prewitt
    ROBERTS = skimage.filters.roberts
    CANNY = skimage.feature.canny


#
## imsuite
#
def imread(image_path: Union[os.PathLike, BinaryIO, str], wild=False) -> np.ndarray:
    """
    Loads an image from a file.

    Parameters
    ----------
    image_path : Union[os.PathLike, BinaryIO]
        Path to image.
    wild : bool, optional
        If True, will not warn about niche formats, by default False.

    Returns
    -------
    np.ndarray
        Array of pixel values.

    Warnings
    --------
    This function can cause an OOM error if the image is too large!
    Use wsiread if your image is large.
    """
    if isinstance(image_path, BinaryIO):
        return pv.Image.new_from_buffer(image_path, "").numpy()
    image_path = str(image_path)
    if image_path.endswith(".nii") or image_path.endswith(".nii.gz"):
        if not wild:
            LOGGER.warning(
                "Nifti detected, use niftiread() for clarity if possible otherwise enable wild. Using niftiread..."  # pylint: disable=line-too-long
            )
        return niftiread(image_path)
    if image_path.endswith(".dcm"):
        if not wild:
            LOGGER.warning(
                "Dicom detected, use dicomread() for clarity if possible otherwise enable wild. Using dicomread..."  # pylint: disable=line-too-long
            )
        return dicomread(image_path)
    if os.path.isdir(image_path):
        if not wild:
            LOGGER.warning(
                "Dicom directory detected, use dicomread_volume() for clarity if possible otherwise enable wild. Using dicomread_volume"  # pylint: disable=line-too-long
            )
        return dicomread_volume(image_path)

    img = pv.Image.new_from_file(str(image_path))
    assert isinstance(img, pv.Image)
    if not is_memsafe_pvimg(img):
        LOGGER.warning("Image is too large for memory! Use wsiread() for large images.")
        return wsiread(image_path)
    return img.numpy()


def niftiread(image_path: Union[os.PathLike, str]) -> np.ndarray:
    """Loads a nifti from a file.

    Parameters
    ----------
    image_path : os.PathLike or string representing a file
        Path to nifti.

    Returns
    -------
    np.ndarray
        Array of pixel values.
    """
    return nib.load(str(image_path)).get_fdata()  # type: ignore


def dicomread(image_path: Union[os.PathLike, str]) -> np.ndarray:
    """Reads a dicom from a file.

    Parameters
    ----------
    image_path : os.PathLike or string representing a file
        Path to a dicom file.

    Returns
    -------
    np.ndarray
        Array of pixel values.
    """
    return pydicom.dcmread(image_path).pixel_array


def dicomread_volume(
    dicom_dir: Union[os.PathLike, str], file_extension=".dcm"
) -> np.ndarray:
    """Reads a dicom series from a directory.

    Parameters
    ----------
    dicom_dir : os.PathLike or string representing a directory
        Directory with the dicoms.

    Returns
    -------
    np.ndarray
        Dicom series.
    """

    # Get a list of all DICOM files in the directory
    dicom_files = [
        os.path.join(dicom_dir, filename)
        for filename in os.listdir(dicom_dir)
        if filename.endswith(file_extension)
    ]

    # Sort the DICOM files by instance number to ensure correct order
    dicoms = [pydicom.dcmread(file) for file in dicom_files]
    dicoms.sort(key=lambda x: x.InstanceNumber)

    # Read the first DICOM file to get image dimensions
    first_ds = dicoms[0]
    rows = int(first_ds.Rows)
    columns = int(first_ds.Columns)
    slices = len(dicoms)

    # Initialize a 3D array to store pixel data
    volume = np.zeros((slices, rows, columns), dtype=np.uint16)

    # Read each DICOM file and store pixel data in the volume array
    for i, file_path in enumerate(dicom_files):
        ds = pydicom.dcmread(file_path)
        volume[i, :, :] = ds.pixel_array
    return volume


def wsiread(image_path: Union[os.PathLike, str]) -> pv.Image:
    """Reads a Whole Slide Image from a file.

    Allows operations on images larger than memory.
    wrapper for pyvips.Image.new_from_file()

    Warnings
    --------
    This function does not return a numpy array!
    You'll need to use proper WSI workflows (like tilewise operations) to do analyses on this!
    See the pyvips documentation for more information.

    Parameters
    ----------
    image_path : os.PathLike or string representing a file
        Path to WSI.

    Returns
    -------
    pv.Image
        PyVips Image, see documentation for usage.
    """
    return pv.Image.new_from_file(str(image_path))


def imwrite(
    img: Union[np.ndarray, pv.Image], path: os.PathLike, **kwargs
) -> os.PathLike:
    """Writes an image to path. kwargs are passthrough to wrapped function.

    Parameters
    ----------
    img : Union[np.ndarray, pv.Image]
        Numpy array or PyVips image.
    path : os.PathLike
        Path to desired file.

    Returns
    -------
    os.PathLike
        Path of newly created file.
    """
    if not isinstance(img, pv.Image):
        assert isinstance(img, np.ndarray)
        img = pv.Image.new_from_array(img)
    assert isinstance(img, pv.Image)
    # need to figure out default compression settings some day
    settings = (
        kwargs if kwargs else {}
    )  # lavlab.ctx.histology.get_compression_settings()
    img.write_to_file(path, **settings)
    return path


def imresize2d(img: pv.Image, scale: tuple[float, float]) -> pv.Image:
    """
    Resizes a 2D image using pyvips' resize function.

    Parameters
    ----------
    img : pv.Image
        Input 2D image.
    target_size : Tuple[int, int]
        Desired dimensions as a tuple of (width, height).

    Returns
    -------
    pv.Image
        Resized image as a pyvips Image.
    """
    return img.resize(scale[0], vscale=scale[1])


def imresize3d(img: np.ndarray, target_size: tuple[int, int, int]) -> np.ndarray:
    """
    Resizes a 3D image using skimage's resize function.

    Parameters
    ----------
    img : np.ndarray
        Input 3D image.
    target_size : Tuple[int, int, int]
        Desired dimensions as a tuple of (depth, height, width).

    Returns
    -------
    np.ndarray
        Resized image as a NumPy array.
    """
    return skimage.transform.resize(img, target_size)


def imresize(
    img: Union[np.ndarray, pv.Image], factor: Union[int, float, tuple[int, ...]]
) -> Union[np.ndarray, pv.Image]:
    """
    Convenience wrapper for resize functions. Uses pyvips for 2D and skimage for 3D.

    Parameters
    ----------
    img : Union[np.ndarray, pv.Image]
        Input image as a NumPy array or pyvips Image.
    factor : Union[int, float, Tuple[int, ...]]
        Scale factor (if int or float) or desired dimensions (if tuple).

    Returns
    -------
    Union[np.ndarray, pv.Image]
        Resized image as a NumPy array or pyvips Image.
    """
    if isinstance(img, np.ndarray):
        dimensions = len(img.shape)
        if dimensions == 2:
            height, width = img.shape
    elif isinstance(img, pv.Image):
        dimensions = 2
        width, height = img.width, img.height
    else:
        raise TypeError(
            "Unsupported image type. Only np.ndarray and pyvips.Image are supported."
        )

    if dimensions == 2:
        if isinstance(factor, (int, float)):
            factor_tuple = (float(factor), float(factor))
        elif isinstance(factor, tuple) and len(factor) == 2:
            factor_tuple = (factor[1] / width, factor[0] / height)

        if isinstance(img, np.ndarray):
            pv_img = pv.Image.new_from_array(img)
            return imresize2d(pv_img, factor_tuple).numpy()
        return imresize2d(img, factor_tuple)
    if dimensions == 3:
        assert isinstance(factor, tuple)
        return imresize3d(img, (factor[0], factor[1], factor[2]))
    raise ValueError(
        "Unsupported image dimensions. Only 2D and 3D images are supported."
    )


def imrotate(
    img_arr: Union[np.ndarray, pv.Image], degrees: Union[int, float]
) -> Union[np.ndarray, pv.Image]:
    """Rotates an input image using pyvips.

    Parameters
    ----------
    img_arr : Union[np.ndarray, pv.Image]
        Input image as a NumPy array or pyvips Image.
    degrees : int or float
        Rotation in degrees.

    Returns
    -------
    Union[np.ndarray, pv.Image]
        Rotated image.
    """
    img = img_arr
    if not isinstance(img, pv.Image):
        assert isinstance(img, np.ndarray)
        img_arr = pv.Image.new_from_array(img)
    assert isinstance(img_arr, pv.Image)

    rotated_img = img_arr.rotate(degrees, interpolate=pv.Interpolate.new("nearest"))
    assert isinstance(rotated_img, pv.Image)

    # Retain the input type in the output
    if isinstance(img, np.ndarray):
        return rotated_img.numpy()
    return rotated_img


def imcrop(
    img_arr: Union[np.ndarray, pv.Image], dims: tuple[int, int, int, int]
) -> Union[np.ndarray, pv.Image]:
    """Crops a region from an image using pyvips

    Parameters
    ----------
    img_arr : Union[np.ndarray, pv.Image]
        Input image as a NumPy array or pyvips Image.
    dims : tuple[int,int,int,int]
        Desired dimensions: left, top, width, height.

    Returns
    -------
    Union[np.ndarray, pv.Image]
        Cropped image.
    """
    img = img_arr
    if not isinstance(img, pv.Image):
        assert isinstance(img, np.ndarray)
        img_arr = pv.Image.new_from_array(img)
    assert isinstance(img_arr, pv.Image)

    rotated_img = img_arr.crop(*dims)
    assert isinstance(rotated_img, pv.Image)

    # Retain the input type in the output
    if isinstance(img, np.ndarray):
        return rotated_img.numpy()
    return rotated_img


def imwarp(
    img_arr: Union[np.ndarray, pv.Image],
    affine_matrix: tuple[float, float, float, float],
) -> Union[np.ndarray, pv.Image]:
    """Affine warps a region from an image using pyvips

    Parameters
    ----------
    img_arr : Union[np.ndarray, pv.Image]
        Input image as a NumPy array or pyvips Image.
    affine : tuple[float,float,float,float]
        4 element affine transform matrix

    Returns
    -------
    Union[np.ndarray, pv.Image]
        Warped image
    """
    img = img_arr
    if not isinstance(img, pv.Image):
        assert isinstance(img, np.ndarray)
        img_arr = pv.Image.new_from_array(img)
    assert isinstance(img_arr, pv.Image)

    rotated_img = img_arr.affine(
        affine_matrix, interpolate=pv.Interpolate.new("nearest")
    )
    assert isinstance(rotated_img, pv.Image)

    # Retain the input type in the output
    if isinstance(img, np.ndarray):
        return rotated_img.numpy()
    return rotated_img


# pyvips affine doesn't support translation, wait until this is required
# def imtranslate(img_arr: Union[np.ndarray, pv.Image], tx: int, ty: int,
#              ) -> Union[np.ndarray, pv.Image]:
#     """Translates a region from an image using pyvips

#     :param img_arr: Input image as a NumPy array or pyvips Image.
#     :type img_arr: Union[np.ndarray, pv.Image]
#     :param tx: x axis pixel shift
#     :type tx: int
#     :param ty: y axis pixel shift
#     :type ty: int
#     :return: translated image
#     :rtype: Union[np.ndarray, pv.Image]
#     """
#     img = img_arr
#     if not isinstance(img, pv.Image):
#         assert isinstance(img, np.ndarray)
#         img_arr = pv.Image.new_from_array(img)
#     assert isinstance(img_arr, pv.Image)

#     translated_img = img_arr.affine((1,0,tx,0,1,ty), interpolate=pv.Interpolate.new('nearest'))
#     assert isinstance(translated_img, pv.Image)

#     # Retain the input type in the output
#     if isinstance(img, np.ndarray):
#         return translated_img.numpy()
#     return translated_img


def imadjust(
    img_arr: np.ndarray,
    tol: int = 1,
    vin: tuple[int, int] = (0, 255),
    vout: tuple[int, int] = (0, 255),
) -> np.ndarray:
    """
    Matlab's imadjust in Python.

    Parameters
    ----------
    img_arr : np.ndarray
        Grayscale image.
    tol : int, optional
        Tolerance, from 0 to 100, defaults to 1.
    vin : tuple[int, int], optional
        Input image bounds, defaults to (0,255).
    vout : tuple[int, int], optional
        Output image bounds, defaults to (0,255).

    Returns
    -------
    np.ndarray
        Intensity adjusted image.
    """
    assert len(img_arr.shape) == 2, "Input image should be 2-dims"

    tol = max(0, min(100, tol))

    if tol > 0:
        # Compute in and out limits
        hist = np.histogram(img_arr, bins=256, range=(0, 255))[0]

        # Cumulative histogram
        cum = hist.cumsum()

        # Compute bounds
        total = img_arr.size
        low_bound = total * tol / 100
        upp_bound = total * (100 - tol) / 100
        vin = (bisect.bisect_left(cum, low_bound), bisect.bisect_left(cum, upp_bound))

    # Avoid division by zero by setting vin[1] if it's the same as vin[0]
    if vin[0] == vin[1]:
        vin = (vin[0], vin[0] + 1)

    # Stretching
    scale = (vout[1] - vout[0]) / (vin[1] - vin[0])
    vs = img_arr - vin[0]
    vs[img_arr < vin[0]] = 0
    vd = vs * scale + 0.5 + vout[0]
    vd[vd > vout[1]] = vout[1]
    dst = vd

    return dst.astype(np.uint8)


def imhist(img_arr: np.ndarray, bins=256) -> None:
    """Plots and displays histogram of image intensity values

    Parameters
    ----------
    img_arr : np.ndarray
        Input image.
    bins : int, optional
        Number of bins, defaults to 256.

    Returns
    -------
    None
    """
    # Calculate the histogram
    matplotlib.pyplot.hist(img_arr.ravel(), bins=bins, range=(0.0, 256.0))

    # Set the title and labels
    matplotlib.pyplot.title("Histogram of Image Intensity Values")
    matplotlib.pyplot.xlabel("Intensity Value")
    matplotlib.pyplot.ylabel("Frequency")

    # Show the histogram
    matplotlib.pyplot.show()


def imcomplement(img_arr: np.ndarray) -> np.ndarray:
    """Generates the image's complement (inverts the image).

    Parameters
    ----------
    img_arr : np.ndarray
        Input image.

    Returns
    -------
    np.ndarray
        Inverted image.
    """
    return ~img_arr


def edge(img_arr: np.ndarray, method: str = "SOBEL", **kwargs) -> np.ndarray:
    """Edge detection.

    Parameters
    ----------
    img_arr : np.ndarray
        Input image as a numpy array.
    method : str, optional
        One of the enumerated methods: SOBEL, PREWITT, ROBERTS, CANNY. Defaults to SOBEL.

    Returns
    -------
    np.ndarray
        Edge map.
    """
    function = getattr(EdgeDetectionMethods, method, None)
    if function is None:
        raise KeyError(f"{method} is not a valid edge detection method!")
    return function(img_arr, **kwargs)


imshow = matplotlib.pyplot.imshow
"""Just matplotlib.pyplot.imshow() see docs for more"""

rgb2gray = skimage.color.rgb2gray
"""Just skimage.color.rgb2gray() see docs for more"""

histeq = skimage.exposure.equalize_hist
"""Just skimage.exposure.equalize_hist() see docs for more"""

imdilate = skimage.morphology.dilation
"""Just skimage.morphology.dilation() see docs for more"""

imerode = skimage.morphology.erosion
"""Just skimage.morphology.erosion() see docs for more"""

imfill = skimage.morphology.remove_small_holes
"""Just skimage.morphology.remove_small_holes() see docs for more"""

imopen = skimage.morphology.opening
"""just skimage.morphology.opening() see docs for more"""

imclose = skimage.morphology.closing
"""just skimage.morphology.closing() see docs for more"""

imtophat = skimage.morphology.white_tophat
"""just skimage.morphology.white_tophat() see docs for more"""

imbothat = skimage.morphology.black_tophat
"""just skimage.morphology.black_tophat() see docs for more"""

imreconstruct = skimage.morphology.reconstruction
"""just skimage.morphology.reconstruction() see docs for more"""

bwareaopen = skimage.morphology.remove_small_objects
"""Just skimage.morphology.remove_small_objects() see docs for more"""

watershed = skimage.segmentation.watershed
"""just skimage.segmentation.watershed() see docs for more"""

medfilt2 = skimage.filters.rank.median
"""just skimage.filters.rank.median() see docs for more"""

imbinarize = skimage.filters.threshold_otsu
"""just skimage.filters.threshold_otsu() see docs for more"""

regionprops = skimage.measure.regionprops
"""just skimage.measure.regionprops() see docs for more"""

bwlabel = scipy.ndimage.label
"""just scipy.ndimage.label() see docs for more"""

impyramid_expand = skimage.transform.pyramid_expand
"""just skimage.transform.pyramid_expand() see docs for more"""

impyramid_reduce = skimage.transform.pyramid_reduce
"""just skimage.transform.pyramid_reduce() see docs for more"""


#
## Drawing and Masking
#
def draw_shapes(
    img: Union[np.ndarray, pv.Image],
    shape_points: list[tuple[int, tuple[int, int, int], list[tuple[int, int]]]],
) -> Union[np.ndarray, pv.Image]:
    """
    Draws a list of shape points onto the input image using PyVips with SVG xml.

    Warnings
    --------
       No safety checks! Make sure img and shape_points are for the same downsample factor!

    Parameters
    ----------
    img : Union[np.ndarray, pyvips.Image]
        Input image as a NumPy array or pyvips Image.
    shape_points : List[Tuple[int, Tuple[int, int, int], List[Tuple[int, int]]]]
        List of tuples containing shape ID, RGB color, and list of points.
        Expected to use output from lavlab.omero_util.getShapesAsPoints.

    Returns
    -------
    Union[np.ndarray, pyvips.Image]
        Modified image with shapes drawn, same type as input.
    """
    if isinstance(img, np.ndarray):
        # Convert numpy array to pyvips Image
        pv_img = pv.Image.new_from_array(img)
    else:
        pv_img = img

    width = pv_img.width
    height = pv_img.height
    svg_header = (
        f'<svg viewBox="0 0 {width} {height}" '
        f'xmlns="http://www.w3.org/2000/svg" '
        f'shape-rendering="crispEdges">'
    )
    svg_shapes = ""

    for _, rgb, xy in shape_points:
        hex_color = f"#{rgb[0]:02x}{rgb[1]:02x}{rgb[2]:02x}"
        points = " ".join([f"{x},{y}" for y, x in xy])
        svg_shapes += f'<polygon points="{points}" fill="{hex_color}"/>'

    svg_footer = "</svg>"
    svg = svg_header + svg_shapes + svg_footer
    svg_img = pv.Image.svgload_buffer(svg.encode("utf-8"))

    # Composite the SVG image over the original image
    composite_img = pv_img.composite2(svg_img, "over", x=0, y=0)

    # Return the modified image in the same type as the input
    if isinstance(img, np.ndarray):
        return composite_img.numpy()
    return composite_img


def apply_mask(
    img: Union[np.ndarray, pv.Image], mask: Union[np.ndarray, pv.Image]
) -> Union[np.ndarray, pv.Image]:
    """
    Applies a binary mask to an image using PyVips.

    Parameters
    ----------
    img : np.ndarray
        Image as a NumPy array.
    mask : np.ndarray
        Binary mask as a NumPy array (same dimensions as img).

    Returns
    -------
    np.ndarray
        Image with the mask applied.
    """
    if isinstance(img, np.ndarray):
        # Convert numpy array to pyvips Image
        vips_img = pv.Image.new_from_array(img)
    else:
        vips_img = img
    if isinstance(mask, np.ndarray):
        # Convert numpy array to pyvips Image
        vips_mask = pv.Image.new_from_array(mask)
    else:
        vips_mask = mask

    # Convert numpy arrays to pyvips images
    vips_img = pv.Image.new_from_array(img)
    vips_mask = pv.Image.new_from_array(mask)

    # Create a masked image using PyVips
    masked_img = vips_img * vips_mask

    # Convert the PyVips image back to a numpy array
    if isinstance(img, np.ndarray):
        return masked_img.numpy()
    return masked_img


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
    assert isinstance(img_arr, np.ndarray)
    mask_bin = np.all(img_arr == rgb_val, axis=axis)
    contours = skimage.measure.find_contours(mask_bin, level=0.5)
    del mask_bin
    # wrap in lavlab shape convention
    for i, contour in enumerate(contours):
        contour = [(x, y) for y, x in contour]
        contours[i] = (None, rgb_val, contour)
    return contours


#
## helpers
#


def get_downsample_from_dimensions(
    base_shape: tuple[int, ...], sample_shape: tuple[int, ...]
) -> tuple[float, ...]:
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
    return tuple(np.divide(base_shape, sample_shape))

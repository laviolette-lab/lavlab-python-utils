"""Helps get full images from OMERO image objects"""

import os
from io import BytesIO
from typing import Generator, Optional

import numpy as np
import scipy  # type: ignore
from skimage import morphology
from lavlab import imsuite, tissuemask
from lavlab.omero.helpers import (
    force_image_wrapper,
    force_rps,
    get_downsampled_xy_dimensions,
    get_closest_resolution_level,
    get_rps_xy,
)
from lavlab.omero.tiles import (
    create_full_tile_list,
    get_tiles,
    create_tile_list_from_image,
)
from lavlab.python_util import create_array
from omero.gateway import BlitzGateway, ImageWrapper  # type: ignore


def get_plane_at_resolution_level(
    img: ImageWrapper, res_lvl: int, z: int, c: int, t: int, conn: BlitzGateway = None
) -> np.ndarray:
    """Gets a single 2d plane from an image

    Parameters
    ----------
    img : ImageWrapper
        OMERO Image Wrapper
    res_lvl : int
        Layer of image pyramid to pull
    z : int
        z-index to pull (usually 0 for lavlab purposes)
    c : int
        channel to pull, in rgb that's 0: red, 1: green, 2: blue
    t : int
        timepoint to pull (usually 0 for lavlab purposes)
    conn : BlitzGateway, optional
        OMERO Blitz Gateway, defaults to None and uses the one in the wrapper

    Returns
    -------
    np.ndarray
        2D numpy array
    """
    rps, close_rps = force_rps(img)
    img = force_image_wrapper(conn, img)
    rps.setResolutionLevel(res_lvl)

    size_x, size_y = get_rps_xy(rps)

    arr = None
    plane_size = size_x * size_y
    # TODO use context
    max_bytes = (
        int(img._conn.getProperty("Ice.MessageSizeMax")) * 1000  # pylint: disable=W0212
    )

    if plane_size * 8 > max_bytes:
        arr = create_array((size_y, size_x), np.uint8)
        tiles = create_full_tile_list([z], [c], [t], size_x, size_y, rps.getTileSize())
        for tile, (z, c, t, coord) in get_tiles(img, tiles, res_lvl):
            arr[coord[1] : coord[1] + coord[3], coord[0] : coord[0] + coord[2]] = tile
    else:
        arr = np.frombuffer(rps.getPlane(z, c, t), dtype=np.uint8).reshape(
            (size_y, size_x)
        )

    if close_rps:
        rps.close()

    return arr


def get_channels_at_resolution(
    img: ImageWrapper, xy_dim: tuple[int, int], channels: Optional[list[int]] = None
) -> Generator[tuple[int, np.ndarray[np.uint8]], None, None]:
    """
    Gathers tiles and scales down to desired resolution.

    Parameters
    ----------
    img: omero.gateway.ImageWrapper
        Omero Image object from conn.getObjects().
    xy_dim: tuple(x,y)
        Tuple of desired dimensions (x,y)
    channels: tuple(int,...), default: all channels
        Array of channels to gather.
        To grab only blue channel: channels=(2,)

    Returns
    -------
    PIL.Image.Image
        Python Image Object
    """
    res_lvl, _ = get_closest_resolution_level(img, xy_dim)
    if channels is None:
        channels = list(range(img.getSizeC()))
    for channel in channels:
        arr = get_plane_at_resolution_level(img, res_lvl, 0, channel, 0)
        if arr.shape != (xy_dim[1], xy_dim[0]):
            arr = imsuite.imresize(arr, (xy_dim[1], xy_dim[0]))
        yield channel, arr


def get_image_at_resolution_level(
    img: ImageWrapper, res_lvl: int, conn: BlitzGateway = None
) -> np.ndarray[np.uint8]:
    """Gets a full OMERO image from a given pyramid level.

    Parameters
    ----------
    img : ImageWrapper
        OMERO Image Object
    res_lvl : int
        Resolution level to pull from
    conn : BlitzGateway, optional
        OMERO Blitz Gateway, defaults to None

    Returns
    -------
    np.ndarray[np.uint8]
        OMERO Image as numpy array
    """
    if conn is None:
        conn = img._conn  # pylint: disable=W0212
    rps, close_rps = force_rps(img)
    img = force_image_wrapper(conn, img)
    rps.setResolutionLevel(res_lvl)

    size_x, size_y = get_rps_xy(rps)
    arr = create_array((size_y, size_x, img.getSizeC()), np.uint8)
    for c in range(img.getSizeC()):
        arr[:, :, c] = get_plane_at_resolution_level(rps, res_lvl, 0, c, 0, conn=conn)

    if close_rps is True:
        rps.close()
    return arr


def get_image_at_resolution(
    img: ImageWrapper, xy: tuple[int, int]
) -> np.ndarray[np.uint8]:
    """
    Gathers tiles of full rgb image and scales down to desired resolution.

    Parameters
    ----------
    img: omero.gateway.ImageWrapper
        Omero Image object from conn.getObjects().
    xy_dim: tuple(x,y)
        Tuple of desired dimensions (x,y)

    Returns
    -------
    PIL.Image.Image
        Python Image Object
    """
    arr = create_array((xy[1], xy[0], img.getSizeC()), np.uint8)
    for i, channel in get_channels_at_resolution(img, xy, list(range(img.getSizeC()))):
        arr[:, :, i] = channel
    return arr


def get_large_recon(img: ImageWrapper, ds=10) -> np.ndarray:
    """
    Gets a LargeRecon as a numpy array.
    LargeRecon10s are 1/10th the size of the full histology and our standard for processing.

    Parameters
    ----------
    img : ImageWrapper
        OMERO Image Object
    ds : int, optional
        Downsample Factor (dimension = raw_dim * 1/ds), by default 10

    Returns
    -------
    np.ndarray
        numpy array
    """
    xy_dim = get_downsampled_xy_dimensions(img, ds)
    return get_image_at_resolution(img, xy_dim)


def pull_large_recon(
    img: ImageWrapper, filename: os.PathLike, ds=10, **write_args
) -> os.PathLike:
    """
    Gets a LargeRecon then writes to a given filename.

    Parameters
    ----------
    img : ImageWrapper
        OMERO Image Object
    filename : os.PathLike
        Path to write image to
    ds : int, optional
        Downsample Factor (dimension = raw_dim * 1/ds), by default 10

    Returns
    -------
    os.PathLike
        path to large recon
    """
    arr = get_large_recon(img, ds)
    return imsuite.imwrite(arr, filename)


# TODO this needs some reworking
def mask_omero_tissue_loosely(img_obj: ImageWrapper, mpp=728) -> np.ndarray:
    """Generates a loose tissue mask for a given OMERO object.

    Parameters
    ----------
    img_obj : ImageWrapper
        OMERO Image object
    mpp : int, optional
        Allows custom resolutions during masking, defaults to 728

    Returns
    -------
    np.ndarray
        Tissue mask, not full image resolution, scaled to desired operating resolution.
    """
    phys_w = img_obj.getPixelSizeX()
    downsample_factor = mpp / phys_w
    scaled_dims = get_downsampled_xy_dimensions(img_obj, downsample_factor)

    # get img ( at super low res )
    arr = imsuite.imread(BytesIO(img_obj.getThumbnail(scaled_dims)))

    # # tia tissue masker (too fine for our purposes)
    mask = tissuemask.MorphologicalMasker(mpp=mpp).fit_transform(np.array([arr]))[0]

    # clean up mask
    mask = morphology.remove_small_holes(mask)
    mask = morphology.remove_small_objects(mask)

    # increase resolution
    scale = 32 / mpp
    imsuite.imresize(mask, scale)

    # smooth up mask
    mask = scipy.ndimage.binary_dilation(mask, iterations=16)
    mask = scipy.ndimage.gaussian_filter(mask.astype(float), sigma=24)
    mask = mask > 0.5

    # invert mask
    return ~mask


def load_image_smart(img: ImageWrapper):
    """
    Attempts to only request tiles with tissue, with the rest being filled in by white space.
    """
    mask = mask_omero_tissue_loosely(img)
    # Overall image dimensions
    image_width, image_height = img.getSizeX(), img.getSizeY()

    # Scaling factors
    scale_x = mask.shape[1] / image_width
    scale_y = mask.shape[0] / image_height

    tiles = create_tile_list_from_image(img)
    arr = create_array((image_height, image_width, img.getSizeC()), np.uint8)
    # Empty list to store tiles that land on the mask
    tiles_on_land = []

    for z, c, t, tile in tiles:
        x, y, width, height = tile

        # Calculate downscaled coordinates and dimensions
        x_ds, y_ds = int(x * scale_x), int(y * scale_y)
        width_ds, height_ds = int(width * scale_x), int(height * scale_y)

        # Check if any pixel in the corresponding mask area is True (assuming binary mask)
        if np.any(mask[y_ds : (y_ds + height_ds), x_ds : (x_ds + width_ds)]):
            tiles_on_land.append((z, c, t, tile))
    for tile, (z, c, t, coord) in get_tiles(img, tiles_on_land):  # type: ignore
        arr[coord[1] : coord[1] + coord[3], coord[0] : coord[0] + coord[2], c] = tile
    return arr

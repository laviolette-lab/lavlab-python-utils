"""OmeroPy tile helpers"""

import threading
from concurrent.futures import as_completed
from typing import Generator, Optional

import numpy as np
import omero.gateway  # type: ignore
from omero.gateway import ImageWrapper

import lavlab
from lavlab.python_util import interlace_lists


def get_tiles(  # pylint: disable=R0914
    img: omero.gateway.Image,
    tiles: list[tuple[int, int, int, tuple[int, int, int, int]]],
    res_lvl: Optional[int] = None,
    rps_bypass: bool = True,
    conn: omero.gateway.BlitzGateway = None,
) -> Generator[
    tuple[np.ndarray, tuple[int, int, int, tuple[int, int, int, int]]], None, None
]:
    """Pull tiles from omero faster using a ThreadPoolExecutor and executor.map!

    Parameters
    ----------
    img : omero.gateway.Image
        Omero image.
    tiles : list of tuple[int, int, int, tuple[int, int, int, int]]
        List of tiles to pull.
    res_lvl : int, optional
        Resolution level to pull, defaults to None.
    rps_bypass : bool, optional
        Passthrough to rps bypass option, defaults to True.
    conn : omero.gateway.BlitzGateway, optional
        Omero blitz gateway if not using omero image object, defaults to None.

    Yields
    ------
    tuple[np.ndarray, tuple[int, int, int, tuple[int, int, int, int]]]
        Tile and coords.
    """
    with lavlab.ctx.resources.io_pool as tpe:
        if conn is None:
            conn = img._conn  # pylint: disable=W0212
        local = threading.local()

        def work(args):
            """Runs inside a thread pool to get multiple tiles at a time."""
            pix_id, zct, coord, res_lvl, rps_bypass = args
            if getattr(local, "rps", None) is None:
                # Need to prepare a thread-specific rps
                local.rps = conn.c.sf.createRawPixelsStore()
                local.rps.setPixelsId(pix_id, rps_bypass)
                if res_lvl is None:
                    res_lvl = local.rps.getResolutionLevels()
                    res_lvl -= 1
                local.rps.setResolutionLevel(res_lvl)
            raw_data = local.rps.getTile(*zct, *coord)
            return raw_data, (*zct, coord)

        def cleanup():
            """Cleans out the raw pixels stores after work is done."""
            if hasattr(local, "rps"):
                local.rps.close()
                delattr(local, "rps")

        try:
            # Use executor.map for streamlined processing
            pix_id = img.getPrimaryPixels().getId()
            args_iter = ((pix_id, (z, c, t), coord, res_lvl, rps_bypass) for z, c, t, coord in tiles)
            for raw_data, (z, c, t, coord) in tpe.map(work, args_iter):
                processed_data = np.frombuffer(raw_data, dtype=np.uint8).reshape(coord[3], coord[2])
                yield processed_data, (z, c, t, coord)
        finally:
            # Cleanup resources
            for _ in range(tpe._max_workers):  # pylint: disable=W0212
                cleanup()


def create_tile_list_2d(  # pylint: disable=R0913
    z: int,
    c: int,
    t: int,
    size_x: int,
    size_y: int,  # pylint: disable=R0913
    tile_size: tuple[int, int],
) -> list[tuple[int, int, int, tuple[int, int, int, int]]]:
    """
    Creates a list of tile coords for a given 2D plane (z,c,t)

    Notes
    -----
    Tiles are outputed as (z,c,t,(x,y,w,h)) as this is the expected format by omero python bindings.
    This may cause confusion as numpy uses rows,cols (y,x) instead of x,y.

    Parameters
    ----------
    z: int
        z index
    c: int
        channel
    t: int
        timepoint
    size_x: int
        width of full image in pixels
    size_y: int
        height of full image in pixels
    tile_size: tuple(int, int)
        size of tile to gather (x,y)

    Returns
    -------
    list
        list of (z,c,t,(x,y,w,h)) tiles for use in getTiles
    """
    tile_list = []
    width, height = tile_size
    for y in range(0, size_y, height):
        width, height = tile_size  # reset tile size
        # if tileheight is greater than remaining pixels, get remaining pixels
        height = min(height, size_y - y)
        for x in range(0, size_x, width):
            # if tilewidth is greater than remaining pixels, get remaining pixels
            width = min(width, size_x - x)
            tile_list.append((z, c, t, (x, y, width, height)))
    return tile_list


def create_full_tile_list(  # pylint: disable=R0913
    z_indexes: list[int],
    channels: list[int],
    timepoints: list[int],
    width: int,
    height: int,
    tile_size: tuple[int, int],
    weave=False,
) -> list[tuple[int, int, int, tuple[int, int, int, int]]]:
    """
    Creates a list of all tiles for given dimensions.

    Parameters
    ----------
    z_indexes: list[int]
        list containing z_indexes to gather
    channels: list[int]
        list containing channels to gather
    timepoints: list[int]
        list containing timepoints to gather
    width: int
        width of full image in pixels
    height: int
        height of full image in pixels
    tile_size: tuple(int, int)
    weave: bool, Default: False
        Interlace tiles from each channel vs default seperate channels.

    Returns
    -------
    list
        list of (z,c,t,(x,y,w,h)) tiles for use in getTiles


    Examples
    --------
    ```
    >>> createFullTileList((0),(0,2),(0),1000,1000,10,10)
    list[
    (0,0,0,(0,0,10,10)), (0,0,0,(10,0,10,10))...
    (0,2,0,(0,0,10,10)), (0,2,0,(10,0,10,10))...
    ]

    Default will gather each channel separately.

    >>> createFullTileList((0),(0,2),(0),1000,1000,10,10, weave=True)
    list[
    (0,0,0,(0,0,10,10)), (0,2,0,(0,0,10,10)),
    (0,0,0,(10,0,10,10)), (0,2,0,(10,0,10,10))...
    ]

    Setting weave True will mix the channels together. Used  for writing RGB images
    ```
    """
    tile_list = []
    if weave is True:
        orig_c = channels
        channels = [
            0,
        ]
    for z in z_indexes:
        for c in channels:
            for t in timepoints:
                if weave is True:
                    tile_channels = []
                    for channel in orig_c:
                        tile_channels.append(
                            create_tile_list_2d(z, channel, t, width, height, tile_size)
                        )
                    tile_list.extend(interlace_lists(tile_channels))
                else:
                    tile_list.extend(
                        create_tile_list_2d(z, c, t, width, height, tile_size)
                    )

    return tile_list


def create_tile_list_from_image(
    img: ImageWrapper, rgb=False, include_z=True, include_t=True
) -> list[tuple[int, int, int, tuple[int, int, int, int]]]:
    """
    Generates a list of tiles from an omero.model.Image object.

    Parameters
    ----------
    img: omero.gateway.ImageWrapper
        Omero Image object from conn.getObjects().
    rgb: bool, Default: False.
        Puts tile channels next to each other.
    include_z: bool, Default: True
        get tiles for z indexes
    include_t: bool, Default: True
        get tiles for timepoints

    Returns
    -------
    list
        List of (z,c,t,(x,y,w,h)) tiles for use in getTiles.
    """
    width = img.getSizeX()
    height = img.getSizeY()
    z_indexes = list(range(img.getSizeZ()))
    timepoints = list(range(img.getSizeT()))
    channels = list(range(img.getSizeC()))

    img._prepareRenderingEngine()  # pylint: disable=W0212
    tile_size = img._re.getTileSize()  # pylint: disable=W0212
    img._re.close()  # pylint: disable=W0212

    if include_t is False:
        timepoints = [
            0,
        ]
    if include_z is False:
        z_indexes = [
            0,
        ]

    return create_full_tile_list(
        z_indexes, channels, timepoints, width, height, tile_size, rgb
    )

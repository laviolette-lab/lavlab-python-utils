

import numpy as np

from omero.gateway import ImageWrapper

from lavlab import imsuite
from lavlab.python_util import create_array
from lavlab.omero.tiles import getTiles, createFullTileList
from lavlab.omero.helpers import forceRPS, forceImageWrapper, getRPSXY

def getDownsampledXYDimensions(img: ImageWrapper, downsample_factor: int) -> tuple[int,int]:
    """
Returns XY (rows,columns) dimensions of given image at the downsample.

Parameters
----------
img: omero.gateway.ImageWrapper
    Omero Image object from conn.getObjects().
downsample_factor: int
    Takes every nth pixel from the base resolution.

Returns
-------
float
    img.getSizeX() / downsample_factor
float
    img.getSizeY() / downsample_factor
"""
    return (int(img.getSizeX() / int(downsample_factor)), int(img.getSizeY() / int(downsample_factor)))


def getClosestResolutionLevel(img: ImageWrapper, dim: tuple[int,int]) -> tuple[int,tuple[int,int,int,int]]:
    """
Finds the closest resolution to desired resolution.

Parameters
----------
img: omero.gateway.ImageWrapper or RawPixelsStore
    Omero Image object from conn.getObjects() or initialized rps
dim: tuple[int, int]
    tuple containing desired x,y dimensions.

Returns
-------
int
    resolution level to be used in rps.setResolution()
tuple[int,int,int,int]
    height, width, tilesize_y, tilesize_x of closest resolution
    """
    rps, close_rps=forceRPS(img)

    # get res info
    lvls = rps.getResolutionLevels()
    resolutions = rps.getResolutionDescriptions()

    # search for closest res
    for i in range(lvls) :
        res=resolutions[i]
        currDif=(res.sizeX-dim[0],res.sizeY-dim[1])
        # if this resolution's difference is negative in either axis, the previous resolution is closest
        if currDif[0] < 0 or currDif[1] < 0:

            rps.setResolutionLevel(lvls-i)
            tileSize=rps.getTileSize()

            if close_rps is True:
                rps.close()
            return (lvls-i, (resolutions[i-1].sizeX,resolutions[i-1].sizeY,
                             tileSize[0], tileSize[1]))
    # else smaller than smallest resolution, return smallest resolution
    rps.setResolutionLevel(lvls)
    tileSize=rps.getTileSize()
    if close_rps is True:
        rps.close()
    return lvls, (resolutions[i-1].sizeX,resolutions[i-1].sizeY,
                             tileSize[0], tileSize[1])



def getPlaneAtResolutionLevel(img, res_lvl, z, c, t, conn = None):
    rps, close_rps = forceRPS(img)
    img = forceImageWrapper(conn, img)
    rps.setResolutionLevel(res_lvl)

    res_description = getRPSXY(rps)
    size_x = res_description.sizeX
    size_y = res_description.sizeY

    arr = None
    plane_size = size_x * size_y
    MAX_BYTES = int(img._conn.getProperty('Ice.MessageSizeMax')) * 1000  # KB -> B

    if plane_size * 8 > MAX_BYTES:
        arr = create_array((size_y, size_x), np.uint8)
        tiles = createFullTileList([z], [c], [t],
                                   size_x, size_y, rps.getTileSize())
        for tile, (z, c, t, coord) in getTiles(img, tiles, res_lvl):
            arr[
                coord[1]:coord[1]+coord[3],
                coord[0]:coord[0]+coord[2]] = tile
    else:
        arr = np.frombuffer(rps.getPlane(z, c, t),
                            dtype=np.uint8).reshape((size_y, size_x))

    if close_rps:
        rps.close()

    return arr

def getChannelsAtResolution(img: ImageWrapper, xy_dim: tuple[int,int], channels:list[int]=None) -> list[np.ndarray[np.uint8]]:
    """
Gathers tiles and scales down to desired resolution.

Parameters
----------
img: omero.gateway.ImageWrapper
    Omero Image object from conn.getObjects().
xy_dim: tuple(x,y)
    Tuple of desired dimensions (x,y)
channels: tuple(int,...), default: all channelsP
    Array of channels to gather.
    To grab only blue channel: channels=(2,)

Returns
-------
PIL.Image.Image
    Python Image Object
    """
    res_lvl, xy_info = getClosestResolutionLevel(img, xy_dim)
    for channel in channels:
        arr = getPlaneAtResolutionLevel(img, res_lvl, 0, channel, 0)
        if arr.shape != (xy_dim[1], xy_dim[0]):
            arr = imsuite.imresize(arr, (xy_dim[1], xy_dim[0]))
        yield channel, arr

def getImageAtResolutionLevel(img, res_lvl, conn=None) -> np.ndarray[np.uint8]:
    rps, close_rps=forceRPS(img)
    img = forceImageWrapper(conn, img)
    rps.setResolutionLevel(res_lvl)

    res_description = getRPSXY(rps)
    size_x = res_description.sizeX
    size_y = res_description.sizeY
    arr = create_array((size_y, size_x, img.getSizeC()), np.uint8)
    for c in range(img.getSizeC()):
        arr[:,:,c] = getPlaneAtResolutionLevel(rps, res_lvl, 0, c, 0, conn = img._conn)

    if close_rps is True :
        rps.close()
    return arr


def getImageAtResolution(img: ImageWrapper, xy: tuple[int,int]) -> np.ndarray[np.uint8]:
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
    for i, channel in getChannelsAtResolution(img, xy, range(img.getSizeC())):
        arr[:,:,i] = channel
    return arr

def getLargeRecon(img: ImageWrapper, xy: tuple[int,int], conn=None) -> np.ndarray[np.uint8]:
    """
    """
    pass # TODO JIT large recon generation


def loadFullImageSmart(img_obj: ImageWrapper):
    """
Attempts to only request tiles with tissue, with the rest being filled in by white space.
    """

    async def work(img, mask):
        # Overall image dimensions
        image_width, image_height = img_obj.getSizeX(), img_obj.getSizeY()

        # Scaling factors
        scale_x = mask.shape[1] / image_width
        scale_y = mask.shape[0] / image_height

        tiles = createTileListFromImage(img)
        arr = create_array((image_height, image_width, img.getSizeC()), np.uint8)
        # Empty list to store tiles that land on the mask
        tiles_on_land = []

        for z,c,t,tile in tiles:
            x, y, width, height = tile

            # Calculate downscaled coordinates and dimensions
            x_ds, y_ds = int(x * scale_x), int(y * scale_y)
            width_ds, height_ds = int(width * scale_x), int(height * scale_y)

            # Check if any pixel in the corresponding mask area is True (assuming binary mask)
            if np.any(mask[y_ds:(y_ds+height_ds), x_ds:(x_ds+width_ds)]):
                tiles_on_land.append((z,c,t,tile))
        async for tile, (z,c,t,coord) in getTiles(img,tiles_on_land):
            arr [
                coord[1]:coord[1]+coord[3],
                coord[0]:coord[0]+coord[2],
                c ] = tile
        return Image.fromarray(arr)

    mask = maskTissueLoose(img_obj)

    event_loop = asyncio._get_running_loop()
    if event_loop is None:
        return asyncio.run(work(img_obj, mask))
    else:
        return work(img_obj, mask)


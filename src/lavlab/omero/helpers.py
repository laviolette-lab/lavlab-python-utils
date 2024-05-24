"""General helper functions for writing OMERO utilities."""

from contextlib import contextmanager
from typing import Union

from omero.gateway import (  # type: ignore
    BlitzGateway,
    BlitzObjectWrapper,
    ImageWrapper,
    ProxyObjectWrapper,
)

import lavlab.omero

LOGGER = lavlab.omero.LOGGER.getChild("helpers")

## GROUP CONTEXT HELPERS


def match_group(obj: BlitzObjectWrapper) -> int:
    """Switch current context to that of the object.

    Parameters
    ----------
    obj : BlitzObjectWrapper
        Any omero object.

    Returns
    -------
    int
        The group ID.

    """
    group_id = obj.details.group.id.val
    if group_id != obj._conn.SERVICE_OPTS.getOmeroGroup():  # pylint: disable=W0212
        LOGGER.info("Switching to group with ID: %s.", group_id)
        obj._conn.SERVICE_OPTS.setOmeroGroup(group_id)  # pylint: disable=W0212
    return group_id


@contextmanager
def use_group(obj: BlitzObjectWrapper):
    """Uses the object's group and then switches back to the original group.

    Parameters
    ----------
    obj : BlitzObjectWrapper
        Any OMERO object.

    Returns
    -------
    None
    """
    original_group_id = obj._conn.SERVICE_OPTS.getOmeroGroup()  # pylint: disable=W0212
    new_group_id = obj.details.group.id.val

    if original_group_id != new_group_id:
        LOGGER.info("Switching to group with ID: %s.", new_group_id)
        obj._conn.SERVICE_OPTS.setOmeroGroup(new_group_id)  # pylint: disable=W0212

    try:
        yield
    finally:
        if original_group_id != new_group_id:
            LOGGER.info("Reverting to original group with ID: %s.", original_group_id)
            obj._conn.SERVICE_OPTS.setOmeroGroup(  # pylint: disable=W0212
                original_group_id
            )


## property utilities


def get_downsampled_xy_dimensions(
    img: ImageWrapper, downsample_factor: int
) -> tuple[int, int]:
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
    return (
        int(img.getSizeX() / downsample_factor),
        int(img.getSizeY() / downsample_factor),
    )


def get_rps_xy(rps: ProxyObjectWrapper) -> tuple[int, int]:
    """Gets the XY dimensions that the RPS is set for

    Parameters
    ----------
    rps : ProxyObjectWrapper
        OMERO RawPixelsStore.

    Returns
    -------
    tuple[int, int]
        XY dimensions.
    """
    xy = rps.getResolutionDescriptions()[
        rps.getResolutionLevels() - 1 - rps.getResolutionLevel()
    ]
    return (xy.sizeX, xy.sizeY)


def get_closest_resolution_level(
    img: ImageWrapper, dim: tuple[int, int]
) -> tuple[int, tuple[int, int, int, int]]:
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
    rps, close_rps = force_rps(img)

    # get res info
    lvls = rps.getResolutionLevels()
    resolutions = rps.getResolutionDescriptions()

    # search for closest res
    for i in range(lvls):
        res = resolutions[i]
        current_difference = (res.sizeX - dim[0], res.sizeY - dim[1])
        # if this resolution's difference is negative in either axis
        #   the previous resolution is closest
        if current_difference[0] < 0 or current_difference[1] < 0:

            rps.setResolutionLevel(lvls - i)
            tile_size = rps.getTileSize()

            if close_rps is True:
                rps.close()
            return (
                lvls - i,
                (
                    resolutions[i - 1].sizeX,
                    resolutions[i - 1].sizeY,
                    tile_size[0],
                    tile_size[1],
                ),
            )
    # else smaller than smallest resolution, return smallest resolution
    rps.setResolutionLevel(lvls)
    tile_size = rps.getTileSize()
    if close_rps is True:
        rps.close()
    return lvls, (
        resolutions[i - 1].sizeX,
        resolutions[i - 1].sizeY,
        tile_size[0],
        tile_size[1],
    )


def ids_to_image_ids(conn: BlitzGateway, dtype: str, raw_ids: list[int]) -> list[int]:
    """
    Gathers Image ids from given OMERO objects. For Project and Dataset ids.
    Takes Image ids too for compatibility.

    Parameters
    ----------
    conn: omero.gateway.BlitzGateway
        An Omero BlitzGateway with a session.
    dtype: str
        String data type, should be one of: 'Image','Dataset', or 'Project'
    raw_ids: list[int]
        ids for datatype
    return: list[int]
        List of all found Image ids
    """
    if dtype != "Image":
        # project to dataset
        if dtype == "Project":
            project_ids = raw_ids
            raw_ids = []
            for project_id in project_ids:
                for dataset in conn.getObjects("dataset", opts={"project": project_id}):
                    raw_ids.append(dataset.getId())
        # dataset to image
        ids = []
        for dataset_id in raw_ids:
            for image in conn.getObjects("image", opts={"dataset": dataset_id}):
                ids.append(image.getId())
    # else rename image ids
    else:
        ids = raw_ids
    return ids


#
## casting utilities
#


def is_rps(rps: BlitzObjectWrapper):
    """checks if param is a RawPixelsStore, otherwise it's typically assumed you got a BlitzObject

    :param rps: Potential Omero RawPixelsStore
    :type rps: omero.gateway.BlitzObjectWrapper
    :return: _description_
    :rtype: _type_
    """
    if hasattr(rps, "getResolutionLevels"):
        return True
    return False


def force_rps(img_or_rps: Union[BlitzObjectWrapper, ProxyObjectWrapper], bypass=True):
    """
    Forces an ImageWrapper or RawPixelsStore to be an RawPixelsStore object.

    Parameters
    ----------
    img_or_rps: omero.gateway.ImageWrapper or omero.gateway.RawPixelsStore
        Object to be forced to RPS

    Returns
    -------
    omero.gateway.RawPixelsStore
        RawPixelsStore object
    bool
        Whether or not a RawPixelsStore was created, useful for closing.
    """
    if isinstance(img_or_rps, ImageWrapper):
        rps = img_or_rps._conn.createRawPixelsStore()  # pylint: disable=W0212
        rps.setPixelsId(img_or_rps.getPrimaryPixels().getId(), bypass)
        return rps, True
    return img_or_rps, False


def force_image_wrapper(conn, img_or_rps):
    """
    Forces a RawPixelsStore or ImageWrapper to be an ImageWrapper object.

    Parameters
    ----------
    img_or_rps: omero.gateway.RawPixelsStore or omero.gateway.ImageWrapper
        Object to be forced to ImageWrapper
    conn: omero.gateway.BlitzGateway
        Connection object for the OMERO server

    Returns
    -------
    omero.gateway.ImageWrapper
        ImageWrapper object
    """
    # if has getResolutionLevels method it's a rawpixelstore
    if is_rps(img_or_rps):
        if conn is None:
            raise ValueError(
                "Connection obj is required to create ImageWrapper from RawPixelStore"
            )

        qs = conn.getQueryService()
        pixels_id = img_or_rps.getPixelsId()
        pix = qs.get("PixelsI", pixels_id)
        qs.close()

        img = pix.getImage()
        if img is None:
            raise ValueError("Image corresponding to RawPixelsStore could not be found")

        img = conn.getObject("image", img.getId().getValue())
        return img
    return img_or_rps

"""Helps access ROIs from OMERO image objects"""

from omero.gateway import ImageWrapper, RoiWrapper, ShapeWrapper  # type: ignore
from omero_model_EllipseI import EllipseI  # type: ignore
from omero_model_PolygonI import PolygonI  # type: ignore
from omero_model_RectangleI import RectangleI  # type: ignore
from omero_model_RoiI import RoiI  # type: ignore
from skimage import draw

from lavlab.python_util import uint_to_rgba


def get_rois(img: ImageWrapper, roi_service=None) -> RoiWrapper:
    """
    Gathers OMERO RoiI objects.

    Parameters
    ----------
    img: omero.gateway.ImageWrapper
        Omero Image object from conn.getObjects()
    roi_service: omero.RoiService, optional
        Allows roiservice passthrough for performance
    """
    if roi_service is None:
        roi_service = img._conn.getRoiService()  # pylint: disable=W0212
        close_roi = True
    else:
        close_roi = False

    rois = roi_service.findByImage(
        img.getId(), None, img._conn.SERVICE_OPTS  # pylint: disable=W0212
    ).rois
    rois = [RoiWrapper(img._conn, roi) for roi in rois]  # pylint: disable=W0212
    if close_roi:
        roi_service.close()

    return rois


def _get_rectangle_points(shape: RectangleI, img_downsample: int, yx_shape: tuple[int, int]):
    x = np.round(shape.getX().getValue() / img_downsample)
    y = np.round(shape.getY().getValue() / img_downsample)
    w = np.round(shape.getWidth().getValue() / img_downsample)
    h = np.round(shape.getHeight().getValue() / img_downsample)
    points_data = draw.rectangle_perimeter(
        (y, x), (y + h, x + w), shape=yx_shape
    )
    points = [(points_data[1][i], points_data[0][i]) for i in range(len(points_data[0]))]
    return np.array(points, dtype=np.int32)


def _get_ellipse_points(shape: EllipseI, img_downsample: int, yx_shape: tuple[int, int]):
    points_data = draw.ellipse_perimeter(
        np.round(shape.getY().getValue() / img_downsample),
        np.round(shape.getX().getValue() / img_downsample),
        np.round(shape.getRadiusY().getValue() / img_downsample),
        np.round(shape.getRadiusX().getValue() / img_downsample),
        shape=yx_shape,
    )
    points = [(points_data[1][i], points_data[0][i]) for i in range(len(points_data[0]))]
    return np.array(points, dtype=np.int32)


def _get_polygon_points(shape: PolygonI, img_downsample: int, point_downsample: int):
    point_string_array = shape.getPoints().getValue().split(" ")
    yx: list[tuple[float, float]] = []
    for point_pair_str in point_string_array:
        coords = point_pair_str.split(",")
        if len(coords) == 2: # Ensure valid coordinate pair
            yx.append(
                (
                    np.round(float(coords[1]) / img_downsample),
                    np.round(float(coords[0]) / img_downsample),
                )
            )
    if yx:
        return np.array(yx[::point_downsample], dtype=np.int32)
    return None


def get_shapes_as_points(
    img: ImageWrapper, point_downsample=4, img_downsample=1, roi_service=None
) -> list[tuple[int, tuple[int, int, int], np.ndarray[np.int32]]]:
    """
    Gathers Rectangles, Polygons, and Ellipses as a tuple containing the
    shapeId, its rgb val, and a tuple of xy points of its bounds.

    Parameters
    ----------
    img: omero.gateway.ImageWrapper
        Omero Image object from conn.getObjects().
    point_downsample: int, Default: 4
        Grabs every nth point for faster computation.
    img_downsample: int, Default: 1
        How much to scale roi points.
    roi_service: omero.RoiService, optional
        Allows roiservice passthrough for performance.

    Returns
    -------
    returns: list[ shape.id, (r,g,b), list[tuple(x,y)] ]
        list of tuples containing a shape's id, rgb value, and a tuple of row and column points
    """
    yx_shape = (img.getSizeY() // img_downsample, img.getSizeX() // img_downsample)
    processed_shapes = []

    for roi in get_rois(img, roi_service):
        for shape in roi.copyShapes():
            points = None
            if isinstance(shape, RectangleI):
                points = _get_rectangle_points(shape, img_downsample, yx_shape)
            elif isinstance(shape, EllipseI):
                points = _get_ellipse_points(shape, img_downsample, yx_shape)
            elif isinstance(shape, PolygonI):
                points = _get_polygon_points(shape, img_downsample, point_downsample)

            if points is not None:
                shape_id = shape.getId().getValue()
                color_val = shape.getStrokeColor().getValue()
                r, g, b, _ = uint_to_rgba(color_val)
                processed_shapes.append((shape_id, (r, g, b), points))

    return sorted(processed_shapes, key=lambda x: x[0])


def create_roi(img: ImageWrapper, shapes: list[ShapeWrapper]):
    """
    Creates an omero RoiI object for an image from an array of shapes.

    Parameters
    ----------
    img: omero.gateway.ImageWrapper
        Omero Image object from conn.getObjects().
    shapes: list[omero_model_ShapeI.ShapeI]
        List of omero shape objects (Polygon, Rectangle, etc).

    Returns
    omero_model_RoiI.RoiI
        Local Omero ROI object, needs to be saved
    -------

    """
    # create an ROI, link it to Image
    roi = RoiI()
    # use the omero.model.ImageI that underlies the 'image' wrapper
    roi.setImage(img._obj)  # pylint: disable=W0212
    for shape in shapes:
        roi.addShape(shape)
    return roi


# def getShapesAsMasks(img: ImageWrapper, downsample: int, bool_mask=True,
#                      point_downsample=4, roi_service=None) -> list[np.ndarray]:
#     """
# Gathers Rectangles, Polygons, and Ellipses as masks for the image at the given downsampling
# Converts rectangles and ellipses into polygons
# (4 rectangle points into an array of points on the outline)
#     """
#     sizeX = int(img.getSizeX() / downsample)
#     sizeY = int(img.getSizeY() / downsample)

#     masks=[]
#     for id, rgb, points in getShapesAsPoints(img, point_downsample, downsample, roi_service):
#         if bool_mask is True:
#             val = 1
#             dtype = np.bool_
#             arr_shape=(sizeY,sizeX)
#         else:
#             # want to overwrite region completely, cannot have 0 value
#             for i, c in enumerate(rgb):
#                 if c == 0: rgb[i]=1

#             val = rgb
#             dtype = np.uint8
#             arr_shape=(sizeY,sizeX, img.getSizeC())

#         mask=np.zeros(arr_shape, dtype)
#         rr,cc = draw.polygon(*points)
#         mask[rr,cc]=val
#         masks.append(mask)

#     if not masks: # if masks is empty, return none
#         return None

#     return masks

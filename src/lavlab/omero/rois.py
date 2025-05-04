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


def get_shapes_as_points(  # pylint: disable=R0914
    img: ImageWrapper, point_downsample=4, img_downsample=1, roi_service=None
) -> list[tuple[int, tuple[int, int, int], list[tuple[float, float]]]]:
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
    yx_shape = (img.getSizeY() / img_downsample, img.getSizeX() / img_downsample)

    shapes = []
    for roi in get_rois(img, roi_service):
        points = None
        for shape in roi.copyShapes():

            if isinstance(shape, RectangleI):
                x = float(shape.getX().getValue()) / img_downsample
                y = float(shape.getY().getValue()) / img_downsample
                w = float(shape.getWidth().getValue()) / img_downsample
                h = float(shape.getHeight().getValue()) / img_downsample
                # points = [(x, y),(x+w, y), (x+w, y+h), (x, y+h), (x, y)]
                points = draw.rectangle_perimeter(
                    (y, x), (y + h, x + w), shape=yx_shape
                )
                points = [(points[1][i], points[0][i]) for i in range(len(points[0]))]

            if isinstance(shape, EllipseI):
                points = draw.ellipse_perimeter(
                    int(shape.getY().getValue() // img_downsample),
                    int(shape.getX().getValue() // img_downsample),
                    int(shape.getRadiusY().getValue() // img_downsample),
                    int(shape.getRadiusX().getValue() // img_downsample),
                    shape=yx_shape,
                )
                points = [(points[1][i], points[0][i]) for i in range(len(points[0]))]

            if isinstance(shape, PolygonI):
                point_string_array = shape.getPoints().getValue().split(" ")

                xy: list[tuple[float, float]] = []
                for i, points in enumerate(point_string_array):
                    coords = points.split(",")
                    xy.append(
                        (
                            float(coords[0]) / img_downsample,
                            float(coords[1]) / img_downsample,
                        )
                    )
                if xy:
                    points = xy

            if points is not None:
                color_val = shape.getStrokeColor().getValue()

                r, g, b, _ = uint_to_rgba(color_val)

                if (len(points) / point_downsample) <= 2:
                    point_downsample = 1
                    print('Point downsampling too high, leading to destroyed polygon. Setting downsampling = 1.')

                points = [
                    (float(x), float(y))
                    for x, y in 
                        points[::point_downsample]
                ]

                # Ensure shape ID is an integer
                shape_id = int(shape.getId().getValue())
                shapes.append((shape_id, (r, g, b), points))

    # Sorting by the first element of the tuple which is the ID
    return sorted(shapes, key=lambda x: x[0])


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

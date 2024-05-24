# pylint: skip-file
# type: ignore
from lavlab.omero.rois import get_rois, get_shapes_as_points
from omero.gateway import RoiWrapper


def test_get_rois(sample_image):
    # Call the get_rois function
    rois = get_rois(sample_image)
    print(type(rois[0]))
    # Verify the output
    assert isinstance(rois, list), "Expected a list of ROIs"
    assert all(
        isinstance(roi, RoiWrapper) for roi in rois
    ), "Expected all items to be of type RoiI"


def test_get_shapes_as_points(sample_image):
    # Call the get_shapes_as_points function
    shapes = get_shapes_as_points(sample_image)

    # Verify the output
    assert isinstance(shapes, list), "Expected a list of shapes"
    for shape_id, color, points in shapes:
        assert isinstance(shape_id, int), "Expected shape ID to be an integer"
        assert (
            isinstance(color, tuple) and len(color) == 3
        ), "Expected color to be a tuple of 3 integers"
        assert all(
            isinstance(point, tuple) and len(point) == 2 for point in points
        ), "Expected each point to be a tuple of two floats"

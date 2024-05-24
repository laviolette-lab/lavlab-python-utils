# pylint: skip-file
# type: ignore
import os

import numpy as np

# Import functions from the file
from lavlab.omero.images import (
    get_channels_at_resolution,
    get_image_at_resolution,
    get_image_at_resolution_level,
    get_plane_at_resolution_level,
    pull_large_recon,
)

# from omero.gateway import ImageWrapper, BlitzGateway


def test_get_plane_at_resolution_level(sample_image):
    # Call the get_plane_at_resolution_level function
    result = get_plane_at_resolution_level(
        sample_image, res_lvl=0, z_idx=0, c_idx=0, t_idx=0
    )

    # Verify the output
    assert isinstance(result, np.ndarray), "Expected result to be a NumPy array"
    assert result.ndim == 2, "Expected a 2D array"


def test_get_channels_at_resolution(sample_image):
    # Define a resolution
    xy_dim = (512, 512)

    # Call the get_channels_at_resolution function
    channels = list(get_channels_at_resolution(sample_image, xy_dim))

    # Verify the output
    assert len(channels) > 0, "Expected to find at least one channel"
    for channel, arr in channels:
        assert isinstance(arr, np.ndarray), "Expected each channel to be a NumPy array"
        assert arr.shape == (
            xy_dim[1],
            xy_dim[0],
        ), "Expected array shape to match specified dimensions"


def test_get_image_at_resolution_level(sample_image):
    # Call the get_image_at_resolution_level function
    result = get_image_at_resolution_level(sample_image, res_lvl=0)

    # Verify the output
    assert isinstance(result, np.ndarray), "Expected result to be a NumPy array"
    assert result.ndim == 3, "Expected a 3D array (xyc)"


def test_get_image_at_resolution(sample_image):
    # Define a resolution
    xy = (512, 512)

    # Call the get_image_at_resolution function
    result = get_image_at_resolution(sample_image, xy)

    # Verify the output
    assert isinstance(result, np.ndarray), "Expected result to be a NumPy array"
    assert result.ndim == 3, "Expected a 3D array"
    assert result.shape[:2] == (
        xy[1],
        xy[0],
    ), "Expected array shape to match specified dimensions"


# tests get_large_recon vicariously
def test_pull_large_recon(sample_image):
    ds = 100
    result = pull_large_recon(sample_image, "lr100.jpeg", ds)
    assert os.path.exists(result)

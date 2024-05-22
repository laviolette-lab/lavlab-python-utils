# pylint: skip-file
# type: ignore
import os

import nibabel as nib
import nibabel.testing as nibTest
import numpy as np
import pydicom
import pyvips as pv
from lavlab.imsuite import (
    dicomread,
    dicomread_volume,
    edge,
    imadjust,
    imcomplement,
    imcrop,
    imread,
    imresize,
    imrotate,
    imwarp,
    imwrite,
    niftiread,
    wsiread,
    draw_shapes,
    apply_mask,
    get_color_region_contours,
)
from pydicom.data import get_testdata_file, get_testdata_files


# TODO comprehensive testing, a lot of allowed permutations are untested and some do not actually test that the given function works, just that it doesn't error.
def test_imread():
    # Create a test image and save it temporarily
    test_image_path = "temp_test_image.png"
    test_array = np.random.rand(100, 100)
    pv.Image.new_from_array(test_array * 255).write_to_file(test_image_path)

    try:
        result = imread(test_image_path)
        assert isinstance(result, np.ndarray), "The result should be a numpy array."
        assert (
            result.shape == test_array.shape
        ), "The shape of the read image should match the original."
    finally:
        os.remove(test_image_path)


def test_niftiread():
    # Fetch sample data
    nifti_path = os.path.join(nibTest.data_path, "example4d.nii.gz")

    # Test the niftiread function
    result = niftiread(nifti_path)
    assert isinstance(result, np.ndarray), "The result should be a numpy array."
    assert (
        result.shape == nib.load(nifti_path).get_fdata().shape
    ), "The shape of the loaded NIFTI should match the original."


def test_dicomread():
    # Path to a sample DICOM file included with pydicom
    dicom_path = get_testdata_file("CT_small.dcm")

    # Test the dicomread function
    result = dicomread(dicom_path)
    assert isinstance(result, np.ndarray), "The result should be a numpy array."
    assert (
        result.shape == pydicom.dcmread(dicom_path).pixel_array.shape
    ), "The shape of the read DICOM should match the original."


def test_dicomread_volume():
    # Test the dicomread_volume function
    directory = get_testdata_file("dicomdirtests/98892003/MR1")
    # dicoms from pydicom do not have .dcm extension, ignore it.
    result = dicomread_volume(directory, file_extension="")
    assert isinstance(result, np.ndarray), "The result should be a numpy array."


def test_wsiread():
    # Create a test WSI image and save it temporarily
    test_wsi_path = "temp_test_wsi.v"
    test_array = np.random.rand(1000, 1000)
    pv.Image.new_from_array(test_array * 255).write_to_file(test_wsi_path)

    try:
        result = wsiread(test_wsi_path)
        assert isinstance(result, pv.Image), "The result should be a PyVips Image."
    finally:
        os.remove(test_wsi_path)


def test_imwrite():
    # Create an image array and write it
    img = np.random.rand(100, 100)
    path = "test_output.png"
    imwrite(img, path)

    # Check if the file was created
    assert os.path.exists(path), "The output file should exist."
    os.remove(path)


def test_imresize_np2d():
    img = np.random.rand(200, 100)
    resized_img = imresize(img, (100, 50))
    assert resized_img.shape == (
        100,
        50,
    ), "The resized image should have the new dimensions."


def test_imrotate_np90():
    img = np.random.rand(50, 100)
    rotated_img = imrotate(img, 90)
    assert rotated_img.shape == (
        100,
        50,
    ), "The rotated image should flip its dimensions."


def test_imcrop_np2d():
    img = np.random.rand(100, 100)
    cropped_img = imcrop(img, (25, 25, 50, 50))
    assert cropped_img.shape == (
        50,
        50,
    ), "The cropped image should have the correct dimensions."


def test_imwarp_nowarp_np2d():
    img = np.random.rand(100, 100)
    warp_matrix = (1, 0, 0, 1)
    warped_img = imwarp(img, warp_matrix)
    assert (
        warped_img.shape == img.shape
    ), "The warped image should maintain its dimensions."


# def test_imtranslate_np2d():
#     img = np.random.rand(100, 100)
#     translated_img = imtranslate(img, 10, 10)
#     assert translated_img.shape == img.shape, "The translated image should maintain its dimensions."


def test_imadjust():
    img = np.random.rand(100, 100)
    adjusted_img = imadjust(img)
    assert (
        adjusted_img.shape == img.shape
    ), "The adjusted image should maintain its dimensions."
    assert (
        adjusted_img.max() <= 255 and adjusted_img.min() >= 0
    ), "The adjusted image values should be within [0, 1]."


def test_imcomplement_np2d():
    img = np.array([[0, 255], [255, 0]], dtype=np.uint8)
    complemented_img = imcomplement(img)
    assert np.array_equal(
        complemented_img, np.array([[255, 0], [0, 255]], dtype=np.uint8)
    ), "The complemented image should invert the values."


def test_edge_np2d():
    img = np.random.rand(100, 100)
    edged_img = edge(img)
    assert isinstance(
        edged_img, np.ndarray
    ), "The edge detection should return a numpy array."


def test_draw_shapes():
    # Create a test image (100x100) and shape points
    img = np.zeros((100, 100), dtype=np.uint8)
    shape_points = [
        (1, (255, 0, 0), [(10, 10), (20, 20), (10, 30)]),  # Red triangle
        (2, (0, 255, 0), [(40, 40), (50, 50), (40, 60)]),  # Green triangle
    ]

    # Test draw_shapes with NumPy image
    result_img_np = draw_shapes(img, shape_points)
    assert isinstance(result_img_np, np.ndarray), "Expected NumPy array as output"

    # Test draw_shapes with PyVips image
    img_vips = pv.Image.new_from_array(img)
    result_img_vips = draw_shapes(img_vips, shape_points)
    assert isinstance(result_img_vips, pv.Image), "Expected pyvips Image as output"


def test_apply_mask():
    # Create a test image (100x100) and mask
    img = np.random.rand(100, 100, 3).astype(np.float32)
    mask = np.ones((100, 100, 3), dtype=np.float32)
    mask[25:75, 25:75, :] = 0  # A central masked region

    # Test apply_mask with NumPy image
    result_img_np = apply_mask(img, mask)
    assert isinstance(result_img_np, np.ndarray), "Expected NumPy array as output"

    # Test apply_mask with PyVips image
    img_vips = pv.Image.new_from_array(img)
    mask_vips = pv.Image.new_from_array(mask)
    result_img_vips = apply_mask(img_vips, mask_vips)
    assert isinstance(result_img_vips, pv.Image), "Expected pyvips Image as output"


def test_get_color_region_contours():
    # Create a test image with red and green regions
    img = np.zeros((100, 100, 3), dtype=np.uint8)
    img[10:30, 10:30, :] = [255, 0, 0]  # Red square
    img[40:60, 40:60, :] = [0, 255, 0]  # Green square

    # Test get_color_region_contours for red region
    red_contours = get_color_region_contours(img, (255, 0, 0))
    assert len(red_contours) == 1, "Expected one red region"

    # Test get_color_region_contours for green region
    green_contours = get_color_region_contours(img, (0, 255, 0))
    assert len(green_contours) == 1, "Expected one green region"

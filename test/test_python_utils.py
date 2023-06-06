import unittest
import os
import asyncio
import numpy as np
from PIL import Image

from omero.gateway import ColorHolder

from lavlab import python_util

class TestPythonUtil(unittest.TestCase):

    def test_lookup_filetype_by_name(self):
        filename = '/fake/but/ignored/path/to/test.jpg'
        set, format = python_util.lookup_filetype_by_name(filename)
        self.assertEqual(set, "SKIMAGE_FORMATS")
        self.assertEqual(format, "JPEG")

    def test_chunkify(self):
        test_list = list(range(10))
        expected_output = [[0, 1, 2], [3, 4, 5], [6, 7, 8], [9]]
        self.assertEqual(python_util.chunkify(test_list, 3), expected_output)

    def test_interlace_lists(self):
        list1 = [1, 3, 5]
        list2 = [2, 4, 6]
        expected_output = [1, 2, 3, 4, 5, 6]
        self.assertEqual(python_util.interlace_lists(list1, list2), expected_output)

    # test probably broken?
    def test_rgba_to_uint(self):
        c_vals = [(255,255,255,255), (0,0,0,0), (128,128,128,128)]
        for cval in c_vals:
            self.assertEqual(python_util.rgba_to_uint(*cval), ColorHolder.fromRGBA(*cval).getInt())

    def test_draw_shapes_np(self):
        img = np.zeros((10, 10, 3), dtype=np.uint8)
        points = (None, (255, 255, 255), ((0,0),(0,2),(2,0)))
        python_util.draw_shapes(img, [points])
        self.assertTrue(np.array_equal(img[0][0], [255, 255, 255])) # contour should be colored
        self.assertTrue(np.array_equal(img[1][1], [255, 255, 255])) # inside should be colored
        self.assertTrue(np.array_equal(img[8][8], [0, 0, 0])) # outside should not be colored

    def test_draw_shapes_pil(self):
        img = Image.new('RGB', (10, 10), (0, 0, 0))
        points = (None, (255, 255, 255), ((0,0),(0,2),(2,0)))
        python_util.draw_shapes(img, [points])
        self.assertTrue(np.array_equal(img.getpixel((0,0)), [255, 255, 255])) # contour should be colored
        self.assertTrue(np.array_equal(img.getpixel((1,1)), [255, 255, 255])) # inside should be colored
        self.assertTrue(np.array_equal(img.getpixel((8,8)), [0, 0, 0])) # outside should not be colored

    def test_get_color_region_contours_np(self):
        img = np.zeros((10, 10, 3), dtype=np.uint8)
        contours = python_util.get_color_region_contours(img, (0,0,0))
        self.assertEqual(len(contours), 0)

    def test_get_color_region_contours_pil(self):
        img = Image.new('RGB', (10, 10), (0, 0, 0))
        contours = python_util.get_color_region_contours(img, (0, 0, 0))
        self.assertEqual(len(contours), 0)  # Since the whole image is black, no contour is found

    def test_merge_async_iters(self):
        async def test():
            iterables = [
                self._aiter([1, 2, 3]),
                self._aiter([4, 5, 6]),
                self._aiter([7, 8, 9])
            ]
            merged = python_util.merge_async_iters(*iterables)
            result = [item async for item in merged]
            # async will have items out of order, but should be same qty
            self.assertCountEqual(result, [1, 2, 3, 4, 5, 6, 7, 8, 9])
        asyncio.run(test())
        
    def test_desync(self):
        async def test():
            iterable = [1, 2, 3, 4, 5]
            result = [item async for item in python_util.desync(iterable)]
            self.assertEqual(result, iterable)
        asyncio.run(test())


    async def _aiter(self, iterable):
        for item in iterable:
            yield item

if __name__ == '__main__':
    unittest.main()

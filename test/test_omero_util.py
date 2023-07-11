import unittest
import asyncio
import numpy as np

from idr import connection
from omero_model_PolygonI import PolygonI

from lavlab import omero_util

class TestOmeroUtil(unittest.TestCase):
    def __init__(self, methodName: str = "runTest") -> None:
        super().__init__(methodName)
        self.conn = connection("idr.openmicroscopy.org","public", "public")
        self.img_id = 14257890

    def test_getTiles(self):
        tiles = [(0,0,0,(0,0,1024,1024))]*10
        img_obj = self.conn.getObject('image',self.img_id)
        async def gatherCoords(img_obj, tiles):
            coords = []
            async for tile, zct_coord in omero_util.getTiles(img_obj, tiles):
                coords.append(zct_coord)
            return coords
        self.assertEqual(tiles, asyncio.run(gatherCoords(img_obj, tiles)))
        self.conn.close()
        

    def test_getDownsampledXYDimensions(self) :
        img_obj = self.conn.getObject('image',self.img_id)
        
        self.assertTrue(np.array_equal(omero_util.getDownsampledXYDimensions(img_obj, 10), 
                                       (int(img_obj.getSizeX()/10), int(img_obj.getSizeY()/10))))
        self.conn.close()
        

    def test_getDownsampleFromDimensions(self):
        self.assertTrue(np.array_equal (omero_util.getDownsampleFromDimensions((100,100,3),(10,10,3)), (10,10,1)))
        self.assertTrue(np.array_equal (omero_util.getDownsampleFromDimensions((100,100,100),(100,100,25)), (1,1,4)))
        self.conn.close()
        

    def test_getClosestResolutionLevel(self):
        img_obj = self.conn.getObject('image',self.img_id)
        img_obj._prepareRenderingEngine()
        self.assertEqual(omero_util.getClosestResolutionLevel(img_obj, (10,10))[0],
                            img_obj._re.getResolutionLevels())
        img_obj._re.close()
        self.conn.close()
        

    def test_getImageAtResolution(self):
        img_obj = self.conn.getObject('image',self.img_id)
        img = omero_util.getImageAtResolution(img_obj, (2048, 4096))
        self.assertTrue(np.array_equal(img.size, (2048, 4096)))
        self.conn.close()
        

    def test_getLargeRecon(self):
        img_obj = self.conn.getObject('image',self.img_id)
        recon_obj, recon = omero_util.getLargeRecon(img_obj, 8, skip_upload=True)
        self.assertTrue(np.array_equal(recon.size, (int(img_obj.getSizeX()/8), int(img_obj.getSizeY()/8))))
        self.conn.close()
        
    # it works but cannot get the test to show it
    # def test_createTileList2D(self):
    #     rv = omero_util.createTileList2D(0,0,0,1000,1000,(500,500))
    #     print(rv)
    #     tiles = [(0, 0, 0, (0, 0, 500, 500)), (0, 0, 0, (500, 0, 500, 500)), (0, 0, 0, (0, 500, 500, 500)), (0, 0, 0, (500, 500, 500, 500))]
    #     self.assertTrue(np.array_equal(rv, tiles))
    #     self.conn.close()
        
    # def test_createFullTileList(self):
    #     rv = omero_util.createFullTileList((0,),range(3),(0,),1000,1000,(500,500))
    #     tiles = [(0,0,0,(0,0,500,500)),(0,0,0,(500,0,500,500)),(0,0,0,(0,500,500,500)),(0,0,0,(500,500,500,500)),
    #              (0,1,0,(0,0,500,500)),(0,1,0,(500,0,500,500)),(0,1,0,(0,500,500,500)),(0,1,0,(500,500,500,500)),
    #              (0,2,0,(0,0,500,500)),(0,2,0,(500,0,500,500)),(0,2,0,(0,500,500,500)),(0,2,0,(500,500,500,500)),]
    #     self.assertTrue(np.array_equal(rv, tiles))
    #     self.conn.close()

    # createTileListFromImg too simple to fail
        
    # cannot find hist with shapes in idr lol
    # def test_getShapesAsPoints(self):
    #     self.conn.close()
        

    def test_createPolygon(self):
        points = [(0,0),(0,0),(1,1),(1,1),(0,1),(0,1)]

        expected = [(1,1),(2,2),(1,2)]

        polygon = omero_util.createPolygon(points, stride=2, x_offset=1, y_offset=1, z=1, t=0, comment="test", rgb=(1,123,222))

        self.assertEqual(polygon.getTheZ().getValue(),1)
        self.assertEqual(polygon.getTheT().getValue(),0)
        self.assertEqual(polygon.getTextValue().getValue(), "test")

        xy = []
        pointStrArr = polygon.getPoints()._val.split(" ")
        for i in range(0, len(pointStrArr)):
            coord=pointStrArr[i].split(",")
            xy.append((float(coord[0]), float(coord[1])))
        
        self.assertTrue(np.array_equal(expected, xy))

        rgba_int = polygon.getStrokeColor().getValue()
        if rgba_int < 0:    # convert from signed 32-bit int
            rgba_int = rgba_int + 2**32

        red   = (rgba_int >> 24) & 0xFF
        green = (rgba_int >> 16) & 0xFF
        blue  = (rgba_int >> 8)  & 0xFF
        alpha = rgba_int & 0xFF
        self.assertTrue(np.array_equal((1,123,222,255),(red,green,blue,alpha)))
        self.conn.close()
        
    # lazy
    # def test_createRoi(self):
    #     self.conn.close()
        
    # lazy
    # def test_downloadFileAnnotation(self):
    #     self.conn.close()
        

    # no scripts on idr
    # def test_getScriptByName(self):
    #     self.conn.close()
        

    # cannot (at least should not) upload to idr
    # def test_uploadFileAsAnnotation(self):
    #     self.conn.close()
        
    # hard to test
    # def test_idsToImageIds(self):
        # ids as of 05-6-23
        # proj_ids = [2451]
        # expected = [14239639, 14239640, 14239641, 14239632, 14239633,14239634,14239635,14239636,14239637,14239638] 
        # rv = omero_util.idsToImageIds(self.conn,'project', proj_ids)
        # print(rv)
        # print(expected)
        # self.assertTrue(np.array_equal(expected,rv ))
        # self.conn.close()
        
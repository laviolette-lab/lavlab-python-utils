"""
Helper functions that handle high-level operations and translating asynchronous requests for easy development.
"""
import os
import asyncio
import tempfile

import numpy as np
from skimage import draw

from collections.abc import AsyncGenerator

from omero.gateway import _BlitzGateway, ImageWrapper, FileAnnotationWrapper, ShapeWrapper
import omero.model.enums as omero_enums
from omero.rtypes import rint, rstring

from omero_model_RoiI import RoiI
from omero_model_PolygonI import PolygonI
from omero_model_EllipseI import EllipseI
from omero_model_PolygonI import PolygonI
from omero_model_RectangleI import RectangleI
from omero_model_FileAnnotationI import FileAnnotationI

from lavlab import omero_asyncio
from lavlab.python_util import chunkify, merge_async_iters, interlace_lists, lookup_filetype_by_name, FILETYPE_DICTIONARY, save_image_binary, rgba_to_int, resize_image_array

PARALLEL_STORE_COUNT=4
"""Number of pixel stores to be created for an image."""

OMERO_DICTIONARY = {
    # TODO add real pixeltype support
    "PIXEL_TYPES": {
        omero_enums.PixelsTypeint8: np.int8,
        omero_enums.PixelsTypeuint8: np.uint8,
        omero_enums.PixelsTypeint16: np.int16,
        omero_enums.PixelsTypeuint16: np.uint16,
        omero_enums.PixelsTypeint32: np.int32,
        omero_enums.PixelsTypeuint32: np.uint32,
        omero_enums.PixelsTypefloat: np.float32,
        omero_enums.PixelsTypedouble: np.float64,
    },
    "BYTE_MASKS": {
        np.uint8: {
            "RED": 0xFF000000, 
            "GREEN": 0xFF0000, 
            "BLUE": 0xFF00,  
            "ALPHA": 0xFF
        }
    },
    "SKIMAGE_FORMATS": FILETYPE_DICTIONARY["SKIMAGE_FORMATS"]
}
"""
Dictionary for converting omero info into python equivalents.

```
OMERO_DICTIONARY = {
"PIXEL_TYPES": {
    omero_enums.PixelsTypeint8: np.int8,
    omero_enums.PixelsTypeuint8: np.uint8,
    omero_enums.PixelsTypeint16: np.int16,
    omero_enums.PixelsTypeuint16: np.uint16,
    omero_enums.PixelsTypeint32: np.int32,
    omero_enums.PixelsTypeuint32: np.uint32,
    omero_enums.PixelsTypefloat: np.float32,
    omero_enums.PixelsTypedouble: np.float64,
},
"BYTE_MASKS": {
    np.uint8: {
        "RED": 0xFF000000, 
        "GREEN": 0xFF0000, 
        "BLUE": 0xFF00,  
        "ALPHA": 0xFF
    }
},
"SKIMAGE_FORMATS": FILETYPE_DICTIONARY["SKIMAGE_FORMATS"]
}
```

See Also
--------
lavlab.python_utils.FILETYPE_DICTIONARY : Contains file extensions and mimetypes for commonly used files.
"""


#
## IMAGE DATA
#
def getTiles(img: ImageWrapper, tiles: list[tuple[int,int,int,tuple[int,int,int,int]]],
            resLvl: int=None, rps_bypass=True) -> AsyncGenerator[tuple[np.ndarray,tuple[int,int,int,tuple[int,int,int,int]]]]:
    """
Asynchronous tile generator. 

Creates and destroys parallel asynchronous RawPixelsStores to await multiple tiles at once.

Parameters
----------
img: omero.gateway.ImageWrapper
    Omero Image object from conn.getObjects().
tiles: list[ (z,c,t,(x,y,w,h) ]), ... ]
    list of tiles to gather.
resLvl: default: -1
    what resolution level are these tiles on, default highest res
rps_bypass: default: True
    passthrough for rawPixelsStore.setPixelsId(pixels.id, rps_bypass)

Returns
-------
Async tile generator
    python async generator that yields omero tiles as numpy arrays

Examples
--------
```
import asyncio
async def work(img, tiles, res_lvl, dims):
    bin = np.zeros(dims, np.uint8)
    async for tile, (z,c,t,coord) in getTiles(img,tiles,res_lvl):
        bin [
            coord[1]:coord[1]+coord[3],
            coord[0]:coord[0]+coord[2], 
            c ] = tile 
    return bin
asyncio.run(work(img, tiles, res_lvl, dims))
```
"""
    # tile request group
    async def work(id, tiles, resLvl):
        # create and init rps for this group
        rps = await session.createRawPixelsStore()
        await rps.setPixelsId(id,rps_bypass)

        # set res and get default res level if necessary
        if resLvl is None:
            resLvl = await rps.getResolutionLevels()
        await rps.setResolutionLevel(resLvl)

        # request and return tiles
        i=1
        tile_len=len(tiles)
        for z,c,t,tile in tiles:
            rv = np.frombuffer(await rps.getTile(z,c,t,*tile), dtype=np.uint8)
            rv.shape=tile[3],tile[2]
            if i == tile_len: await rps.close()
            else: i+=1
            yield rv, (z,c,t,tile)


    # force async client
    session =  omero_asyncio.AsyncSession(img._conn.c.sf)

    # create parallel raw pixels stores
    jobs=[]
    for chunk in chunkify(tiles, int(len(tiles)/PARALLEL_STORE_COUNT)+1):
        jobs.append(work(img.getPrimaryPixels().getId(), chunk, resLvl))
    return merge_async_iters(*jobs)

def getDownsampledYXDimensions(img: ImageWrapper, downsample_factor: int) -> tuple[int,int]:
    """
Returns yx (rows,columns) dimensions of given image at the downsample.

Parameters
----------
img: omero.gateway.ImageWrapper
    Omero Image object from conn.getObjects().
downsample_factor: int
    Takes every nth pixel from the base resolution.

Returns
-------
int
    img.getSizeY() / downsample_factor
int 
    img.getSizeX() / downsample_factor
"""
    return (int(img.getSizeY() / downsample_factor),
            int(img.getSizeX() / downsample_factor))

def getDownsampleFromDimensions(base_shape:tuple[int,...], sample_shape:tuple[int,...]) -> tuple[float,...]:
    """
Essentially an alias for np.divide().

Finds the ratio between a base array shape and a sample array shape by dividing each axis.

Parameters
----------
base_shape: tuple(int)*x
    Shape of the larger image. (base_nparray.shape)
sample_shape: tuple(int)*x
    Shape of the smaller image. (sample_nparray.shape)

Raises
------
AssertionError
    Asserts that the input shapes have the same amount of axes

Returns
-------
tuple(int)*x
    Returns a tuple containing the downsample factor of each axis for the sample array.

""" 
    assert len(base_shape) == len(sample_shape)
    return np.divide(base_shape, sample_shape)


def getClosestResolutionLevel(img: ImageWrapper, dim: tuple[int,int]) -> tuple[int,tuple[int,int,int,int]]:
    """
Finds the closest resolution to desired resolution.

Parameters
----------
img: omero.gateway.ImageWrapper or RawPixelsStore
    Omero Image object from conn.getObjects() or initialized rps
dim: tuple[int, int]
    tuple containing desired y,x dimensions. 

Returns
-------
int
    resolution level to be used in rps.setResolution() 
tuple[int,int,int,int]
    height, width, tilesize_y, tilesize_x of closest resolution
    """
    # if has getResolutionLevels method it's a rawpixelstore
    if type(img) is hasattr(img, 'getResolutionLevels'): rps = img
    # else assume it's an ImageWrapper obj and use it to create an rps
    else:
        rps = img._conn.createRawPixelsStore()
        rps.setPixelsId(img.getPrimaryPixels().getId(), True)
        close_rps=True
        
    # get res info
    lvls = rps.getResolutionLevels()
    resolutions = rps.getResolutionDescriptions()

    # search for closest res
    for i in range(lvls) :
        res=resolutions[i]
        currDif=(res.sizeX-dim[1],res.sizeY-dim[0])
        # if this resolution's difference is negative in either axis, the previous resolution is closest
        if currDif[0] < 0 or currDif[1] < 0:

            rps.setResolutionLevel(lvls-i)
            tileSize=rps.getTileSize()

            if close_rps is True: rps.close()

            return (lvls-i, (resolutions[i-1].sizeY,resolutions[i-1].sizeX,
                             tileSize[1], tileSize[0]))
        

def getImageAtResolution(img: ImageWrapper, yx_dim: tuple[int,int], channels:list[int]=None) -> np.ndarray:
    """
Gathers tiles and scales down to desired resolution.

Warns
-------
Out of Memory issues ahead! Request a reasonable resolution!

Parameters
----------
img: omero.gateway.ImageWrapper
    Omero Image object from conn.getObjects().
yx_dim: tuple(y,x)
    Tuple of desired dimensions (row, col)
channels: tuple(int,...), default: all channels
    Array of channels to gather.
    To grab only blue channel: channels=(2) 

Returns
-------
np.ndarray 
    Array of rbg values for given img.
    """
    async def work(img, tiles, res_lvl, current_dims, des_shape):
        bin = np.zeros(current_dims, np.uint8)
        async for tile, (z,c,t,coord) in getTiles(img,tiles,res_lvl):
            bin [
                coord[1]:coord[1]+coord[3],
                coord[0]:coord[0]+coord[2], 
                c ] = tile 
        if bin.shape != des_shape:
            bin = resize_image_array(bin,(yx_dim[1],yx_dim[0]))
        return bin
    
    res_lvl, dims = getClosestResolutionLevel(img, yx_dim)

    if channels is None:
        channels = range(img.getSizeC())
    
    if len(channels) > 1: 
        des_shape = (*yx_dim, len(channels))
        current_dims = (dims[0],dims[1],len(channels))
    else: 
        des_shape = yx_dim
        current_dims = dims[:1]
        
    tiles = createFullTileList([0,],channels,[0,],dims[1],dims[0],(dims[3],dims[2]))
    return asyncio.run(work(img, tiles, res_lvl, current_dims, des_shape))


def getLargeRecon(img:ImageWrapper, downsample_factor, workdir='./', save_format="JPEG"):
    """
Checks OMERO for a pregenerated large recon, if none are found, it will generate and upload one.

Parameters
----------
img: omero.gateway.ImageWrapper
    Omero Image object from conn.getObjects().
downsample_factor: int
    Which large recon size to get.

Returns
-------
omero.gateway.AnnotationWrapper
    remote large recon object
str
    local path to large recon
    """
    NS = "LargeRecon."+str(downsample_factor)
    EXT, MIME = OMERO_DICTIONARY["SKIMAGE_FORMATS"][save_format]
    recon = img.getAnnotation(NS)
    if recon is None:
        name = img.getName()
        sizeX = img.getSizeX()
        sizeY = img.getSizeY()

        print(f"No large recon {downsample_factor} for img: {name} Generating...")

        yx_dim=getDownsampledYXDimensions(img, downsample_factor)
        reconPath = workdir + os.sep + f"LR{downsample_factor}_{name.replace('.ome.tiff',EXT)}"
        recon = img.getAnnotation(NS)
        
        if recon is None: 
                print(f"Downsampling: {name} from {(sizeY,sizeX)} to {yx_dim}")
                reconBin = getImageAtResolution(img, yx_dim)
                
                if save_format == 'JPEG': 
                    jpeg=True 
                else: 
                    jpeg=False

                save_image_binary(reconPath,reconBin, jpeg)
                
                print("Downsampling Complete! Uploading to OMERO...")
                recon = img._conn.createFileAnnfromLocalFile(reconPath, mimetype=MIME, ns=NS)
                img.linkAnnotation(recon)
        else:
            reconPath = downloadFileAnnotation(recon, workdir)

    return recon, reconPath

#
## TILES
#
def createTileList2D(z:int, c:int, t:int, size_x:int, size_y:int,
        tile_size:tuple[int,int]) -> list[tuple[int,int,int,tuple[int,int,int,int]]]:
    """
Creates a list of tile coords for a given 2D plane (z,c,t)

Notes
-----
Tiles are outputed as (z,c,t,(x,y,w,h)) as this is the expected format by omero python bindings.\n
This may cause confusion as numpy uses rows,cols (y,x) instead of x,y. \n
Tile lists generated by lavlab.omero_utils are compatible with omero_util and official omero functions.

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
    tileList = []
    width, height = tile_size 
    for y in range(0, size_y, height):
        width, height = tile_size # reset tile size
        # if tileheight is greater than remaining pixels, get remaining pixels
        if size_y-y < height: height = size_y-y
        for x in range(0, size_x, width):
        # if tilewidth is greater than remaining pixels, get remaining pixels
            if size_x-x < width: width = size_x-x
            tileList.append((z,c,t,(x,y,width,height)))
    return tileList


def createFullTileList(z_indexes: int, channels: int, timepoints: int, width: int, height:int, 
        tile_size:tuple[int,int], weave=False) -> list[tuple[int,int,int,tuple[int,int,int,int]]]:
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
    tileList = []
    if weave is True: 
        origC = channels
        channels = (0)
    for z in z_indexes:
        for c in channels:
            for t in timepoints:
                if weave is True:
                    tileChannels = []
                    for channel in origC:
                        tileChannels.append(createTileList2D(z,channel,t,width, height, tile_size)) 
                    tileList.extend(interlace_lists(tileChannels))
                else:
                    tileList.extend(createTileList2D(z,c,t,width, height, tile_size))
        
    return tileList

def createTileListFromImage(img: ImageWrapper, rgb=False, include_z=True, include_t=True) -> list[int,int,int,tuple[int,int,int,int]]:
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
    width = range(img.getSizeX()) 
    height = range(img.getSizeY())
    z_indexes = range(img.getSizeZ())
    timepoints = range(img.getSizeT())
    channels = range(img.getSizeC())

    img._prepareRenderingEngine()
    tile_size = img._re.getTileSize()
    img._re.close()

    if include_t is False: timepoints = [0,]
    if include_z is False: z_indexes = [0,]

    return createFullTileList(z_indexes,channels,timepoints,width,height,tile_size, rgb)


#
## ROIS
#
def getShapesAsPoints(img: ImageWrapper, point_downsample=4, img_downsample=1, 
                      roi_service=None) -> list[tuple[int, tuple[int,int,int], tuple[np.ndarray, np.ndarray]]]:
    """
Gathers Rectangles, Polygons, and Ellipses as a tuple containing the shapeId, its rgb val, and a tuple of yx points of its bounds.

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
returns: list[shape.id, (r,g,b), (row_points, column_points))]
    list of tuples containing a shape's id, rgb value, and a tuple of row and column points
    """
    if roi_service is None:
        roi_service=img._conn.getRoiService()
        close_roi=True

    sizeX = img.getSizeX() / img_downsample
    sizeY = img.getSizeY() / img_downsample
    yx_shape = (sizeY,sizeX)

    result = roi_service.findByImage(img.getId(), None)

    shapes=[]
    for roi in result.rois:
        points= None
        for shape in roi.copyShapes():
            if type(shape) == RectangleI:
                x = float(shape.getX().getValue()) / img_downsample
                y = float(shape.getY().getValue()) / img_downsample
                w = float(shape.getWidth().getValue()) / img_downsample
                h = float(shape.getHeight().getValue()) / img_downsample
                points = draw.rectangle_perimeter((y,x),(y+h,x+w), shape=yx_shape)

            if type(shape) == EllipseI:
                points = draw.ellipse_perimeter(float(shape._y._val / img_downsample),float(shape._x._val / img_downsample),
                            float(shape._radiusY._val / img_downsample),float(shape._radiusX._val / img_downsample),
                            shape=yx_shape)
            
            if type(shape) == PolygonI:
                pointStrArr = shape.getPoints()._val.split(" ")

                y = []
                x = []
                for i in range(0, len(pointStrArr)):
                    coordList=pointStrArr[i].split(",")
                    y.append(float(coordList[1]) / img_downsample)
                    x.append(float(coordList[0]) / img_downsample)

                points = draw.polygon_perimeter(y, x, shape=yx_shape)

            if points is not None:
                color_val = shape.getStrokeColor()._val
                masks = OMERO_DICTIONARY["BYTE_MASKS"][np.uint8]
                red = (color_val & masks["RED"]) >> 24  
                green = (color_val & masks["GREEN"]) >> 16  
                blue = (color_val & masks["BLUE"]) >> 8 
                points=(points[0][::point_downsample], points[1][::point_downsample])
                
                shapes.append((shape.getId()._val, (red,green,blue), points))

    if not shapes : # if no shapes in shapes return none
        return None
    
    if close_roi: roi_service.close()

    # make sure is in correct order
    return sorted(shapes)


def createPolygon(contour:tuple[np.ndarray, np.ndarray], x_offset=0, y_offset=0, z=None, t=None, comment=None, rgb=(0,0,0)) -> PolygonI:
    """ 
Creates a local omero polygon obj from a list of points, and parameters.
    
Parameters
----------
contour: tuple[rows, cols]
    Expects contour as outputed by skimage.measure.find_contours
x_offset: int, Default: 0
    Inherited from where I ripped this code. Lets you shift coords I guess.
y_offset: int, Default: 0
    Inherited from where I ripped this code. Lets you shift coords I guess.
z: int, optional
    Allows polygon to exist in a specific z_index for multi dimensional ROIs.
t: int, optional
    Allows polygon to exist in a specific timepoint for multi dimensional ROIs.
comment: str, optional
    Description of polygon, recommended to use to keep track of which programs generate which shapes.
rgb: tuple[int,int,int], Default: (0,0,0) 
    What color contour should this polygon's contour be.
    
Returns
-------
omero_model_PolygonI.PolygonI
    Local Omero Polygon object, likely needs to linked to an ROI
    """
    stride = 64
    coords = []
    # points in contour are adjacent pixels, which is too verbose
    # take every nth point
    for count, xy in enumerate(contour):
        if count%stride == 0:
            coords.append(xy)
    if len(coords) < 2:
        return
    points = ["%s,%s" % (xy[1] + x_offset, xy[0] + y_offset) for xy in coords]
    points = ", ".join(points)
    polygon = PolygonI()
    if z is not None:
        polygon.theZ = rint(z)
    if t is not None:
        polygon.theT = rint(t)
    if comment is not None:
        polygon.setTextValue(rstring(comment))
    polygon.strokeColor = rint(rgba_to_int(*rgb))
    polygon.points = rstring(points)
    return polygon

def createRoi(img: ImageWrapper, shapes: list[ShapeWrapper]):
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
    roi.setImage(img._obj)
    for shape in shapes:
        roi.addShape(shape)
    return roi


def drawShapes(input_img: np.ndarray, shape_points:tuple[int,tuple[int,int,int],tuple[np.ndarray, np.ndarray]]) -> None:
    """
Draws a list of shape points onto the input numpy array.

Warns
-------
NO SAFETY CHECKS! MAKE SURE input_img AND shape_points ARE FOR THE SAME DOWNSAMPLE FACTOR!

Parameters
----------
input_img: np.ndarray
    3 channel numpy array
shape_points: tuple(int, tuple(int,int,int), tuple(row, col))
    output from getShapesAsPoints

Returns
-------
``None``
    """
    for id, rgb, points in shape_points:
        rr,cc = draw.polygon(*points)
        input_img[rr,cc]=rgb




# TODO SLOW AND broken for rgb = 0,0,0 annotations
# def getShapesAsMasks(img: ImageWrapper, downsample: int, bool_mask=True, 
#                      point_downsample=4, roi_service=None) -> list[np.ndarray]:
#     """
# Gathers Rectangles, Polygons, and Ellipses as masks for the image at the given downsampling
# Converts rectangles and ellipses into polygons (4 rectangle points into an array of points on the outline)
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

# 
## FILES
#  

def downloadFileAnnotation(file_annot: FileAnnotationWrapper, outdir=".") -> str:
    """
Downloads FileAnnotation from OMERO into a local directory.

Parameters
----------
file_annot: omero.gateway.FileAnnotationWrapper
    Remote Omero File Annotation object.
out_dir: str, Default: '.' 
    Where to download this file.

Returns
-------
str 
    String path to downloaded file.
    """
    path = os.path.abspath(outdir) + os.sep + file_annot.getFile().getName()
    print(f"Downloading {path}...")
    with open(path, 'wb') as f:
        for chunk in file_annot.getFileInChunks():
            f.write(chunk)
    print(f"{path} downloaded!")
    return path


# TODO checkUserScripts
def getScriptByName(conn: _BlitzGateway, fn: str, absolute=False, checkUserScripts=False) -> int:
    """
Searches for an omero script in the host with the given name.

Parameters
----------
conn: omero.gateway.BlitzGateway
    An Omero BlitzGateway with a session.
fn: str
    Name of remote Omero.Script
absolute: bool, Default: False
    Absolute uses omero's getScriptID(). This method does not accept wildcards and requires a path.
    Default will get all remote script names and compare the filename using python equivelency (allows wildcards).
checkUserScripts: bool, Default: False
    Not implemented.

Returns
-------
int
    Omero.Script Id
    """
    if checkUserScripts: print("getScriptByName not fully implemented! May cause unexpected results!")
    scriptService=conn.getScriptService()
    try:
        if absolute is True: return scriptService.getScriptID(fn)
        for script in scriptService.getScripts():
            if script.getName().getValue() == fn:
                return script.getId().getValue()
    finally:
        scriptService.close()

def uploadFileAsAnnotation(parent_obj: ImageWrapper, file_path: str, namespace:str, 
        mime:str=None, overwrite=True) -> FileAnnotationI:
    """
Uploads a given filepath to omero as an annotation for parent_obj under namespace.

parent_obj: omero.gateway.ParentWrapper
    Object that should own the annotation. (typically an ImageWrapper)
file_path: str
    Local path of file to upload as annotation.
namespace: str 
    Remote namespace to put the file annotation
mime: str, optional 
    Mimetype for filetype. If None this will be guessed based on file extension and filetype dictionary
overwrite: bool, Default: True
    Overwrites existing file annotation in this namespace.
return: omero.gateway.FileAnnotationWrapper
    Uploaded FileAnnotation object.
    """
    conn = parent_obj._conn

    # if no mime provided try to parse from filename, if cannot, assume plaintext
    if mime is None:
        mime = FILETYPE_DICTIONARY.get(
            lookup_filetype_by_name(file_path),
            FILETYPE_DICTIONARY["GENERIC_FILES"]["TXT"]
        )["MIME"]
        
    # if overwrite is true and an annotation already exists in this namespace, delete it
    if overwrite is True: 
        obj = parent_obj.getAnnotation(namespace)
        if obj is not None:
            conn.deleteObjects('Annotation',[obj.id], wait=True)

    # create, link, and return new annotation
    annot_obj = conn.createFileAnnfromLocalFile(file_path, mimetype=mime, ns=namespace)
    parent_obj.linkAnnotation(annot_obj)
    return annot_obj

#
## PARSING
#
def idsToImageIds(conn: _BlitzGateway, dType: str, rawIds: list[int]) -> list[int]:
    """
Gathers image ids from given OMERO objects. For Project and Dataset ids. Takes Image ids too for compatibility.

Parameters
----------
conn: omero.gateway.BlitzGateway
    An Omero BlitzGateway with a session.
dType: str 
    String data type, should be one of: 'Image','Dataset', or 'Project'
rawIds: list[int]
    ids for datatype
return: list[int]
    List of all found Image ids
    """
    if dType != "Image" :
        # project to dataset
        if dType == "Project" :
            projectIds = rawIds; rawIds = []
            for projectId in projectIds :
                for dataset in conn.getObjects('dataset', opts={'project' : projectId}) :
                    rawIds.append(dataset.getId())
        # dataset to image
        ids=[]
        for datasetId in rawIds :
            for image in conn.getObjects('image', opts={'dataset' : datasetId}) :
                ids.append(image.getId())
    # else rename image ids
    else : 
        ids = rawIds
    return ids

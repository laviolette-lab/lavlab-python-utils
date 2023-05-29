"""
Helper functions that handle high-level operations and translating asynchronous requests for easy development.
"""
import os
import asyncio
from collections.abc import AsyncGenerator

import numpy as np
from skimage import draw
from PIL import Image
from omero.gateway import _BlitzGateway, ImageWrapper, FileAnnotationWrapper
from omero_model_EllipseI import EllipseI
from omero_model_PolygonI import PolygonI
from omero_model_RectangleI import RectangleI
from omero_model_FileAnnotationI import FileAnnotationI
import omero.model.enums as omero_enums

from lavlab import omero_asyncio
from lavlab.python_util import chunkify, merge_async_iters, interlaceLists, lookup_filetype_by_name, FILETYPE_DICTIONARY

PARALLEL_STORE_COUNT=4
"""Number of pixel stores to be created for an image"""

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
        "UINT_8": {
            "RED": 0xFF000000, 
            "GREEN": 0xFF0000, 
            "BLUE": 0xFF00,  
            "ALPHA": 0xFF
        }
    },
    "SKIMAGE_FORMATS": FILETYPE_DICTIONARY["SKIMAGE_FORMATS"]
}
"""
Dictionary for dealing with omero/pixel datatypes.\n
PIXEL_TYPES: { omero.PixelType: numpy.datatype }\n
BYTE_MASKS: { "UINT_8": { "RED": 0x... } }\n
SKIMAGE_FORMATS: alias FILETYPE_DICTIONARY["SKIMAGE_FORMATS"]
"""


#
## IMAGE DATA
#
def getTiles(img: ImageWrapper, tiles: list[tuple[int,int,int,tuple[int,int,int,int]]],
            resLvl: int=None, rps_bypass=True) -> AsyncGenerator[tuple[np.ndarray,tuple[int,int,int,tuple[int,int,int,int]]]]:
    """
Asynchronous tile generator. Creates and destroys parallel RawPixelsStores to request tiles.\n
img: omero.gateway.ImageWrapper\n
tiles: list of tiles (z,c,t,(x,y,w,h))\n
resLvl: what resolution level are these tiles on, default highest res\n
rps_bypass: alias for rawPixelsStore.setPixelsId(pixels.id, rps_bypass)\n
returns: async generator. \n
usage: async for nparray, zctxywh in getTiles(img, [zctxywh,...]):
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

def getDownsampledYXDimensions(img: ImageWrapper, downsample_factor: int):
    """Returns img yx dimensions after being divided by downsample factor"""
    return (int(img.getSizeY() / downsample_factor),
            int(img.getSizeX() / downsample_factor))

def getDownsampleFromDimensions(base_shape:tuple[int,...], sample_shape:tuple[int,...]) -> tuple[float,...]:
    """Essentially an alias for np.divide()"""
    assert len(base_shape) == len(sample_shape)
    return np.divide(base_shape, sample_shape)


def getClosestResolutionLevel(img: ImageWrapper, dim: tuple[int,int]
                              ) -> tuple[int,tuple[int,int,int,int]]:
    """
Finds the closest resolution to desired resolution.\n
Returns resolution level to be used in store.setResolution() and actual y,x dimensions of that resolution.\n
return value schema: ( level, (width,height,x_tile_size,y_tile_size) )
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
        
# https://docs.scipy.org/doc/scipy-1.2.1/reference/generated/scipy.misc.imresize.html
def resizeImage(input_array: np.ndarray, shape_xy: tuple[int,int], interpolation=Image.NEAREST):
    """
Resizes input image array to desired xy dimensions using PIL.Image.resize.\n
input_array: 2-3(+?) dimensional numpy array.\n
shape_xy: desired width and height of output.\n
interpolation: default nearest neighbor, PIL.Image.INTERPOLATION_TYPE\n
Returns: smaller np.ndarray
    """
    return np.asarray(Image.fromarray(input_array).
                      resize(shape_xy, interpolation), input_array.dtype)

def getImageAtResolution(img: ImageWrapper, yx_dim: tuple[int,int], channels:list[int]=None) -> np.ndarray:
    """
Gathers tiles and scales down to desired resolution.\n
Out of Memory issues ahead! Request a reasonable resolution!\n
img: ImageWrapper\n
yx_dim: tuple of desired dimensions (row, col)\n
channels: array of channels to gather, to grab only blue channel: channels=(2) default: all channels\n
returns: np array of color values for given img
    """
    async def work(img, tiles, res_lvl, current_dims, des_shape):
        bin = np.zeros(current_dims, np.uint8)
        async for tile, (z,c,t,coord) in getTiles(img,tiles,res_lvl):
            bin [
                coord[1]:coord[1]+coord[3],
                coord[0]:coord[0]+coord[2], 
                c ] = tile 
        if bin.shape != des_shape:
            bin = resizeImage(bin,(yx_dim[1],yx_dim[0]))
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

def applyMask(img_bin: np.ndarray, mask_bin: np.ndarray, where=None):
    """Essentially an alias for np.where()"""
    if where is None:
        where=mask_bin!=0
    return np.where(where, mask_bin, img_bin)

#
## TILES
#
def createTileList2D(z:int, c:int, t:int, size_x:int, size_y:int, 
        tile_size:tuple[int,int]) -> list[tuple[int,int,int,tuple[int,int,int,int]]]:
    """
Creates a list of tile coords for a given plane (z,c,t)\n
z: z index\n
c: channel\n
t: timepoint\n
size_x: width of full image\n
size_y: height of full image\n
tile_size: tuple(desired_tile_width, desired_tile_height)\n
Return: [(z,c,t(0,0,tile_size[0],tile_size[1])), ...]
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
Creates a list of all tiles for given dimensions.\n
size_z: z index\n
size_c: channel\n
size_t: timepoint\n
size_x: width of full image\n
size_y: height of full image\n
tile_size: tuple(desired_tile_width, desired_tile_height)\n
rgb: Interlace tiles from each channel vs default seperate channels.\n
    Default: False. \n
    Example False: [0,0,0,tile],[0,0,0,tile2],...[0,1,0,tile],...\n
    Example True: [0,0,0,tile],[0,1,0,tile],[0,2,0,tile]... \n
Return: [(z,c,t(0,0,tile_size[0],tile_size[1])), ...]
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
                    tileList.extend(interlaceLists(tileChannels))
                else:
                    tileList.extend(createTileList2D(z,c,t,width, height, tile_size))
        
    return tileList

def createTileListFromImage(img: ImageWrapper, rgb=False, include_z=True, include_t=True) -> list[int,int,int,tuple[int,int,int,int]]:
    """
Generates a list of tiles from an omero:model:Image object.
img: omero.model.ImageWrapper
rgb: Puts tile channels next to each other.
    Default: False.\n
    Example False: [0,0,0,tile],[0,0,0,tile2],...[0,1,0,tile],...\n
    Example True: [0,0,0,tile],[0,1,0,tile],[0,2,0,tile]... \n
include_z: get tiles for z indexes\n
include_t: get tiles for timepoints\n
returns: [(z,c,t(0,0,tile_size[0],tile_size[1])), ...]
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
Gathers Rectangles, Polygons, and Ellipses as a tuple containing the shapeId, its rgb val, and a tuple of yx points of its bounds.\n
img: ImageWrapper\n
point_downsample: grab every nth point, default: 4\n
img_downsample: how much to scale roi points\n
roi_service: allows roiservice passthrough for performance\n
returns: [(shape.id, (r,g,b), (row_points, column_points)),...]
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
                masks = OMERO_DICTIONARY["BYTE_MASKS"]["UINT_8"]
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

def drawShapes(input_img, shape_points):
    """
Draws a list of shape points (from getShapesAsPoints) onto a given numpy array.\n
NO SAFETY CHECKS! MAKE SURE input_img AND shape_points ARE FOR THE SAME DOWNSAMPLE FACTOR!\n
input_img: 3 channel numpy array\n
shape_points: [(shape.id, (r,g,b), (row_points, column_points)),...]
    """
    for id, rgb, points in shape_points:
        rr,cc = draw.polygon(*points)
        input_img[rr,cc]=rgb

# TODO SLOW AND broken for rgb = 0,0,0 annotations
# def getShapesAsMasks(img: ImageWrapper, downsample: int, bool_mask=True, 
#                      point_downsample=4, roi_service=None) -> list[np.ndarray]:
#     """
# Gathers Rectangles, Polygons, and Ellipses as masks for the image at the given downsampling\n
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
Downloads FileAnnotation from OMERO into a local directory.\n
file_annot: file annotation from previous processing script.\n
out_dir: where to download this file.\n
return: string path to downloaded file
    """
    path = os.path.abspath(outdir) + os.sep + file_annot.getFile().getName()
    print(f"Downloading {path}...")
    with open(path, 'wb') as f:
        for chunk in file_annot.getFileInChunks():
            f.write(chunk)
    print(f"{path} downloaded!")
    return path


# TODO checkUserScripts
def getScriptByName(conn: _BlitzGateway, fn: str, absolute=False, checkUserScripts=True) -> int:
    """
Searches for an omero script in the host with the given name.\n
conn
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
        mime:str=None, conn:_BlitzGateway=None, overwrite=True) -> FileAnnotationI:
    """
Uploads a given filepath to omero as an annotation for parent_obj under namespace.\n
parent_obj: object that should own an annotation (typically an ImageWrapper)\n
file_path: path of file for uploading\n
namespace: namespace to put future file annotation\n
mime: mimetype for filetype. if None this will be guessed based on file extension and filetype dictionary\n
conn: conn passthrough\n
overwrite: overwrite existing file annotation in this namespace.\n
return: created FileAnnotation object
    """
    if conn is None:
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
Gathers image ids from given OMERO objects.\n
For Project and Dataset ids. Takes Image ids too for compatibility.\n
conn: connected BlitzGateway\n
dType: string data type, should be one of: 'Image','Dataset', or 'Project'\n
rawIds: ids for datatype\n
return: [imageId,...]
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

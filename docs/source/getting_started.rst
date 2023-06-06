Getting Started
===============

This library is for simplifying omero.py operations. 
Below covers connecting to an omero server and using it with the library, 
as well as providing a few approaches to image processing.

* **Connecting omero-py to your omero server**

.. code-block:: python

   from omero.gateway import BlitzGateway
   conn = BlitzGateway(USERNAME, PASSWORD, host=HOST, port=PORT)
   conn.connect()


* **Large-Recon-based processing approach**

For a list of ids, download the large recon, then load image from path.

.. code-block:: python

   import numpy as np
   from tempfile import TemporaryDirectory
   from lavlab.omero_util import getLargeRecon

   SCRIPT_DOWNSAMPLE_FACTOR=10
   for img_id in img_ids:
      with TemporaryDirectory() as workdir
         img_obj = conn.getObject('Image', img_id)
         lr_obj, lr_img = getLargeRecon(img_obj, SCRIPT_DOWNSAMPLE_FACTOR, workdir)
         local_lr_path = lr_img.filename
         lr_bin = np.array(lr_img)
         # do work


* **Downsampled-in-memory processing approach**

Does the same as above, but instead uses the omero api to request tiles and does not use the disk.

.. code-block:: python

   import numpy as np
   from lavlab.omero_util import getDownsampledYXDimensions, getImageAtResolution

   SCRIPT_DOWNSAMPLE_FACTOR=10
   for img_id in img_ids:
      img_obj = conn.getObject('Image', img_id)
      lr_dims = getDownsampledYXDimensions(img_obj, SCRIPT_DOWNSAMPLE_FACTOR)
      lr_img = getImageAtResolution(img_obj, lr_dims)
      lr_bin = np.array(lr_img)
      # do work

* **Tile-based processing approach**

Operates on a list of tiles. Each tile is processed individually and asynchronously.
Can be tricky but it's the only way to operate on a full resolution WSI at a reasonable pace without >100GB RAM. 

.. code-block:: python

   from lavlab.omero_util import createTileListFromImage, getTiles

   async def example(img, tiles):
      async for tile, (z,c,t,coord) in getTiles(img,tiles,res_lvl):
         # do work
      return results

   SCRIPT_DOWNSAMPLE_FACTOR=10
   for img_id in img_ids:
      img_obj = conn.getObject('Image', img_id)
      tiles = createTileListFromImage(img_obj)
      results = asyncio.run(example(img_obj, tiles))


from concurrent.futures import as_completed

from lavlab.omero import LOGGER, ctx
from lavlab.omero.helpers import prepareIOThreads

def getTiles(img, all_tiles, res_lvl=None, rps_bypass=True, conn = None):
    LOGGER.info(f"Getting: {len(all_tiles)} tiles.")
    def work(z, c, t, coord):
        return ctx.thread_local.rps.getTile(z, c, t, *coord), (z, c, t, coord)

    prepareIOThreads(img, res_lvl, rps_bypass, conn=conn)
    with ctx.io_thread_pool as executor:
        futures = [executor.submit(work, z, c, t, coord) for z, c, t, coord in all_tiles]

        for future in as_completed(futures):
            raw_data, (z,c,t,coord) = future.result()
            processed_data = np.frombuffer(raw_data, dtype=np.uint8).reshape(coord[3],coord[2])
            yield processed_data, (z,c,t,coord)

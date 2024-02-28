from contextlib import contextmanager

from omero.gateway import ImageWrapper

from lavlab.omero import LOGGER, ctx

#
## GROUP CONTEXT HELPERS
#
def matchGroup(obj: ImageWrapper) -> int:
    group_id = obj.details.group.id.val
    if group_id != obj._conn.SERVICE_OPTS.getOmeroGroup():
        LOGGER.info(f"Switching to group with ID: {group_id}.")
        obj._conn.SERVICE_OPTS.setOmeroGroup(group_id)
    return group_id

@contextmanager
def useGroup(obj: ImageWrapper):
    original_group_id = obj._conn.SERVICE_OPTS.getOmeroGroup()
    new_group_id = obj.details.group.id.val

    if original_group_id != new_group_id:
        LOGGER.info(f"Switching to group with ID: {new_group_id}.")
        obj._conn.SERVICE_OPTS.setOmeroGroup(new_group_id)

    try:
        yield
    finally:
        if original_group_id != new_group_id:
            LOGGER.info(f"Reverting to original group with ID: {original_group_id}.")
            obj._conn.SERVICE_OPTS.setOmeroGroup(original_group_id)

#
## IO HELPERS
#
RPS_THREAD_PREFIX="RawPixelsStore-"
def prepareIOThreads(img, res_lvl, rps_bypass, conn = None):
    img = forceImageWrapper(conn, img)
    ctx.io_thread_prefix = RPS_THREAD_PREFIX
    ctx.io_thread_initializer = _initialize_threaded_rps
    ctx.io_thread_init_args  = (img._conn, img.getPrimaryPixels().getId(), res_lvl, rps_bypass)

def _initialize_threaded_rps(conn, pix_id, res_lvl, rps_bypass):
    rps = conn.c.sf.createRawPixelsStore()
    rps.setPixelsId(pix_id, rps_bypass)
    if res_lvl is None:
        res_lvl = rps.getResolutionLevels()
        res_lvl -= 1
    rps.setResolutionLevel(res_lvl)

    ctx.thread_local.rps = rps
    atexit.register(ctx.thread_local.rps.close)

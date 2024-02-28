import logging

from lavlab import ctx
from lavlab.login import prompt_kwargs

LOGGER = logging.getLogger(__name__)

def setOmeroLoggingLevel(level: logging._Level):
    """
Sets a given python logging._Level in all omero loggers.

Parameters
----------
level: logging._Level

Returns
-------
None
    """
    LOGGER.info(f"Setting Omero logging level to {level}.")
    for name in logging.root.manager.loggerDict.keys():
        if name.startswith('omero'):
            logging.getLogger(name).setLevel(level)

def parse_login_kwarg_prompts():
    known_kwargs = ["username", "password", "host", "port", "secure"]
    rv = {}
    for key, val in prompt_kwargs('Known kwargs: ' + ', '.join(known_kwargs) + '\n'):
        if key in known_kwargs:
            rv[key] = val
        else:
            LOGGER.warning(f"Unknown key: {key}")
    return rv


def login(conn, username="", password="", host="", port=None, secure=None):
    """
    Attempts to login to an omero server with multiple login flows.
    """
    promptSave = False
    needed_kwargs = []
    if host == "": # if host is directly provided
        if ctx.omero_url is None: # if no contextual url
            needed_kwargs.append("host")
    if username == "":
        needed_kwargs.append("username")



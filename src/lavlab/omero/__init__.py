"""OMERO Utility module"""

import logging

from omero.gateway import BlitzGateway  # type: ignore

import lavlab
from lavlab.login import AbstractServiceProvider

try:
    import idr  # type: ignore

    class IDRServiceProvider(AbstractServiceProvider):
        """
        Provides a connection to the IDR using idr-py, primarily for testing purposes.
        """

        def login(self) -> BlitzGateway:
            return idr.connection("idr.openmicroscopy.org", "public", "public")

except ImportError:
    pass


LOGGER = logging.getLogger(__name__)


# TODO service providers, forcing myself to wait until another update.
class OmeroServiceProvider(AbstractServiceProvider):
    """
    Provides a connection to a defined OMERO server using omero-py.
    """

    SERVICE = "OMERO"

    def login(self) -> BlitzGateway:
        """
        Logins into configured omero server.

        Returns
        -------
        BlitzGateway
            OMERO API gateway

        Raises
        ------
        RuntimeError
            Could not login to OMERO server.
        """
        details = lavlab.ctx.histology.service.copy()
        if details.get("username") is None or details.get("passwd") is None:
            username, password = self.cred_provider.get_credentials()
            details.update({"username": username, "passwd": password})

        conn = BlitzGateway(**details)
        if conn.connect():
            return conn
        raise RuntimeError("Unable to connect to OMERO server.")


def connect() -> BlitzGateway:
    """
    Uses the UtilContext to connect to the configured omero server

    Returns
    -------
    BlitzGateway
        omero api gateway
    """
    return lavlab.ctx.histology.get_service_provider().login()


def set_omero_logging_level(level: str):
    """
    Sets a given python logging._Level in all omero loggers.

    Parameters
    ----------
    level: logging._Level

    Returns
    -------
    None
    """
    LOGGER.info("Setting Omero logging level to %s.", level)
    for name in logging.root.manager.loggerDict.keys():  # pylint: disable=E1101
        if name.startswith("omero"):
            logging.getLogger(name).setLevel(level)

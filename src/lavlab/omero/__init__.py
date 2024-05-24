"""OMERO Utility module"""

import logging

from omero.gateway import BlitzGateway  # type: ignore

import lavlab
from lavlab.login import AbstractServiceProvider

LOGGER = lavlab.LOGGER.getChild("omero")


class OmeroServiceProvider(AbstractServiceProvider):  # pylint: disable=R0903
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
    if not lavlab.ctx.histology.service.get("name").upper() == "OMERO":
        raise RuntimeError("Service is not OMERO.")
    return lavlab.ctx.histology.service_provider.login()


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

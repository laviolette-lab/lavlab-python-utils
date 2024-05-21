import logging

from omero.gateway import BlitzGateway

from lavlab.login import AbstractServiceProvider
import lavlab

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
        details = lavlab.ctx.histology.service.copy()
        if details.get("username") is None or details.get("passwd") is None:
            username, password = self.cred_provider.get_credentials()
            details.update({"username": username, "passwd": password})

        conn = BlitzGateway(**details)
        if conn.connect():
            return conn


def connect() -> BlitzGateway:
    """
    Uses the UtilContext to connect to the configured omero server

    Returns
    -------
    BlitzGateway
        omero api gateway
    """
    return lavlab.ctx.histology.get_service_provider().login()


def setOmeroLoggingLevel(level: str):
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
        if name.startswith("omero"):
            logging.getLogger(name).setLevel(level)

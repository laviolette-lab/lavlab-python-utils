"""XNAT Utility module"""

import logging

import xnat  # type: ignore

import lavlab
from lavlab.login import AbstractServiceProvider

LOGGER = lavlab.LOGGER.getChild("xnat")


class XNATServiceProvider(AbstractServiceProvider):  # pylint: disable=R0903
    """
    Provides a connection to a defined XNAT server using xnatpy.
    """

    SERVICE = "XNAT"

    def login(self) -> xnat.XNATSession:
        """
        Logins into configured XNAT server.

        Returns
        -------
        xnat.XNATSession
            XNAT API session

        Raises
        ------
        RuntimeError
            Could not login to XNAT server.
        """
        details = lavlab.ctx.histology.service.copy()
        if details.get("username") is None or details.get("passwd") is None:
            username, password = self.cred_provider.get_credentials()
            details.update({"username": username, "passwd": password})

        # Convert field names to match xnatpy expectations
        connection_params = {
            "server": details.get("host"),
            "username": details.get("username"),
            "password": details.get("passwd"),
        }

        try:
            session = xnat.connect(**connection_params)
            return session
        except Exception as e:
            raise RuntimeError(f"Unable to connect to XNAT server: {e}") from e


def connect() -> xnat.XNATSession:
    """
    Uses the UtilContext to connect to the configured XNAT server

    Returns
    -------
    xnat.XNATSession
        XNAT API session
    """
    if not lavlab.ctx.histology.service.get("name").upper() == "XNAT":
        raise RuntimeError("Service is not XNAT.")
    return lavlab.ctx.histology.service_provider.login()


def set_xnat_logging_level(level: str):
    """
    Sets a given python logging._Level in all xnat loggers.

    Parameters
    ----------
    level: logging._Level

    Returns
    -------
    None
    """
    LOGGER.info("Setting XNAT logging level to %s.", level)
    for name in logging.root.manager.loggerDict.keys():  # pylint: disable=E1101
        if name.startswith("xnat"):
            logging.getLogger(name).setLevel(level)
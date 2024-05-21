"""Login utilities"""

import getpass
import json
import logging
import os
from typing import Optional

import keyring

import lavlab

LOGGER = logging.getLogger(__name__)  # TODO proper logger and logging


class KeychainCredentialsProvider:
    """
    A class to manage credentials using the keyring library.
    """

    def __init__(self, service: str):
        self._service = "lavlab-python-utils." + service
        self.store_credentials = True
        if keyring.get_keyring() is None:
            LOGGER.warning(
                "No keyring found, credentials will not be stored. \
                    Can likely be ignored in containerized environments."
            )
            self.store_credentials = False

    def _prompt_for_credentials(self, username=None) -> dict[str, str]:
        if username is None:
            username = input("Enter username: ")
        password = getpass.getpass("Enter password: ")
        return {"username": username, "password": password}

    def get_credentials(self) -> tuple[str, str]:
        """
        Potentially interactive credential flow, using keyring to store credentials.

        Returns
        -------
        tuple[str, str]
            username, password

        Raises
        ------
        RuntimeError
            this is raised if unable to determine credentials in non-interactive mode.
        """
        # Load credentials from keyring
        if self.store_credentials is False:
            raise RuntimeError("No keyring found, unable to retrieve credentials.")
        credentials_str = keyring.get_password(self._service, "credentials")
        if credentials_str:
            credentials = json.loads(credentials_str)
        else:
            credentials = []

        # If there's a configuration specifying the username, use it
        configured_username = keyring.get_password(self._service, "configured_username")
        if configured_username:
            for cred in credentials:
                if cred["username"] == configured_username:
                    return cred["username"], cred["password"]

        # If there's only one entry, use that
        if len(credentials) == 1:
            return credentials[0]["username"], credentials[0]["password"]

        # If there are multiple entries, try using the Unix username
        unix_username = os.getenv("USER")
        for cred in credentials:
            if cred["username"] == unix_username:
                return cred["username"], cred["password"]

        # If interactive, prompt for clarification
        if lavlab.ctx.noninteractive is not True:
            print("Unable to load credentials automatically, prompting user.")
            new_username = input(f"Enter username [{unix_username}]: ") or unix_username
            for cred in credentials:
                if cred["username"] == new_username:
                    return cred["username"], cred["password"]
            # New credential set
            new_credentials = self._prompt_for_credentials(new_username)
            credentials.append(new_credentials)
            keyring.set_password(self._service, "credentials", json.dumps(credentials))
            return new_credentials["username"], new_credentials["password"]

        # If non-interactive and unable to determine credentials, raise an error
        raise RuntimeError("Unable to determine credentials in non-interactive mode.")


class AbstractServiceProvider:
    """Abstract service provider. Uses a credential provider to login to a service
    Expected to be inherited by service-specific provider.
    """

    CREDENTIAL_PROVIDER = KeychainCredentialsProvider
    # SERVICE: Optional[str] = None # with entrypoints may not be required

    def __init__(self, **kwargs):
        self.cred_provider = self.CREDENTIAL_PROVIDER(str(self.__class__.__name__))
        self.service_kwargs = kwargs

    def login(self):
        """
        Uses keyword arguments from construction to login to the service.
        """
        raise NotImplementedError

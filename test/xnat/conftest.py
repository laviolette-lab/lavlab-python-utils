# pylint: skip-file
# type: ignore
import pytest
from unittest.mock import Mock, patch

import lavlab


@pytest.fixture
def mock_xnat_session():
    """
    A fixture that provides a mock XNAT session for testing.
    """
    mock_session = Mock()
    mock_session.projects = {}
    mock_session.experiments = {}
    return mock_session


@pytest.fixture
def xnat_service_config():
    """
    A fixture that provides XNAT service configuration for testing.
    """
    return {
        "name": "XNAT",
        "host": "https://example-xnat.org",
        "username": "testuser",
        "passwd": "testpass"
    }
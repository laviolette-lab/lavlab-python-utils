# pylint: skip-file
# type: ignore
import pytest
from unittest.mock import Mock, patch
import lavlab
from lavlab.xnat import XNATServiceProvider, connect, set_xnat_logging_level


class TestXNATServiceProvider:
    """Test the XNAT service provider."""

    def test_service_name(self):
        """Test that the service name is correctly set."""
        provider = XNATServiceProvider()
        assert provider.SERVICE == "XNAT"

    @patch('xnat.connect')
    def test_login_success(self, mock_connect, xnat_service_config):
        """Test successful login to XNAT server."""
        mock_session = Mock()
        mock_connect.return_value = mock_session
        
        provider = XNATServiceProvider()
        # Mock the credential provider to avoid keyring issues
        provider.cred_provider = Mock()
        provider.cred_provider.get_credentials.return_value = ("testuser", "testpass")
        
        # Mock lavlab context
        with patch.object(lavlab.ctx.histology, 'service', xnat_service_config):
            session = provider.login()
        
        assert session == mock_session
        mock_connect.assert_called_once_with(
            server="https://example-xnat.org",
            username="testuser", 
            password="testpass"
        )

    @patch('xnat.connect')
    def test_login_failure(self, mock_connect, xnat_service_config):
        """Test login failure handling."""
        mock_connect.side_effect = Exception("Connection failed")
        
        provider = XNATServiceProvider()
        provider.cred_provider = Mock()
        provider.cred_provider.get_credentials.return_value = ("testuser", "testpass")
        
        with patch.object(lavlab.ctx.histology, 'service', xnat_service_config):
            with pytest.raises(RuntimeError, match="Unable to connect to XNAT server"):
                provider.login()

    @patch('xnat.connect')
    def test_login_with_credentials_from_config(self, mock_connect, xnat_service_config):
        """Test login when credentials are already in config."""
        mock_session = Mock()
        mock_connect.return_value = mock_session
        
        provider = XNATServiceProvider()
        
        with patch.object(lavlab.ctx.histology, 'service', xnat_service_config):
            session = provider.login()
        
        assert session == mock_session
        mock_connect.assert_called_once_with(
            server="https://example-xnat.org",
            username="testuser",
            password="testpass"
        )


class TestXNATConnect:
    """Test the XNAT connect function."""

    @patch.object(lavlab.ctx.histology, 'service_provider')
    def test_connect_success(self, mock_provider, xnat_service_config):
        """Test successful connection through context."""
        mock_session = Mock()
        mock_provider.login.return_value = mock_session
        
        with patch.object(lavlab.ctx.histology, 'service', xnat_service_config):
            session = connect()
        
        assert session == mock_session
        mock_provider.login.assert_called_once()

    def test_connect_wrong_service(self):
        """Test connection failure when service is not XNAT."""
        wrong_config = {"name": "OMERO"}
        
        with patch.object(lavlab.ctx.histology, 'service', wrong_config):
            with pytest.raises(RuntimeError, match="Service is not XNAT"):
                connect()


class TestXNATLogging:
    """Test XNAT logging configuration."""

    @patch('logging.getLogger')
    @patch('logging.root.manager.loggerDict', {'xnat.test': Mock(), 'other.logger': Mock()})
    def test_set_xnat_logging_level(self, mock_get_logger):
        """Test setting logging level for XNAT loggers."""
        mock_logger = Mock()
        mock_get_logger.return_value = mock_logger
        
        set_xnat_logging_level("DEBUG")
        
        mock_get_logger.assert_called_with('xnat.test')
        mock_logger.setLevel.assert_called_with("DEBUG")
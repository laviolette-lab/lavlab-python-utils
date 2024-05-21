# pylint: skip-file
# type: ignore
# TODO idr (i assume) doesn't allow writes. we need a way to test write methods
import pytest
from lavlab.omero import connect
import lavlab


@pytest.fixture(scope="session")
def idr_client():
    """
    A fixture that provides access to the IDR client for testing.
    """
    conn = connect()
    yield conn
    conn.close()


@pytest.fixture
def sample_image(idr_client):
    """
    Fetches a sample image from the IDR for testing.
    """
    # Assuming a valid image ID; adjust this to an actual ID
    image_id = 7018
    image = idr_client.getObject("Image", image_id)
    return image

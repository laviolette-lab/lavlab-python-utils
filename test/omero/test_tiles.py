# pylint: skip-file
# type: ignore
from lavlab.omero.tiles import (
    get_tiles,
    create_tile_list_2d,
    create_full_tile_list,
    create_tile_list_from_image
)
def test_get_tiles(sample_image):
    # Create a list of tiles to fetch (e.g., z, c, t, coords)
    tiles = [(0, 0, 0, (0, 0, 100, 100))] * 100

    # Call the get_tiles function
    tile_gen = get_tiles(sample_image, tiles)

    # Verify the output
    tiles_list = list(tile_gen)
    assert len(tiles_list) == len(tiles)
    for tile, coords in tiles_list:
        assert tile.shape == (100, 100)
        assert coords in tiles


def test_create_tile_list_2d():
    z, c, t = 0, 0, 0
    size_x, size_y = 100, 100
    tile_size = (50, 50)
    
    # Generate tiles list
    tiles = create_tile_list_2d(z, c, t, size_x, size_y, tile_size)

    # Verify the output
    assert len(tiles) == 4  # 4 tiles for a 100x100 image with 50x50 tiles
    for tile in tiles:
        assert tile[0] == z
        assert tile[1] == c
        assert tile[2] == t
        assert isinstance(tile[3], tuple)


def test_create_full_tile_list():
    z_indexes = [0]
    channels = [0, 1]
    timepoints = [0]
    width, height = 100, 100
    tile_size = (50, 50)

    # Generate full tiles list
    tiles = create_full_tile_list(z_indexes, channels, timepoints, width, height, tile_size)

    # Verify the output
    assert len(tiles) == 8  # 8 tiles, considering two channels
    for tile in tiles:
        assert tile[0] in z_indexes
        assert tile[1] in channels
        assert tile[2] in timepoints
        assert isinstance(tile[3], tuple)


def test_create_tile_list_from_image(sample_image):
    # Generate tiles list from the sample image
    tiles = create_tile_list_from_image(sample_image)

    # Verify the output
    assert len(tiles) > 0
    for tile in tiles:
        assert isinstance(tile[3], tuple)
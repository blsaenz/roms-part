"""Unit + integration tests for join_schema.py: filename/attribute
cross-checking and complete-tile-set discovery, including the error paths
(missing node, mismatched nnodes, filename/attribute disagreement).
"""
from __future__ import annotations

import shutil

import netCDF4
import pytest

from roms_part.join_schema import TileSetError, discover_tile_set, extract_node_and_root


@pytest.mark.parametrize(
    "filename,nnodes,expected_root,expected_node",
    [
        ("root.3.nc", 5, "root.nc", 3),
        ("root.00.nc", 23, "root.nc", 0),
        ("root.0", 5, "root", 0),
        ("root.123.000.nc", 151, "root.123.nc", 0),
    ],
)
def test_extract_node_and_root(filename, nnodes, expected_root, expected_node):
    root, node = extract_node_and_root(filename, nnodes)
    assert (root, node) == (expected_root, expected_node)


def test_extract_node_and_root_rejects_wrong_width():
    # nnodes=5 expects 1 digit; a 2-digit segment shouldn't parse as valid.
    with pytest.raises(TileSetError):
        extract_node_and_root("root.03.nc", 5)


def test_discover_tile_set_happy_path(tiny_grid_factory, make_tiles):
    grid_path = tiny_grid_factory(24, 20, name="grid.nc")
    tile_paths = make_tiles(grid_path, 2, 2)
    tiles = discover_tile_set([str(p) for p in tile_paths])
    assert [t.node for t in tiles] == [0, 1, 2, 3]
    assert all(t.nnodes == 4 for t in tiles)


def test_discover_tile_set_missing_node(tiny_grid_factory, make_tiles):
    grid_path = tiny_grid_factory(24, 20, name="grid.nc")
    tile_paths = make_tiles(grid_path, 2, 2)
    with pytest.raises(TileSetError, match="missing node"):
        discover_tile_set([str(p) for p in tile_paths[:-1]])


def test_discover_tile_set_filename_attribute_mismatch(tiny_grid_factory, make_tiles, tmp_path):
    grid_path = tiny_grid_factory(24, 20, name="grid.nc")
    tile_paths = make_tiles(grid_path, 2, 2)
    # Rename tile 1 to claim it's tile 2 -- filename now disagrees with the
    # 'partition' attribute baked into the file's contents.
    renamed = tmp_path / "grid.2.nc"
    shutil.copy(tile_paths[1], renamed)
    mixed = [str(tile_paths[0]), str(renamed), str(tile_paths[2]), str(tile_paths[3])]
    with pytest.raises(TileSetError, match="filename encodes node"):
        discover_tile_set(mixed)


def test_discover_tile_set_rejects_non_tile_file(tiny_grid_factory):
    grid_path = tiny_grid_factory(24, 20, name="grid.nc")  # never partitioned
    with pytest.raises(TileSetError, match="no 'partition'"):
        discover_tile_set([str(grid_path)])


def test_discover_tile_set_duplicate_node(tiny_grid_factory, make_tiles, tmp_path):
    grid_path = tiny_grid_factory(24, 20, name="grid.nc")
    tile_paths = make_tiles(grid_path, 2, 2)
    # A second copy of tile 0's file, under its own correctly-formed name,
    # living in a different directory -- same node/nnodes per its
    # 'partition' attribute, so this is a genuine duplicate-node case.
    dup_dir = tmp_path / "dup_dir"
    dup_dir.mkdir()
    dup = dup_dir / tile_paths[0].name
    shutil.copy(tile_paths[0], dup)
    with pytest.raises(TileSetError, match="duplicate node"):
        discover_tile_set([str(tile_paths[0]), str(dup), str(tile_paths[1]), str(tile_paths[2]), str(tile_paths[3])])

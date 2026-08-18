"""Validate join_decomposition's reconstruction (global extent, edge flags,
write placement) against real `partit`-produced tiles, across a matrix of
NP_XI/NP_ETA and grid sizes including uneven-remainder cases -- mirroring
test_decomposition.py's approach but for the join-side (reverse) math,
which must recover the ORIGINAL grid size purely from tile metadata.
"""
from __future__ import annotations

import netCDF4
import pytest

from roms_part.decomposition import mpi_setup
from roms_part.dims import PARTITIONABLE_DIMS
from roms_part.join_decomposition import TileInfo, reconstruct_edges, reconstruct_global_extent

GRID_CASES = [
    (1, 1, 10, 12),
    (1, 4, 10, 42),
    (4, 1, 42, 10),
    (2, 2, 20, 20),
    (3, 5, 23, 51),
    (5, 3, 51, 23),
    (7, 7, 100, 100),
]


def _tile_infos_from_tiles(tile_paths):
    infos = {}
    for p in tile_paths:
        with netCDF4.Dataset(p) as ds:
            node, nnodes, xi_start, eta_start = (int(x) for x in ds.getncattr("partition"))
            local_dims = {k: len(v) for k, v in ds.dimensions.items() if k in PARTITIONABLE_DIMS}
        infos[node] = TileInfo(node=node, xi_start=xi_start, eta_start=eta_start, local_dims=local_dims)
    return infos


@pytest.mark.parametrize("np_xi,np_eta,xi_rho,eta_rho", GRID_CASES)
def test_reconstructs_original_extent(
    tiny_grid_factory, make_tiles, np_xi, np_eta, xi_rho, eta_rho
):
    grid_path = tiny_grid_factory(xi_rho, eta_rho, name="grid.nc")
    tile_paths = make_tiles(grid_path, np_xi, np_eta)

    infos = _tile_infos_from_tiles(tile_paths)
    extent = reconstruct_global_extent(list(infos.values()))
    assert (extent.xi_rho, extent.eta_rho) == (xi_rho, eta_rho)


@pytest.mark.parametrize("np_xi,np_eta,xi_rho,eta_rho", GRID_CASES)
def test_reconstructs_edges_matching_mpi_setup(
    tiny_grid_factory, make_tiles, np_xi, np_eta, xi_rho, eta_rho
):
    grid_path = tiny_grid_factory(xi_rho, eta_rho, name="grid.nc")
    tile_paths = make_tiles(grid_path, np_xi, np_eta)

    infos = _tile_infos_from_tiles(tile_paths)
    extent = reconstruct_global_extent(list(infos.values()))

    ground_truth = {n.node: n for n in mpi_setup(np_xi, np_eta, xi_rho, eta_rho)}
    for node, ti in infos.items():
        edges = reconstruct_edges(ti, extent)
        gt = ground_truth[node]
        assert (edges.western_edge, edges.eastern_edge, edges.southern_edge, edges.northern_edge) == (
            gt.western_edge,
            gt.eastern_edge,
            gt.southern_edge,
            gt.northern_edge,
        )

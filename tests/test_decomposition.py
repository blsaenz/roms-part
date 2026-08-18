"""Validate decomposition.mpi_setup / node_dim_extent against the real
compiled `partit` binary's actual output, across a matrix of NP_XI/NP_ETA
and grid sizes chosen so the interior extent sometimes divides evenly and
sometimes doesn't (exercising the off_xi/off_eta remainder-handling
branches).
"""
from __future__ import annotations

import subprocess

import netCDF4
import pytest

from roms_part.decomposition import mpi_setup, node_dim_extent
from roms_part.naming import insert_node_filename

GRID_CASES = [
    # (np_xi, np_eta, xi_rho, eta_rho)
    (1, 1, 10, 12),
    (1, 4, 10, 42),
    (4, 1, 42, 10),
    (2, 2, 20, 20),
    (3, 5, 23, 51),  # doesn't divide evenly either direction
    (5, 3, 51, 23),
    (7, 7, 100, 100),
    (4, 4, 674, 1322),  # matches the real production grid used in Phase 0
]


@pytest.mark.parametrize("np_xi,np_eta,xi_rho,eta_rho", GRID_CASES)
def test_matches_real_partit(
    partit_binary, tiny_grid_factory, tmp_path, np_xi, np_eta, xi_rho, eta_rho
):
    grid_path = tiny_grid_factory(xi_rho, eta_rho, name="grid.nc")
    out_dir = tmp_path / "out"
    out_dir.mkdir()

    subprocess.run(
        [partit_binary, str(np_xi), str(np_eta), grid_path.name],
        cwd=grid_path.parent,
        check=True,
        capture_output=True,
        text=True,
    )
    # partit.F always writes into the CWD, stripping the source directory.
    fortran_out_dir = grid_path.parent

    nodes = mpi_setup(np_xi, np_eta, xi_rho, eta_rho)
    nnodes = np_xi * np_eta
    assert len(nodes) == nnodes

    for node in nodes:
        fname = insert_node_filename("grid.nc", node.node, nnodes)
        with netCDF4.Dataset(fortran_out_dir / fname) as ds:
            for dim_kind in ("xi_rho", "xi_u", "eta_rho", "eta_v"):
                expected = len(ds.dimensions[dim_kind])
                got = node_dim_extent(dim_kind, node)
                assert got == expected, (np_xi, np_eta, node.node, dim_kind, expected, got)

            partition_attr = list(ds.getncattr("partition"))
            assert partition_attr == [node.node, nnodes, node.xi_start, node.eta_start]

"""End-to-end: run the real compiled `partit` and `roms-part` on the same
synthetic file and diff every resulting node file bit-for-bit -- dimension
sizes, the `partition` global attribute, all other global attributes, and
every variable's data. This is the actual correctness gate; the unit tests
in test_decomposition.py/test_naming.py/test_dims.py cover the pure-function
building blocks in isolation, but this is what proves the assembled tool
produces identical output to the Fortran original.
"""
from __future__ import annotations

import subprocess
import sys

import netCDF4
import numpy as np
import pytest

from roms_part.naming import insert_node_filename


def _run_roms_part(src_path, np_xi, np_eta, output_dir, n_workers=2):
    subprocess.run(
        [
            sys.executable,
            "-m",
            "roms_part.cli",
            str(np_xi),
            str(np_eta),
            str(src_path),
            "--output-dir",
            str(output_dir),
            "--n-workers",
            str(n_workers),
        ],
        check=True,
        capture_output=True,
        text=True,
    )


def _run_partit(partit_binary, src_path, np_xi, np_eta):
    # partit.F always writes into the CWD, stripping the source directory.
    subprocess.run(
        [partit_binary, str(np_xi), str(np_eta), src_path.name],
        cwd=src_path.parent,
        check=True,
        capture_output=True,
        text=True,
    )


def _assert_node_files_equal(fortran_dir, python_dir, basename, nnodes):
    for node in range(nnodes):
        fname = insert_node_filename(basename, node, nnodes)
        fds = netCDF4.Dataset(fortran_dir / fname)
        pds = netCDF4.Dataset(python_dir / fname)
        try:
            fdims = {n: len(d) for n, d in fds.dimensions.items()}
            pdims = {n: len(d) for n, d in pds.dimensions.items()}
            assert fdims == pdims, (node, "DIM MISMATCH", fdims, pdims)

            assert list(fds.getncattr("partition")) == list(pds.getncattr("partition")), (
                node,
                "PARTITION ATTR MISMATCH",
            )

            fattrs = set(fds.ncattrs()) - {"partition"}
            pattrs = set(pds.ncattrs()) - {"partition"}
            assert fattrs == pattrs, (node, "GLOBAL ATTR SET MISMATCH", fattrs ^ pattrs)

            fvars = set(fds.variables.keys())
            pvars = set(pds.variables.keys())
            assert fvars == pvars, (node, "VAR SET MISMATCH", fvars ^ pvars)

            for v in fvars:
                fv = np.ma.filled(fds.variables[v][:], np.nan) if np.issubdtype(
                    fds.variables[v].dtype, np.floating
                ) else np.ma.filled(fds.variables[v][:], 0)
                pv = np.ma.filled(pds.variables[v][:], np.nan) if np.issubdtype(
                    pds.variables[v].dtype, np.floating
                ) else np.ma.filled(pds.variables[v][:], 0)
                assert fv.shape == pv.shape, (node, v, "SHAPE MISMATCH", fv.shape, pv.shape)
                assert np.array_equal(fv, pv, equal_nan=np.issubdtype(fv.dtype, np.floating)), (
                    node,
                    v,
                    "DATA MISMATCH",
                )
        finally:
            fds.close()
            pds.close()


@pytest.mark.parametrize("np_xi,np_eta", [(2, 2), (3, 2)])
def test_basic_grid(partit_binary, tiny_grid_factory, tmp_path, np_xi, np_eta):
    grid_path = tiny_grid_factory(24, 20, name="grid.nc")
    py_out = tmp_path / "py_out"
    py_out.mkdir()

    _run_partit(partit_binary, grid_path, np_xi, np_eta)
    _run_roms_part(grid_path, np_xi, np_eta, py_out)

    _assert_node_files_equal(grid_path.parent, py_out, "grid.nc", np_xi * np_eta)


def test_obsolete_dims(partit_binary, tiny_grid_factory, tmp_path):
    grid_path = tiny_grid_factory(24, 20, name="grid.nc", with_obsolete_dims=True)
    py_out = tmp_path / "py_out"
    py_out.mkdir()

    _run_partit(partit_binary, grid_path, 2, 2)
    _run_roms_part(grid_path, 2, 2, py_out)

    _assert_node_files_equal(grid_path.parent, py_out, "grid.nc", 4)

    # obsolete dims must not survive into either tool's output
    with netCDF4.Dataset(py_out / insert_node_filename("grid.nc", 0, 4)) as ds:
        for obsolete in ("xi_psi", "xi_v", "eta_psi", "eta_u"):
            assert obsolete not in ds.dimensions


def test_boundary_vars(partit_binary, tiny_grid_factory, tmp_path):
    grid_path = tiny_grid_factory(24, 20, name="grid.nc", with_boundary_vars=True)
    py_out = tmp_path / "py_out"
    py_out.mkdir()

    _run_partit(partit_binary, grid_path, 2, 2)
    _run_roms_part(grid_path, 2, 2, py_out)

    _assert_node_files_equal(grid_path.parent, py_out, "grid.nc", 4)

    # node 0 is SW corner (western+southern edge): should have _west and
    # _south vars but not _east/_north; node 3 is NE corner: the reverse.
    with netCDF4.Dataset(py_out / insert_node_filename("grid.nc", 0, 4)) as ds:
        assert "temp_west" in ds.variables and "temp_south" in ds.variables
        assert "temp_east" not in ds.variables and "temp_north" not in ds.variables
    with netCDF4.Dataset(py_out / insert_node_filename("grid.nc", 3, 4)) as ds:
        assert "temp_east" in ds.variables and "temp_north" in ds.variables
        assert "temp_west" not in ds.variables and "temp_south" not in ds.variables


def test_extra_dtypes(partit_binary, tiny_grid_factory, tmp_path):
    grid_path = tiny_grid_factory(24, 20, name="grid.nc", extra_dtypes=True)
    py_out = tmp_path / "py_out"
    py_out.mkdir()

    _run_partit(partit_binary, grid_path, 2, 2)
    _run_roms_part(grid_path, 2, 2, py_out)

    _assert_node_files_equal(grid_path.parent, py_out, "grid.nc", 4)


def test_time_records(partit_binary, tiny_grid_factory, tmp_path):
    grid_path = tiny_grid_factory(24, 20, name="grid.nc", n_records=6)
    py_out = tmp_path / "py_out"
    py_out.mkdir()

    _run_partit(partit_binary, grid_path, 2, 2)
    _run_roms_part(grid_path, 2, 2, py_out)

    _assert_node_files_equal(grid_path.parent, py_out, "grid.nc", 4)


def test_time_records_with_batching(partit_binary, tiny_grid_factory, tmp_path):
    grid_path = tiny_grid_factory(24, 20, name="grid.nc", n_records=6)
    py_out = tmp_path / "py_out"
    py_out.mkdir()

    _run_partit(partit_binary, grid_path, 2, 2)
    subprocess.run(
        [
            sys.executable,
            "-m",
            "roms_part.cli",
            "2",
            "2",
            str(grid_path),
            "--output-dir",
            str(py_out),
            "--n-workers",
            "2",
            "--time-batch-size",
            "2",
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    _assert_node_files_equal(grid_path.parent, py_out, "grid.nc", 4)

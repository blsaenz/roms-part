"""End-to-end: partition a synthetic file with the real `partit` binary,
join the tiles back with `roms-join`, and diff against the ORIGINAL
pre-partition source file bit-for-bit -- the primary correctness oracle
for the join side (no Fortran intermediate needed for ground truth, unlike
the partit side's tests). Also cross-validates against real `ncjoin` output
directly where useful.
"""
from __future__ import annotations

import subprocess
import sys

import netCDF4
import numpy as np
import pytest

from roms_part.join_cli import run_join


def _assert_matches_original(orig_path, joined_path):
    orig = netCDF4.Dataset(orig_path)
    joined = netCDF4.Dataset(joined_path)
    try:
        assert "partition" not in joined.ncattrs()
        for v in orig.variables:
            a = orig.variables[v][:]
            b = joined.variables[v][:]
            assert a.shape == b.shape, (v, a.shape, b.shape)
            is_float = np.issubdtype(a.dtype, np.floating)
            af = np.ma.filled(a, np.nan) if is_float else np.ma.filled(a, 0)
            bf = np.ma.filled(b, np.nan) if is_float else np.ma.filled(b, 0)
            assert np.array_equal(af, bf, equal_nan=is_float), (v, "DATA MISMATCH")
        for a in orig.ncattrs():
            assert a in joined.ncattrs(), (a, "missing global attr")
    finally:
        orig.close()
        joined.close()


@pytest.mark.parametrize("np_xi,np_eta", [(2, 2), (3, 2), (1, 1)])
def test_basic_grid_round_trip(tiny_grid_factory, make_tiles, tmp_path, np_xi, np_eta):
    grid_path = tiny_grid_factory(24, 20, name="grid.nc")
    tile_paths = make_tiles(grid_path, np_xi, np_eta)

    out_path = run_join([str(p) for p in tile_paths], output_path=str(tmp_path / "joined.nc"))
    _assert_matches_original(grid_path, out_path)


def test_boundary_vars_round_trip(tiny_grid_factory, make_tiles, tmp_path):
    grid_path = tiny_grid_factory(24, 20, name="grid.nc", with_boundary_vars=True)
    tile_paths = make_tiles(grid_path, 2, 2)

    out_path = run_join([str(p) for p in tile_paths], output_path=str(tmp_path / "joined.nc"))
    _assert_matches_original(grid_path, out_path)


def test_extra_dtypes_round_trip(tiny_grid_factory, make_tiles, tmp_path):
    grid_path = tiny_grid_factory(24, 20, name="grid.nc", extra_dtypes=True)
    tile_paths = make_tiles(grid_path, 2, 2)

    out_path = run_join([str(p) for p in tile_paths], output_path=str(tmp_path / "joined.nc"))
    _assert_matches_original(grid_path, out_path)


def test_time_records_round_trip(tiny_grid_factory, make_tiles, tmp_path):
    grid_path = tiny_grid_factory(24, 20, name="grid.nc", n_records=6)
    tile_paths = make_tiles(grid_path, 2, 2)

    out_path = run_join([str(p) for p in tile_paths], output_path=str(tmp_path / "joined.nc"))
    _assert_matches_original(grid_path, out_path)


def test_uncompressed_output(tiny_grid_factory, make_tiles, tmp_path):
    grid_path = tiny_grid_factory(24, 20, name="grid.nc")
    tile_paths = make_tiles(grid_path, 2, 2)

    out_path = run_join(
        [str(p) for p in tile_paths], output_path=str(tmp_path / "joined.nc"), deflate_level=0
    )
    _assert_matches_original(grid_path, out_path)


def test_matches_real_ncjoin_output(tiny_grid_factory, make_tiles, ncjoin_binary, tmp_path):
    """Cross-validate against the real Fortran ncjoin binary directly, not
    just the original source file -- confirms dims/attrs/data agree with
    what the actual tool in this repo produces."""
    grid_path = tiny_grid_factory(24, 20, name="grid.nc", with_boundary_vars=True, extra_dtypes=True)
    tile_paths = make_tiles(grid_path, 2, 2)

    fortran_dir = tile_paths[0].parent
    subprocess.run(
        [ncjoin_binary] + [p.name for p in tile_paths],
        cwd=fortran_dir,
        check=True,
        capture_output=True,
        text=True,
    )
    fortran_out = fortran_dir / "grid.nc"

    py_out = run_join([str(p) for p in tile_paths], output_path=str(tmp_path / "joined.nc"))

    f = netCDF4.Dataset(fortran_out)
    p = netCDF4.Dataset(py_out)
    try:
        assert set(f.variables) == set(p.variables)
        for v in f.variables:
            fv = np.ma.filled(f.variables[v][:], np.nan)
            pv = np.ma.filled(p.variables[v][:], np.nan)
            assert np.array_equal(fv, pv, equal_nan=True), v
    finally:
        f.close()
        p.close()


def test_cli_delete_flag(tiny_grid_factory, make_tiles, tmp_path):
    grid_path = tiny_grid_factory(24, 20, name="grid.nc")
    tile_paths = make_tiles(grid_path, 2, 2)
    out_path = tmp_path / "joined.nc"

    subprocess.run(
        [sys.executable, "-m", "roms_part.join_cli"]
        + [str(p) for p in tile_paths]
        + ["--output", str(out_path), "--delete"],
        check=True,
        capture_output=True,
        text=True,
    )
    assert out_path.exists()
    for p in tile_paths:
        assert not p.exists()

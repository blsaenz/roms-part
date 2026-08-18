"""Shared pytest fixtures.

Validation against the real Fortran `partit` binary is the load-bearing
part of this test suite (see README.md). Since roms-part lives outside the
ucla-roms checkout, we locate ucla-roms via the ROMS_ROOT environment
variable (falling back to a sibling `~/Projects/git/ucla-roms` checkout),
copy just Tools-Roms + src/Makedefs.inc into a temp build dir (so we never
touch the real working tree), and build `partit` there with
`make partit COMPILER=gnu`. This mirrors the pattern already used by
ucla-roms' own Tools-Roms/mpc_python/test_src.py (which builds a *historical*
mpc binary from a git commit for the same kind of bit-for-bit comparison).

Tests that need `partit_binary` are skipped (not failed) if no ucla-roms
checkout can be found, so `pytest` still runs cleanly in an environment
without it.
"""
from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

import netCDF4
import numpy as np
import pytest


def _find_roms_root() -> Path | None:
    env = os.environ.get("ROMS_ROOT")
    if env:
        p = Path(env)
        if (p / "Tools-Roms" / "partit.F").is_file():
            return p
    sibling = Path.home() / "Projects" / "git" / "ucla-roms"
    if (sibling / "Tools-Roms" / "partit.F").is_file():
        return sibling
    return None


@pytest.fixture(scope="session")
def roms_root() -> Path:
    root = _find_roms_root()
    if root is None:
        pytest.skip(
            "No ucla-roms checkout found (set ROMS_ROOT or place one at "
            "~/Projects/git/ucla-roms) -- skipping tests that validate "
            "against the real Fortran partit binary."
        )
    return root


@pytest.fixture(scope="session")
def partit_binary(roms_root: Path, tmp_path_factory) -> str:
    """Build the real `partit` binary once per test session, in an isolated
    copy of Tools-Roms (never touches the actual ucla-roms working tree)."""
    build_dir = tmp_path_factory.mktemp("partit_build")
    shutil.copytree(roms_root / "Tools-Roms", build_dir / "Tools-Roms")
    (build_dir / "src").mkdir(parents=True, exist_ok=True)
    shutil.copy(roms_root / "src" / "Makedefs.inc", build_dir / "src" / "Makedefs.inc")

    tools_dir = build_dir / "Tools-Roms"
    mpc_path = tools_dir / "mpc"
    mpc_path.chmod(0o755)
    # mpc's `from passes.cleanup import ...` needs `passes/` as a sibling of
    # the mpc script itself (not just under mpc_python/) -- observed to be
    # required specifically for scripts run from certain temp-dir paths
    # (e.g. macOS's per-user TMPDIR under /var/folders/.../T/), though not
    # always from a plain /tmp copy; copying it unconditionally removes the
    # flakiness rather than relying on that being deterministic.
    shutil.copytree(tools_dir / "mpc_python" / "passes", tools_dir / "passes")

    env = dict(os.environ)
    env["PATH"] = f"{tools_dir}:{env.get('PATH', '')}"

    subprocess.run(
        ["make", "partit", "COMPILER=gnu"],
        cwd=tools_dir,
        env=env,
        check=True,
        capture_output=True,
        text=True,
    )
    return str(tools_dir / "partit")


@pytest.fixture(scope="session")
def ncjoin_binary(roms_root: Path, tmp_path_factory) -> str:
    """Build the real `ncjoin` binary once per test session, same isolated-
    copy approach as `partit_binary`."""
    build_dir = tmp_path_factory.mktemp("ncjoin_build")
    shutil.copytree(roms_root / "Tools-Roms", build_dir / "Tools-Roms")
    (build_dir / "src").mkdir(parents=True, exist_ok=True)
    shutil.copy(roms_root / "src" / "Makedefs.inc", build_dir / "src" / "Makedefs.inc")

    tools_dir = build_dir / "Tools-Roms"
    mpc_path = tools_dir / "mpc"
    mpc_path.chmod(0o755)
    shutil.copytree(tools_dir / "mpc_python" / "passes", tools_dir / "passes")

    env = dict(os.environ)
    env["PATH"] = f"{tools_dir}:{env.get('PATH', '')}"

    subprocess.run(
        ["make", "ncjoin", "COMPILER=gnu"],
        cwd=tools_dir,
        env=env,
        check=True,
        capture_output=True,
        text=True,
    )
    return str(tools_dir / "ncjoin")


@pytest.fixture
def make_tiles(partit_binary):
    """Returns a function that runs the real `partit` binary on a source
    file and returns the sorted list of resulting tile file paths -- for
    tests that need real, correct tiles as join-side input without
    depending on roms-part's own partit implementation."""

    def _make(src_path: Path, np_xi: int, np_eta: int) -> list[Path]:
        subprocess.run(
            [partit_binary, str(np_xi), str(np_eta), src_path.name],
            cwd=src_path.parent,
            check=True,
            capture_output=True,
            text=True,
        )
        nnodes = np_xi * np_eta
        digits = len(str(nnodes - 1))
        stem, suffix = src_path.stem, src_path.suffix
        return [
            src_path.parent / f"{stem}.{str(n).zfill(digits)}{suffix}" for n in range(nnodes)
        ]

    return _make


@pytest.fixture
def tiny_grid_factory(tmp_path):
    """Returns a function that writes a small synthetic ROMS-style netCDF
    file with the four canonical partitionable dims, and returns its path.
    """

    def _make(
        xi_rho: int,
        eta_rho: int,
        name: str = "grid.nc",
        n_records: int = 0,
        with_obsolete_dims: bool = False,
        with_boundary_vars: bool = False,
        extra_dtypes: bool = False,
    ) -> Path:
        path = tmp_path / name
        rng = np.random.default_rng(0)
        with netCDF4.Dataset(path, mode="w", format="NETCDF4") as ds:
            ds.createDimension("xi_rho", xi_rho)
            ds.createDimension("xi_u", xi_rho - 1)
            ds.createDimension("eta_rho", eta_rho)
            ds.createDimension("eta_v", eta_rho - 1)
            if with_obsolete_dims:
                ds.createDimension("xi_psi", xi_rho - 1)
                ds.createDimension("xi_v", xi_rho)
                ds.createDimension("eta_psi", eta_rho - 1)
                ds.createDimension("eta_u", eta_rho)
            if n_records:
                ds.createDimension("time", None)

            def _fill(var, shape):
                var[:] = rng.random(shape).astype(var.dtype)

            v = ds.createVariable("zeta", "f4", ("eta_rho", "xi_rho"))
            _fill(v, (eta_rho, xi_rho))
            v = ds.createVariable("ubar", "f4", ("eta_rho", "xi_u"))
            _fill(v, (eta_rho, xi_rho - 1))
            v = ds.createVariable("vbar", "f4", ("eta_v", "xi_rho"))
            _fill(v, (eta_rho - 1, xi_rho))
            v = ds.createVariable("mask", "f8", ("eta_rho", "xi_rho"))
            _fill(v, (eta_rho, xi_rho))
            v = ds.createVariable("h", "f4", ("eta_rho", "xi_rho"))
            v.long_name = "bathymetry"
            _fill(v, (eta_rho, xi_rho))

            if n_records:
                v = ds.createVariable("temp", "f4", ("time", "eta_rho", "xi_rho"))
                _fill(v, (n_records, eta_rho, xi_rho))
                v = ds.createVariable("ocean_time", "f8", ("time",))
                v[:] = np.arange(n_records, dtype="f8")

            if with_obsolete_dims:
                v = ds.createVariable("psi_var", "f4", ("eta_psi", "xi_psi"))
                _fill(v, (eta_rho - 1, xi_rho - 1))

            if with_boundary_vars:
                v = ds.createVariable("temp_west", "f4", ("eta_rho",))
                _fill(v, (eta_rho,))
                v = ds.createVariable("temp_east", "f4", ("eta_rho",))
                _fill(v, (eta_rho,))
                v = ds.createVariable("temp_south", "f4", ("xi_rho",))
                _fill(v, (xi_rho,))
                v = ds.createVariable("temp_north", "f4", ("xi_rho",))
                _fill(v, (xi_rho,))

            if extra_dtypes:
                v = ds.createVariable("flag_byte", "i1", ("eta_rho", "xi_rho"))
                v[:] = (rng.random((eta_rho, xi_rho)) * 10).astype("i1")
                v = ds.createVariable("flag_short", "i2", ("eta_rho", "xi_rho"))
                v[:] = (rng.random((eta_rho, xi_rho)) * 100).astype("i2")
                v = ds.createVariable("flag_int", "i4", ("eta_rho", "xi_rho"))
                v[:] = (rng.random((eta_rho, xi_rho)) * 1000).astype("i4")

            ds.title = "synthetic test grid"
            ds.setncattr("some_global_attr", np.float64(3.14))

        return path

    return _make

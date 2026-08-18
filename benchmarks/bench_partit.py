#!/usr/bin/env python3
"""Benchmark roms-part against the real compiled `partit` binary on a real
file, sweeping --n-workers, and (optionally) verify bit-for-bit equivalence
between the two tools' output.

Usage:
    python benchmarks/bench_partit.py \
        --partit /path/to/compiled/partit \
        --file /path/to/big_file.nc \
        --np-xi 16 --np-eta 32 \
        --workers 1 2 4 8 \
        --verify

Not collected by pytest -- this is a manual/CI-triggered tool, not a unit
test (it needs a real compiled `partit` binary and a real, possibly large,
input file, and may take a long time to run at high partition counts).
"""
from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import netCDF4
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from roms_part.naming import insert_node_filename  # noqa: E402


def run_fortran(partit_bin: str, src: Path, np_xi: int, np_eta: int, out_dir: Path) -> float:
    # partit.F always writes into the CWD, stripping the source directory,
    # so it must be run from inside out_dir; a symlink lets us do that
    # without copying the (possibly large) source file.
    out_dir.mkdir(parents=True, exist_ok=True)
    link = out_dir / src.name
    if not link.exists():
        link.symlink_to(src)

    t0 = time.time()
    subprocess.run(
        [partit_bin, str(np_xi), str(np_eta), src.name],
        cwd=out_dir,
        check=True,
        capture_output=True,
        text=True,
    )
    return time.time() - t0


def run_python(src: Path, np_xi: int, np_eta: int, out_dir: Path, n_workers: int) -> float:
    out_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    subprocess.run(
        [
            sys.executable,
            "-m",
            "roms_part.cli",
            str(np_xi),
            str(np_eta),
            str(src),
            "--output-dir",
            str(out_dir),
            "--n-workers",
            str(n_workers),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    return time.time() - t0


def verify_equivalent(fortran_dir: Path, python_dir: Path, basename: str, nnodes: int) -> bool:
    ok = True
    for node in range(nnodes):
        fname = insert_node_filename(basename, node, nnodes)
        fds = netCDF4.Dataset(fortran_dir / fname)
        pds = netCDF4.Dataset(python_dir / fname)
        try:
            if list(fds.getncattr("partition")) != list(pds.getncattr("partition")):
                print(f"  MISMATCH node {node}: partition attribute differs")
                ok = False
                continue
            for v in fds.variables:
                fv = fds.variables[v][:]
                pv = pds.variables[v][:]
                if not np.array_equal(
                    np.ma.filled(fv, np.nan), np.ma.filled(pv, np.nan), equal_nan=True
                ):
                    print(f"  MISMATCH node {node} var {v}: data differs")
                    ok = False
        finally:
            fds.close()
            pds.close()
    return ok


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--partit", required=True, help="path to compiled partit binary")
    parser.add_argument("--file", required=True, help="path to a real ROMS netCDF input file")
    parser.add_argument("--np-xi", type=int, required=True)
    parser.add_argument("--np-eta", type=int, required=True)
    parser.add_argument("--workers", type=int, nargs="+", default=[1, 2, 4, 8])
    parser.add_argument("--verify", action="store_true")
    parser.add_argument("--keep-tmp", action="store_true")
    args = parser.parse_args()

    src = Path(args.file).resolve()
    nnodes = args.np_xi * args.np_eta
    tmp = Path(tempfile.mkdtemp(prefix="bench_partit_"))
    print(f"working dir: {tmp}")

    fortran_dir = tmp / "fortran"
    fortran_time = run_fortran(args.partit, src, args.np_xi, args.np_eta, fortran_dir)
    print(f"fortran partit ({args.np_xi}x{args.np_eta}={nnodes} nodes): {fortran_time:.2f} sec")

    rows = [("fortran", 1, fortran_time, 1.0)]
    for w in args.workers:
        py_dir = tmp / f"python_w{w}"
        py_time = run_python(src, args.np_xi, args.np_eta, py_dir, w)
        speedup = fortran_time / py_time
        rows.append(("roms-part", w, py_time, speedup))
        print(f"roms-part (n_workers={w}): {py_time:.2f} sec  ({speedup:.2f}x vs fortran)")

        if args.verify:
            equal = verify_equivalent(fortran_dir, py_dir, src.name, nnodes)
            print(f"  bit-for-bit equivalence vs fortran: {'PASS' if equal else 'FAIL'}")

    print()
    print(f"{'tool':<12}{'workers':>9}{'seconds':>10}{'speedup':>10}")
    for tool, w, t, s in rows:
        print(f"{tool:<12}{w:>9}{t:>10.2f}{s:>10.2f}")

    if not args.keep_tmp:
        shutil.rmtree(tmp, ignore_errors=True)
    else:
        print(f"kept working dir: {tmp}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

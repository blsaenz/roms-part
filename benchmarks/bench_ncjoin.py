#!/usr/bin/env python3
"""Benchmark roms-join against the real compiled `ncjoin` binary, sweeping
--n-workers, and verifying bit-for-bit equivalence between the two tools'
output (and, when --original is given, against the pre-partition source).

Usage:
    python benchmarks/bench_ncjoin.py \
        --ncjoin /path/to/compiled/ncjoin \
        --tiles '/path/to/tiles/file.*.nc' \
        --workers 1 2 4 8 \
        --original /path/to/original_source.nc \
        --verify

Not collected by pytest -- needs a real compiled `ncjoin` binary and a real
tile set, and may take a long time for large tile counts/files.
"""
from __future__ import annotations

import argparse
import glob
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import netCDF4
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from roms_part.join_cli import run_join  # noqa: E402


def run_fortran_ncjoin(ncjoin_bin: str, tile_paths: list[Path], work_dir: Path) -> tuple[float, Path]:
    work_dir.mkdir(parents=True, exist_ok=True)
    links = []
    for p in tile_paths:
        link = work_dir / p.name
        if not link.exists():
            link.symlink_to(p)
        links.append(link)

    t0 = time.time()
    subprocess.run(
        [ncjoin_bin] + [p.name for p in links],
        cwd=work_dir,
        check=True,
        capture_output=True,
        text=True,
    )
    dt = time.time() - t0

    # ncjoin derives the joined filename by stripping the node digit segment;
    # find whatever new .nc file appeared that isn't one of the tiles.
    tile_names = {p.name for p in links}
    candidates = [
        Path(f) for f in glob.glob(str(work_dir / "*.nc")) if Path(f).name not in tile_names
    ]
    assert len(candidates) == 1, f"expected exactly one joined output, got {candidates}"
    return dt, candidates[0]


def run_python_join(tile_paths: list[Path], out_dir: Path, n_workers: int) -> tuple[float, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "joined.nc"
    t0 = time.time()
    run_join([str(p) for p in tile_paths], output_path=str(out_path), n_workers=n_workers)
    dt = time.time() - t0
    return dt, out_path


def verify_equivalent(path_a: Path, path_b: Path, label_a: str, label_b: str) -> bool:
    a = netCDF4.Dataset(path_a)
    b = netCDF4.Dataset(path_b)
    ok = True
    try:
        if set(a.variables) != set(b.variables):
            print(f"  variable set mismatch: {label_a} vs {label_b}")
            return False
        for v in a.variables:
            av = a.variables[v][:]
            bv = b.variables[v][:]
            is_float = np.issubdtype(av.dtype, np.floating)
            af = np.ma.filled(av, np.nan) if is_float else np.ma.filled(av, 0)
            bf = np.ma.filled(bv, np.nan) if is_float else np.ma.filled(bv, 0)
            if not np.array_equal(af, bf, equal_nan=is_float):
                print(f"  MISMATCH var {v}: {label_a} vs {label_b}")
                ok = False
    finally:
        a.close()
        b.close()
    return ok


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ncjoin", required=True, help="path to compiled ncjoin binary")
    parser.add_argument("--tiles", required=True, help="glob pattern matching all tile files")
    parser.add_argument("--workers", type=int, nargs="+", default=[1, 2, 4, 8])
    parser.add_argument("--original", default=None, help="pre-partition source file, for a correctness oracle")
    parser.add_argument("--verify", action="store_true")
    parser.add_argument("--keep-tmp", action="store_true")
    args = parser.parse_args()

    tile_paths = sorted(Path(p) for p in glob.glob(args.tiles))
    if not tile_paths:
        print(f"no files matched glob: {args.tiles}", file=sys.stderr)
        return 1
    print(f"found {len(tile_paths)} tile files")

    tmp = Path(tempfile.mkdtemp(prefix="bench_ncjoin_"))
    print(f"working dir: {tmp}")

    fortran_dir = tmp / "fortran"
    fortran_time, fortran_out = run_fortran_ncjoin(args.ncjoin, tile_paths, fortran_dir)
    print(f"fortran ncjoin ({len(tile_paths)} tiles): {fortran_time:.2f} sec -> {fortran_out.name}")

    rows = [("ncjoin", 1, fortran_time, 1.0)]
    for w in args.workers:
        py_dir = tmp / f"python_w{w}"
        py_time, py_out = run_python_join(tile_paths, py_dir, w)
        speedup = fortran_time / py_time
        rows.append(("roms-join", w, py_time, speedup))
        print(f"roms-join (n_workers={w}): {py_time:.2f} sec ({speedup:.2f}x vs ncjoin)")

        if args.verify:
            equal = verify_equivalent(fortran_out, py_out, "ncjoin", f"roms-join(w={w})")
            print(f"  bit-for-bit equivalence vs ncjoin: {'PASS' if equal else 'FAIL'}")
            if args.original:
                equal_orig = verify_equivalent(Path(args.original), py_out, "original", f"roms-join(w={w})")
                print(f"  bit-for-bit equivalence vs original source: {'PASS' if equal_orig else 'FAIL'}")

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

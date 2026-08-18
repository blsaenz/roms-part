#!/usr/bin/env python3
"""Command-line entry point for roms-join: a fast, parallel Python
replacement for UCLA-ROMS's Tools-Roms/ncjoin.F, joining partit-produced
tiles back into one whole-domain, compressed netCDF file.

Usage:  roms-join file1.NNN.nc file2.NNN.nc ... [--output out.nc]
"""
from __future__ import annotations

import argparse
import multiprocessing
import os
import shutil
import sys
import tempfile
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

from .join_job import build_join_job, estimate_peak_memory_bytes
from .join_schema import extract_node_and_root
from .join_worker import _init_worker, reconstruct_and_write_shard, stitch
from .mem_util import report_memory_estimate


def _default_output_path(tile_paths: list[str]) -> str:
    first = Path(tile_paths[0])
    # nnodes isn't known yet without reading the partition attribute; reuse
    # build_join_job's tile discovery instead of duplicating that logic --
    # callers with a known nnodes can call extract_node_and_root directly.
    import netCDF4

    with netCDF4.Dataset(first, mode="r") as ds:
        nnodes = int(ds.getncattr("partition")[1])
    root, _ = extract_node_and_root(first.name, nnodes)
    return str(first.parent / root)


def run_join(
    tile_paths: list[str],
    output_path: str | None = None,
    n_workers: int | None = None,
    deflate_level: int = 1,
    shuffle: bool = True,
    keep_shards: bool = False,
) -> str:
    if output_path is None:
        output_path = _default_output_path(tile_paths)

    shard_dir = tempfile.mkdtemp(prefix="roms_join_shards_")
    try:
        job = build_join_job(tile_paths, output_path, shard_dir, deflate_level, shuffle, n_workers)
        report_memory_estimate(estimate_peak_memory_bytes(job), job.n_workers, "roms-join")

        ctx = multiprocessing.get_context("spawn")
        shard_results = {}
        with ProcessPoolExecutor(
            max_workers=job.n_workers, mp_context=ctx, initializer=_init_worker, initargs=(job,)
        ) as pool:
            futures = {
                pool.submit(reconstruct_and_write_shard, vp.name): vp.name
                for vp in job.var_plans
            }
            for fut in as_completed(futures):
                sr = fut.result()
                shard_results[sr.name] = sr

        stitch(job, shard_results)
    finally:
        if not keep_shards:
            shutil.rmtree(shard_dir, ignore_errors=True)

    return output_path


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="roms-join")
    parser.add_argument("files", nargs="+", help="all tile files for one complete partit set")
    parser.add_argument("--output", default=None, help="joined output path (default: derived from tile names)")
    parser.add_argument(
        "--n-workers",
        type=int,
        default=None,
        help="Worker-process count (default: min(cpu_count, 8), capped at the number of variables).",
    )
    parser.add_argument("--deflate-level", type=int, default=1)
    parser.add_argument("--shuffle", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("-d", "--delete", action="store_true", help="delete tile files after a verified-successful join")
    parser.add_argument("--keep-shards", action="store_true", help="keep the temporary per-variable shard files (debugging)")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args(argv)

    t0 = time.time()
    try:
        out_path = run_join(
            args.files, args.output, args.n_workers, args.deflate_level, args.shuffle, args.keep_shards
        )
    except Exception as e:
        print(f"### ERROR: join failed: {e}", file=sys.stderr)
        return 1
    dt = time.time() - t0

    size_mb = os.path.getsize(out_path) / 1e6
    print(f"joined {len(args.files)} tiles -> {out_path} ({size_mb:.1f} MB) in {dt:.2f} sec")

    if args.delete:
        for f in args.files:
            os.remove(f)
        if args.verbose:
            print(f"deleted {len(args.files)} tile files")

    return 0


if __name__ == "__main__":
    sys.exit(main())

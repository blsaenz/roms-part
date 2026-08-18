#!/usr/bin/env python3
"""Command-line entry point for roms-part: a fast, parallel Python
replacement for UCLA-ROMS's Tools-Roms/partit.F.

Usage:  roms-part NP_XI NP_ETA file1.nc [file2.nc ...]
"""
from __future__ import annotations

import argparse
import multiprocessing
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

from .job import build_job, estimate_peak_memory_bytes
from .mem_util import report_memory_estimate
from .schema import AlreadyPartitionedError
from .worker import NodeResult, _init_worker, write_node_file


def partition_one_file(
    src_path: str,
    np_xi: int,
    np_eta: int,
    output_dir: str,
    n_workers: int | None,
    deflate_level: int,
    shuffle: bool,
    time_batch_size: int | None = None,
    verbose: bool = False,
) -> tuple[list[NodeResult], int]:
    """Returns (per-node results, actual worker count used -- may differ
    from the requested n_workers, e.g. if there are fewer tiles than
    workers requested)."""
    job = build_job(
        src_path, np_xi, np_eta, output_dir, deflate_level, shuffle, time_batch_size, n_workers
    )
    report_memory_estimate(estimate_peak_memory_bytes(job), job.n_workers, "roms-part")

    results: list[NodeResult] = []
    ctx = multiprocessing.get_context("spawn")
    with ProcessPoolExecutor(
        max_workers=job.n_workers, mp_context=ctx, initializer=_init_worker, initargs=(job,)
    ) as pool:
        futures = {pool.submit(write_node_file, i): i for i in range(job.nnodes)}
        for fut in as_completed(futures):
            node_id = futures[fut]
            res = fut.result()
            results.append(res)
            if verbose:
                print(f"  node {res.node} -> {res.path} ({res.bytes_written} bytes, pid {res.pid})")
    results.sort(key=lambda r: r.node)
    return results, job.n_workers


def _dry_run(src_path: str, np_xi: int, np_eta: int, output_dir: str, n_workers: int | None) -> None:
    job = build_job(src_path, np_xi, np_eta, output_dir, n_workers=n_workers)
    print(f"{src_path}: {job.nnodes} nodes ({np_xi} x {np_eta})")
    for node, fname in zip(job.nodes, job.node_filenames):
        print(
            f"  node {node.node:4d}  xi=[{node.xi_start},{node.xi_start+node.xi_size}) "
            f"eta=[{node.eta_start},{node.eta_start+node.eta_size})  -> {fname}"
        )
    report_memory_estimate(estimate_peak_memory_bytes(job), job.n_workers, "roms-part")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="roms-part")
    parser.add_argument("np_xi", type=int)
    parser.add_argument("np_eta", type=int)
    parser.add_argument("files", nargs="+")
    parser.add_argument(
        "--n-workers",
        type=int,
        default=None,
        help="Worker-process count (default: min(cpu_count, 8), capped at the number of tiles).",
    )
    parser.add_argument("--deflate-level", type=int, default=1)
    parser.add_argument("--shuffle", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--output-dir", default=".")
    parser.add_argument("--time-batch-size", type=int, default=None)
    parser.add_argument(
        "--keep-going",
        action="store_true",
        help="Continue with remaining files/nodes after a per-file error instead of aborting.",
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args(argv)

    exit_code = 0
    for f in args.files:
        try:
            if args.dry_run:
                _dry_run(f, args.np_xi, args.np_eta, args.output_dir, args.n_workers)
                continue
            t0 = time.time()
            results, n_workers_used = partition_one_file(
                f,
                args.np_xi,
                args.np_eta,
                args.output_dir,
                args.n_workers,
                args.deflate_level,
                args.shuffle,
                args.time_batch_size,
                args.verbose,
            )
            dt = time.time() - t0
            total_bytes = sum(r.bytes_written for r in results)
            print(
                f"{f}: wrote {len(results)} node files, "
                f"{total_bytes / 1e6:.1f} MB, in {dt:.2f} sec "
                f"({n_workers_used} workers)"
            )
        except AlreadyPartitionedError as e:
            print(f"### WARNING: {e}", file=sys.stderr)
            continue
        except Exception as e:
            print(f"### ERROR processing '{f}': {e}", file=sys.stderr)
            exit_code = 1
            if not args.keep_going:
                return exit_code
    return exit_code


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python3
"""Example: partition a large ROMS file at 16x32 (512 tiles) while staying
under a fixed memory budget.

roms-part's per-worker memory footprint is normally small -- each worker
only ever holds one tile's slice of one variable at a time (see
roms_part.job.estimate_peak_memory_bytes) -- so --n-workers is usually
chosen based on available CPU cores, not memory. But for very large
domains, double-precision variables, or coarse tile counts (few, large
tiles), it's worth deriving --n-workers from an actual memory cap instead
of guessing. This script shows how, using roms-part's own
memory-estimation helper: it probes the source file's schema (cheap --
header metadata only, no tile data is read), finds the largest single-node
variable array, and picks the largest --n-workers that keeps
n_workers x largest_node_array_size under the requested cap.

Usage:
    python examples/partition_16x32_with_memory_cap.py SRC_FILE.nc \\
        [--output-dir DIR] [--memory-cap-gb 64] [--max-workers N]

Example:
    python examples/partition_16x32_with_memory_cap.py \\
        /data/roms_output.nc --output-dir /data/tiles --memory-cap-gb 64
"""
from __future__ import annotations

import argparse
import sys

from roms_part.cli import partition_one_file
from roms_part.job import DEFAULT_N_WORKERS, build_job, estimate_peak_memory_bytes
from roms_part.mem_util import format_bytes, total_system_memory_bytes

NP_XI = 16
NP_ETA = 32


def n_workers_for_memory_cap(
    src_path: str,
    np_xi: int,
    np_eta: int,
    output_dir: str,
    memory_cap_bytes: int,
    max_workers: int | None,
):
    """Probes the source file (header-only, no tile data read) and returns
    (n_workers, largest_single_node_variable_bytes, total_tiles)."""
    probe_job = build_job(src_path, np_xi, np_eta, output_dir, n_workers=1)
    max_node_bytes = estimate_peak_memory_bytes(probe_job)  # n_workers=1, so this IS the per-worker size

    if max_node_bytes == 0:
        return max_workers or DEFAULT_N_WORKERS, max_node_bytes, probe_job.nnodes

    affordable = max(1, memory_cap_bytes // max_node_bytes)
    n_workers = min(affordable, max_workers or DEFAULT_N_WORKERS, probe_job.nnodes)
    return n_workers, max_node_bytes, probe_job.nnodes


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("src_file", help="source ROMS netCDF file to partition")
    parser.add_argument("--np-xi", type=int, default=NP_XI)
    parser.add_argument("--np-eta", type=int, default=NP_ETA)
    parser.add_argument("--output-dir", default=".")
    parser.add_argument("--memory-cap-gb", type=float, default=64.0)
    parser.add_argument(
        "--max-workers",
        type=int,
        default=None,
        help="Upper bound regardless of memory, e.g. available CPU cores (default: min(cpu_count, 8))",
    )
    parser.add_argument("--deflate-level", type=int, default=1)
    args = parser.parse_args(argv)

    memory_cap_bytes = int(args.memory_cap_gb * 1024**3)

    n_workers, max_node_bytes, nnodes = n_workers_for_memory_cap(
        args.src_file, args.np_xi, args.np_eta, args.output_dir, memory_cap_bytes, args.max_workers
    )

    print(f"Partitioning {args.src_file} at {args.np_xi}x{args.np_eta} = {nnodes} tiles")
    print(f"Largest single-node variable array: {format_bytes(max_node_bytes)}")
    print(f"Memory cap: {args.memory_cap_gb:.0f} GB -> using --n-workers {n_workers}")

    total_mem = total_system_memory_bytes()
    if total_mem and memory_cap_bytes > total_mem:
        print(
            f"### WARNING: requested memory cap ({format_bytes(memory_cap_bytes)}) "
            f"exceeds this machine's total RAM ({format_bytes(total_mem)})"
        )

    results, n_workers_used = partition_one_file(
        args.src_file,
        args.np_xi,
        args.np_eta,
        args.output_dir,
        n_workers,
        args.deflate_level,
        True,  # shuffle
        None,  # time_batch_size
        True,  # verbose
    )
    total_bytes = sum(r.bytes_written for r in results)
    print(
        f"Wrote {len(results)} tile files, {format_bytes(total_bytes)} total, "
        f"using {n_workers_used} workers."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())

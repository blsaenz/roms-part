"""Per-node netCDF I/O. Runs inside worker processes.

Design notes (see README.md for the full rationale):
  - _init_worker opens the source file ONCE per worker PROCESS (not once
    per task) and stores it in a module-global. Workers are spawned via
    multiprocessing's 'spawn' context, so each worker process has its own
    independent address space and its own independently-loaded copy of
    libhdf5/libnetcdf -- there is no shared HDF5 state between workers,
    which is what actually sidesteps HDF5's thread-safety hazards (that
    hazard is specifically about multiple THREADS sharing one loaded
    library instance; separate processes never share one).
  - Each task (write_node_file) creates its output file, writes it
    completely, and closes it before returning -- at most `n_workers`
    output files are ever open at once, vs. the Fortran design of holding
    all `nnodes` open simultaneously for the whole run.
  - Every worker opens the SAME source file read-only; every worker writes
    to its OWN distinct output file. No two processes ever touch the same
    file, so ordinary concurrent-readers-of-one-read-only-file semantics
    apply (a well-supported, unremarkable case for netCDF/HDF5).
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional

import netCDF4
import numpy as np

from .decomposition import NodeInfo, node_dim_extent, node_read_range
from .dims import PARTITIONABLE_DIMS, remap_dim_name
from .job import PartitionJob, VarPlan

_SRC: Optional[netCDF4.Dataset] = None
_JOB: Optional[PartitionJob] = None

_EDGE_ATTR = {
    "west": "western_edge",
    "east": "eastern_edge",
    "south": "southern_edge",
    "north": "northern_edge",
}


@dataclass(frozen=True)
class NodeResult:
    node: int
    path: str
    bytes_written: int
    pid: int


def _init_worker(job: PartitionJob) -> None:
    global _SRC, _JOB
    _JOB = job
    _SRC = netCDF4.Dataset(job.src_path, mode="r")


def _node_dim_size(dim_name: str, node: NodeInfo, job: PartitionJob) -> int:
    if dim_name in PARTITIONABLE_DIMS:
        return node_dim_extent(dim_name, node)
    return job.schema.dim_sizes[dim_name]


def _var_included_for_node(vp: VarPlan, node: NodeInfo) -> bool:
    if vp.boundary is None:
        return True
    return getattr(node, _EDGE_ATTR[vp.boundary])


def _base_slices(vp: VarPlan, node: NodeInfo, job: PartitionJob):
    """Per-axis (read_slice, write_count) for the FULL variable (no record
    batching applied yet)."""
    read_slices = []
    write_counts = []
    for raw_dim, canon_dim in zip(vp.info.dim_names, vp.canonical_dims):
        if canon_dim in PARTITIONABLE_DIMS:
            start, count = node_read_range(canon_dim, node)
            read_slices.append(slice(start, start + count))
            write_counts.append(count)
        else:
            size = job.schema.dim_sizes[raw_dim]
            read_slices.append(slice(0, size))
            write_counts.append(size)
    return read_slices, write_counts


def _record_axis_index(vp: VarPlan, job: PartitionJob) -> Optional[int]:
    if not vp.has_record_dim or job.schema.unlimited_dim is None:
        return None
    try:
        return vp.canonical_dims.index(job.schema.unlimited_dim)
    except ValueError:
        return None


def _copy_variable(dst_var, src_var, vp: VarPlan, node: NodeInfo, job: PartitionJob) -> None:
    read_slices, write_counts = _base_slices(vp, node, job)
    rec_axis = _record_axis_index(vp, job)

    n_records = write_counts[rec_axis] if rec_axis is not None else None
    batch = job.time_batch_size
    if rec_axis is None or not batch or n_records <= batch:
        data = src_var[tuple(read_slices)]
        dst_var[...] = data
        return

    lo = 0
    while lo < n_records:
        hi = min(lo + batch, n_records)
        batch_slices = list(read_slices)
        batch_slices[rec_axis] = slice(lo, hi)
        out_slices = [slice(0, c) for c in write_counts]
        out_slices[rec_axis] = slice(lo, hi)
        dst_var[tuple(out_slices)] = src_var[tuple(batch_slices)]
        lo = hi


def write_node_file(node_id: int) -> NodeResult:
    job = _JOB
    src = _SRC
    node = job.nodes[node_id]
    out_path = os.path.join(job.output_dir, job.node_filenames[node_id])

    with netCDF4.Dataset(out_path, mode="w", format="NETCDF4") as dst:
        # --- dimensions ---
        created_dims: set[str] = set()
        for raw_dim, size in job.schema.dim_sizes.items():
            canon = remap_dim_name(raw_dim)
            if canon != raw_dim:
                continue  # obsolete/redundant dim: suppressed in output
            if raw_dim == job.schema.unlimited_dim:
                if size == 0:
                    continue  # zero-size unlimited dim: suppressed
                dst.createDimension(raw_dim, None)
            else:
                out_size = _node_dim_size(raw_dim, node, job)
                if out_size == 0:
                    continue  # zero-size dim: suppressed
                dst.createDimension(raw_dim, out_size)
            created_dims.add(raw_dim)

        # --- global attributes + partition attribute ---
        for k, v in job.schema.global_attrs.items():
            dst.setncattr(k, v)
        dst.setncattr(
            "partition",
            np.array([node.node, job.nnodes, node.xi_start, node.eta_start], dtype=np.int32),
        )

        # --- variables ---
        for vp in job.var_plans:
            if not _var_included_for_node(vp, node):
                continue
            if any(d not in created_dims for d in vp.canonical_dims):
                continue  # references a suppressed (zero-size) dimension
            kwargs = dict(
                zlib=job.deflate_level > 0,
                shuffle=job.shuffle,
            )
            if job.deflate_level > 0:
                kwargs["complevel"] = job.deflate_level
            if vp.info.fill_value is not None:
                kwargs["fill_value"] = vp.info.fill_value
            # Use canonical dim names: an obsolete dim (e.g. xi_psi) was never
            # created in the output file -- the variable is redefined onto its
            # replacement (xi_u) instead, per partit.F's dimid remapping.
            var = dst.createVariable(vp.info.name, vp.info.dtype, vp.canonical_dims, **kwargs)
            src_var = src.variables[vp.info.name]
            for a in src_var.ncattrs():
                if a == "_FillValue":
                    continue
                var.setncattr(a, src_var.getncattr(a))

            _copy_variable(var, src_var, vp, node, job)

    return NodeResult(
        node=node_id,
        path=out_path,
        bytes_written=os.path.getsize(out_path),
        pid=os.getpid(),
    )

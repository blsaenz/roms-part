"""Combines schema.py + decomposition.py + naming.py + dims.py into a single
PartitionJob describing everything a worker needs to write one node's file,
without touching the filesystem beyond the initial schema read.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np

from .decomposition import NodeInfo, mpi_setup, node_dim_extent
from .dims import (
    DimAxis,
    PARTITIONABLE_DIMS,
    boundary_edge,
    classify_variable,
    remap_dim_name,
    resolve_partitionable_dims,
)
from .naming import insert_node_filename
from .schema import FileSchema, VarInfo, inspect_source

DEFAULT_N_WORKERS = min(os.cpu_count() or 1, 8)


@dataclass(frozen=True)
class VarPlan:
    info: VarInfo
    canonical_dims: tuple[str, ...]  # dim names after obsolete-dim remap
    part_type: DimAxis
    has_record_dim: bool
    boundary: Optional[str]  # 'west'/'east'/'south'/'north'/None
    max_node_bytes: int  # largest single-node array size for this variable, across all nodes


@dataclass(frozen=True)
class PartitionJob:
    src_path: str
    output_dir: str
    nnodes: int
    np_xi: int
    np_eta: int
    nodes: list[NodeInfo]
    node_filenames: list[str]
    schema: FileSchema
    partitionable: dict[str, int]  # resolved xi_rho/xi_u/eta_rho/eta_v sizes
    var_plans: list[VarPlan]
    deflate_level: int
    shuffle: bool
    n_workers: int
    time_batch_size: Optional[int] = None  # None = copy the record dim whole


def _node_variable_bytes(dim_names: tuple[str, ...], itemsize: int, node: NodeInfo, schema: FileSchema) -> int:
    size = itemsize
    for d in dim_names:
        if d in PARTITIONABLE_DIMS:
            size *= node_dim_extent(d, node)
        else:
            size *= max(schema.dim_sizes[d], 1)
    return size


def build_job(
    src_path: str,
    np_xi: int,
    np_eta: int,
    output_dir: str,
    deflate_level: int = 1,
    shuffle: bool = True,
    time_batch_size: Optional[int] = None,
    n_workers: Optional[int] = None,
) -> PartitionJob:
    schema = inspect_source(src_path)
    partitionable = resolve_partitionable_dims(schema.dim_sizes)
    nodes = mpi_setup(np_xi, np_eta, partitionable["xi_rho"], partitionable["eta_rho"])
    nnodes = np_xi * np_eta

    src_basename = Path(src_path).name
    node_filenames = [insert_node_filename(src_basename, n.node, nnodes) for n in nodes]

    var_plans = []
    for v in schema.variables:
        canonical_dims = tuple(remap_dim_name(d) for d in v.dim_names)
        part_type, has_record = classify_variable(canonical_dims, schema.unlimited_dim)
        itemsize = np.dtype(v.dtype).itemsize
        max_node_bytes = max(
            _node_variable_bytes(canonical_dims, itemsize, node, schema) for node in nodes
        )
        var_plans.append(
            VarPlan(
                info=v,
                canonical_dims=canonical_dims,
                part_type=part_type,
                has_record_dim=has_record,
                boundary=boundary_edge(v.name),
                max_node_bytes=max_node_bytes,
            )
        )

    resolved_n_workers = max(1, min(n_workers or DEFAULT_N_WORKERS, nnodes))

    return PartitionJob(
        src_path=src_path,
        output_dir=output_dir,
        nnodes=nnodes,
        np_xi=np_xi,
        np_eta=np_eta,
        nodes=nodes,
        node_filenames=node_filenames,
        schema=schema,
        partitionable=partitionable,
        var_plans=var_plans,
        deflate_level=deflate_level,
        shuffle=shuffle,
        n_workers=resolved_n_workers,
        time_batch_size=time_batch_size,
    )


def estimate_peak_memory_bytes(job: PartitionJob) -> int:
    """Rough peak-RSS estimate: each of job.n_workers concurrent worker
    processes holds at most one variable's single-node array in memory at
    a time (see worker.py), so peak memory is bounded by
    n_workers x (largest single-node variable size) -- an upper bound, not
    an exact figure (workers may in practice be handling smaller
    variables, or the same node's smaller variables, simultaneously).
    """
    if not job.var_plans:
        return 0
    return job.n_workers * max(vp.max_node_bytes for vp in job.var_plans)

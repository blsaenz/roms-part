"""Combines join_schema.py + join_decomposition.py + dims.py + schema.py into
a single JoinJob describing everything needed to reconstruct and write the
whole-domain joined file from a validated tile set.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional

import netCDF4
import numpy as np

from .dims import DimAxis, PARTITIONABLE_DIMS, boundary_edge, classify_variable
from .join_decomposition import GlobalExtent, TileInfo, reconstruct_edges, reconstruct_global_extent
from .join_schema import TileFile, discover_tile_set
from .schema import VarInfo, inspect_source

DEFAULT_N_WORKERS = min(os.cpu_count() or 1, 8)

_EDGE_ATTR = {
    "west": "western_edge",
    "east": "eastern_edge",
    "south": "southern_edge",
    "north": "northern_edge",
}


@dataclass(frozen=True)
class JoinVarPlan:
    name: str
    dim_names: tuple[str, ...]
    dtype: object
    fill_value: object
    part_type: DimAxis
    has_record_dim: bool
    boundary: Optional[str]
    contributing_nodes: tuple[int, ...]
    is_multidim: bool  # eligible for compression, matches ncjoin.F's vdims(i)>2 rule
    global_bytes: int  # size of the FULL reconstructed array -- what one worker holds in memory


@dataclass(frozen=True)
class JoinJob:
    tile_files: list[TileFile]  # sorted by node, node == index
    tile_infos: dict[int, TileInfo]
    extent: GlobalExtent
    raw_dim_sizes: dict[str, int]  # from tile 0; partitionable entries are LOCAL, not global
    unlimited_dim: Optional[str]
    global_attrs: dict[str, object]
    var_plans: list[JoinVarPlan]
    output_path: str
    shard_dir: str
    deflate_level: int
    shuffle: bool
    n_workers: int


def _local_partitionable_dims(path: str) -> dict[str, int]:
    with netCDF4.Dataset(path, mode="r") as ds:
        return {k: len(v) for k, v in ds.dimensions.items() if k in PARTITIONABLE_DIMS}


def _global_dim_size(dim_name: str, extent: GlobalExtent, raw_dim_sizes: dict[str, int]) -> int:
    if dim_name == "xi_rho":
        return extent.xi_rho
    if dim_name == "xi_u":
        return extent.xi_rho - 1
    if dim_name == "eta_rho":
        return extent.eta_rho
    if dim_name == "eta_v":
        return extent.eta_rho - 1
    return max(raw_dim_sizes[dim_name], 1)


def _global_variable_bytes(
    dim_names: tuple[str, ...], dtype, extent: GlobalExtent, raw_dim_sizes: dict[str, int]
) -> int:
    size = np.dtype(dtype).itemsize
    for d in dim_names:
        size *= _global_dim_size(d, extent, raw_dim_sizes)
    return size


def build_join_job(
    tile_paths: list[str],
    output_path: str,
    shard_dir: str,
    deflate_level: int = 1,
    shuffle: bool = True,
    n_workers: Optional[int] = None,
) -> JoinJob:
    tile_files = discover_tile_set(tile_paths)

    tile_infos: dict[int, TileInfo] = {}
    for tf in tile_files:
        tile_infos[tf.node] = TileInfo(
            node=tf.node,
            xi_start=tf.xi_start,
            eta_start=tf.eta_start,
            local_dims=_local_partitionable_dims(tf.path),
        )
    extent = reconstruct_global_extent(list(tile_infos.values()))
    edges_by_node = {n: reconstruct_edges(ti, extent) for n, ti in tile_infos.items()}

    schema0 = inspect_source(tile_files[0].path, allow_partition=True)
    global_attrs = {k: v for k, v in schema0.global_attrs.items() if k != "partition"}

    # Boundary-suffixed variables (_west/_east/_south/_north) only exist in
    # the tile(s) on the matching physical edge -- tile 0 (an arbitrary,
    # usually interior-or-corner tile) is NOT guaranteed to have every
    # variable that appears somewhere in the tile set, so the catalog must
    # be the UNION across all tiles, not just tile 0's schema.
    catalog: dict[str, VarInfo] = {v.name: v for v in schema0.variables}
    for tf in tile_files[1:]:
        schema_n = inspect_source(tf.path, allow_partition=True)
        for v in schema_n.variables:
            catalog.setdefault(v.name, v)

    var_plans = []
    for v in catalog.values():
        part_type, has_record = classify_variable(v.dim_names, schema0.unlimited_dim)
        boundary = boundary_edge(v.name)
        if boundary is None:
            contributing = tuple(sorted(tile_infos.keys()))
        else:
            attr = _EDGE_ATTR[boundary]
            contributing = tuple(
                n for n in sorted(tile_infos.keys()) if getattr(edges_by_node[n], attr)
            )
        var_plans.append(
            JoinVarPlan(
                name=v.name,
                dim_names=v.dim_names,
                dtype=v.dtype,
                fill_value=v.fill_value,
                part_type=part_type,
                has_record_dim=has_record,
                boundary=boundary,
                contributing_nodes=contributing,
                is_multidim=len(v.dim_names) > 2,
                global_bytes=_global_variable_bytes(v.dim_names, v.dtype, extent, schema0.dim_sizes),
            )
        )

    resolved_n_workers = max(1, min(n_workers or DEFAULT_N_WORKERS, len(var_plans)))

    return JoinJob(
        tile_files=tile_files,
        tile_infos=tile_infos,
        extent=extent,
        raw_dim_sizes=schema0.dim_sizes,
        unlimited_dim=schema0.unlimited_dim,
        global_attrs=global_attrs,
        var_plans=var_plans,
        output_path=output_path,
        shard_dir=shard_dir,
        deflate_level=deflate_level,
        shuffle=shuffle,
        n_workers=resolved_n_workers,
    )


def estimate_peak_memory_bytes(job: JoinJob) -> int:
    """Rough peak-RSS estimate: each of job.n_workers concurrent worker
    processes reconstructs and holds ONE variable's FULL whole-domain array
    in memory at a time (unlike roms-part, where each worker only ever
    holds one node's fraction of a variable) -- so peak memory is bounded
    by n_workers x (largest single variable's global size). An upper
    bound, not exact: workers may in practice be handling smaller
    variables simultaneously.
    """
    if not job.var_plans:
        return 0
    return job.n_workers * max(vp.global_bytes for vp in job.var_plans)

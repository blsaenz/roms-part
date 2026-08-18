"""Per-variable reconstruction + shard write, and the final stitch pass.
Runs the parallel half inside worker processes; the stitch half runs once,
serially, in the orchestrating process.

Architecture (see README.md for the full rationale and the empirical
findings behind it):
  - Each worker reconstructs ONE variable's whole-domain array by scattering
    each contributing tile's local data into a pre-sized array at that
    tile's write_placement offset (validated bit-for-bit against real
    `ncjoin` output).
  - The worker then writes that array into ITS OWN independent shard file
    -- safe, zero-contention, exactly like `partit`'s proven per-file
    writes. Multidim variables are written as a SINGLE HDF5 chunk, using
    the FINAL desired compression settings, so the shard's raw compressed
    bytes are byte-for-byte injectable into the destination later.
  - Plain HDF5 does NOT allow multiple processes to hold one file open for
    writing concurrently (confirmed empirically: it raises
    `BlockingIOError: unable to lock file` rather than corrupting data),
    so the actual destination file is only ever written by the
    orchestrating process, never by workers directly.
  - The stitch pass (in the orchestrating process, after all workers
    finish) uses h5py's low-level `read_direct_chunk`/`write_direct_chunk`
    to copy each multidim shard's ALREADY-COMPRESSED bytes straight into
    the destination's matching chunk -- no decompression, no
    recompression. This is what makes compressed output cheap: the
    expensive compression work already happened in parallel across
    workers; the stitch is just a byte copy (measured at a few hundred ms
    total, vs. tens of seconds for a naive single-process recompression).
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional

import h5py
import netCDF4
import numpy as np

from .dims import PARTITIONABLE_DIMS
from .join_decomposition import write_placement
from .join_job import JoinJob, JoinVarPlan

_JOB: Optional[JoinJob] = None
_TILE_DATASETS: Optional[dict[int, netCDF4.Dataset]] = None


@dataclass(frozen=True)
class ShardResult:
    name: str
    dim_names: tuple[str, ...]
    dtype: str
    attrs: dict
    out_shape: tuple[int, ...]
    is_multidim: bool
    shard_path: str


def _init_worker(job: JoinJob) -> None:
    global _JOB, _TILE_DATASETS
    _JOB = job
    _TILE_DATASETS = {tf.node: netCDF4.Dataset(tf.path, mode="r") for tf in job.tile_files}


def _global_shape(vp: JoinVarPlan, job: JoinJob) -> tuple[int, ...]:
    canon_axes = {d: i for i, d in enumerate(vp.dim_names) if d in PARTITIONABLE_DIMS}
    sample_ds = _TILE_DATASETS[vp.contributing_nodes[0]]
    out_shape = list(sample_ds.variables[vp.name].shape)
    global_size_by_kind: dict[str, int] = {}
    for node in vp.contributing_nodes:
        ti = job.tile_infos[node]
        for kind in canon_axes:
            start, count = write_placement(kind, ti, job.extent)
            global_size_by_kind[kind] = max(global_size_by_kind.get(kind, 0), start + count)
    for kind, axis in canon_axes.items():
        out_shape[axis] = global_size_by_kind[kind]
    return tuple(out_shape), canon_axes


def reconstruct_and_write_shard(varname: str) -> ShardResult:
    job = _JOB
    vp = next(v for v in job.var_plans if v.name == varname)
    attrs_src = _TILE_DATASETS[vp.contributing_nodes[-1]].variables[varname]
    attrs = {a: attrs_src.getncattr(a) for a in attrs_src.ncattrs() if a != "_FillValue"}

    if not any(d in PARTITIONABLE_DIMS for d in vp.dim_names):
        data = _TILE_DATASETS[vp.contributing_nodes[0]].variables[varname][:]
        out_shape = tuple(data.shape)
        data = np.asarray(data)
    else:
        out_shape, canon_axes = _global_shape(vp, job)
        data = np.empty(out_shape, dtype=vp.dtype)
        for node in vp.contributing_nodes:
            ti = job.tile_infos[node]
            arr = _TILE_DATASETS[node].variables[varname][:]
            slices = [slice(None)] * len(vp.dim_names)
            for kind, axis in canon_axes.items():
                start, count = write_placement(kind, ti, job.extent)
                slices[axis] = slice(start, start + count)
            data[tuple(slices)] = arr

    shard_path = os.path.join(job.shard_dir, f"{varname}.h5")
    with h5py.File(shard_path, "w") as f:
        kwargs = {}
        if vp.is_multidim and job.deflate_level > 0:
            kwargs = dict(
                chunks=out_shape,
                compression="gzip",
                compression_opts=job.deflate_level,
                shuffle=job.shuffle,
            )
        f.create_dataset("data", data=data, **kwargs)

    return ShardResult(
        name=varname,
        dim_names=vp.dim_names,
        dtype=str(vp.dtype),
        attrs=attrs,
        out_shape=out_shape,
        is_multidim=vp.is_multidim,
        shard_path=shard_path,
    )


def stitch(job: JoinJob, shard_results: dict[str, ShardResult]) -> None:
    """Runs once, serially, in the orchestrating process (not a worker)."""
    xi_rho, eta_rho = job.extent.xi_rho, job.extent.eta_rho
    dim_size_overrides = {
        "xi_rho": xi_rho,
        "xi_u": xi_rho - 1,
        "eta_rho": eta_rho,
        "eta_v": eta_rho - 1,
    }

    with netCDF4.Dataset(job.output_path, mode="w", format="NETCDF4") as dst:
        for dname, size in job.raw_dim_sizes.items():
            if dname == job.unlimited_dim:
                dst.createDimension(dname, None)
            else:
                dst.createDimension(dname, dim_size_overrides.get(dname, size))
        for k, v in job.global_attrs.items():
            dst.setncattr(k, v)

        for vp in job.var_plans:
            sr = shard_results[vp.name]
            kwargs = {}
            if sr.is_multidim and job.deflate_level > 0:
                kwargs = dict(
                    zlib=True, complevel=job.deflate_level, shuffle=job.shuffle,
                    chunksizes=sr.out_shape,
                )
            if vp.fill_value is not None:
                kwargs["fill_value"] = vp.fill_value
            var = dst.createVariable(vp.name, vp.dtype, vp.dim_names, **kwargs)
            for a, val in sr.attrs.items():
                var.setncattr(a, val)
            if not sr.is_multidim:
                with h5py.File(sr.shard_path, "r") as sf:
                    var[...] = sf["data"][()]

    with h5py.File(job.output_path, "r+") as dest:
        for vp in job.var_plans:
            sr = shard_results[vp.name]
            if not sr.is_multidim:
                continue
            with h5py.File(sr.shard_path, "r") as sf:
                filter_mask, raw = sf["data"].id.read_direct_chunk((0,) * len(sr.out_shape))
            dset = dest[vp.name]
            if dset.shape != sr.out_shape:
                # A variable with an unlimited (record) dimension starts at
                # extent 0 along that axis until grown -- write_direct_chunk
                # requires the dataset's current extent to already cover the
                # target chunk, so resize before injecting.
                dset.resize(sr.out_shape)
            dset.id.write_direct_chunk((0,) * len(sr.out_shape), raw, filter_mask=filter_mask)

# roms-part

Fast, parallel Python replacements for two [UCLA-ROMS](https://github.com/CESR-lab/ucla-roms)
netCDF (serial) domain-decomposition tools:

- **`roms-part`** replaces `Tools-Roms/partit.F` -- splits a whole-domain
  ROMS netCDF file (grid/forcing/initial-condition/boundary) into
  per-MPI-subdomain tiles that the running model reads directly.
- **`roms-join`** replaces `Tools-Roms/ncjoin.F` (and, for the scaling
  problem it solves, competes with the MPI-parallel `ncjoin_mpi.F`) --
  joins those tiles back into one whole-domain, compressed netCDF file.

MPI-parallel FORTRAN partition/join tools do exist but under many
circumstances are difficult to comile and use.  This python package
attempts similar functionality with easier setup.

## Install

```bash
pip install -e ".[test]"
```

Requires Python >=3.12, `netCDF4`, `numpy`, `h5py`.

## Quick start

Use CLI commands to partition/join UCLA-ROMS netCDF files:

### Partition
```bash
roms-part NP_XI NP_ETA file1.nc [file2.nc ...] --n-workers N --output-dir DIR
```
where `N` can range up to ~24 if sufficient cores/memory is available,
and gains can still be realized.  `NP_XI` and `NP_ETA` are integers
describing the tiling, `DIR` is an (optional) output directory where
tiled files will be created.

### Join
```bash
roms-join file1.*.nc [file2.*.nc] ... --n-workers N --output OUTFILE
```
where `N` can range up to 8 cores, if available, although tests showed
that gains using more than 4 cores may not be significant (joining
is a less-parallelizable process than partitioning under most
filesystems). `*` is a wildcard representing the typical MPI tile index
of individual netCDF files, and `OUTFILE` is the joined output netCDF
file.

NOTE: the wildcard `*` expansion happens at the shell level. This is
package is completely untested under windows.


## roms-part

### Why

`partit.F` is correct but slow at scale: for a large file split into many
tiles (e.g. 16x32=512), the dominant cost is single-threaded, CPU-bound
HDF5 deflate compression, executed serially in `nf_close()` for every
output file, one core, one file at a time. Since the output files are
fully independent (no shared state between them), this is an ideal target
for process-level parallelism: `roms-part` fans the per-node writes out
across a process pool, so each independent output file compresses on its
own core with zero contention.

Measured on a real 1.7GB ROMS initial-conditions file (674x1322x100, 4x4
and 8x8 partitions), on an 8-core Mac:

| workers | wall-clock | vs. Fortran `partit` |
|---|---|---|
| Fortran `partit` (serial) | ~16-18s | 1.0x |
| roms-part, 1 worker | ~18-20s | ~0.95-1.0x |
| roms-part, 4 workers | ~5s | ~3.2x |
| roms-part, 8 workers | ~3s | ~5.4x |

verified bit-for-bit identical to the Fortran tool's output at every
worker count (see `benchmarks/bench_partit.py --verify`).

### Design

- **Bit-for-bit compatible.** `roms_part/decomposition.py` is a literal
  port of `partit.F`'s `mpi_setup` domain-decomposition arithmetic;
  `roms_part/naming.py` ports `insert_node.F`'s filename convention;
  `roms_part/dims.py` ports the obsolete-dimension remap
  (`xi_psi`->`xi_u`, `xi_v`->`xi_rho`, `eta_psi`->`eta_v`, `eta_u`->`eta_rho`)
  and boundary-suffix (`_west`/`_east`/`_south`/`_north`) variable
  handling -- including a couple of Fortran quirks/bugs (an asymmetric
  eta_rho/eta_v inference branch, see `dims.py`'s docstring) that are
  deliberately reproduced rather than "fixed," since existing tooling
  (`ncjoin`, `ncjoin_mpi`, the ROMS runtime itself) depends on the exact
  on-disk convention `partit.F` actually produces.
- **Parallelism via `concurrent.futures.ProcessPoolExecutor`, not
  `dask.distributed`.** The work is embarrassingly parallel across fully
  independent output files -- there's no task graph to optimize, so a
  scheduler process and inter-process task dispatch would be pure
  overhead. Workers are spawned with the `spawn` multiprocessing context;
  each worker process gets its own independently-loaded copy of
  libhdf5/libnetcdf, so there's no shared HDF5 state between workers at
  all -- this is what actually sidesteps HDF5's thread-safety hazards
  (which are specifically about multiple *threads* sharing one loaded
  library instance, not separate processes). Every worker opens the same
  source file read-only and writes to its own distinct output file, so no
  two processes ever touch the same file.
- Each worker task creates, writes, and closes one complete output file
  before returning -- at most `n_workers` output files are ever open at
  once, unlike the Fortran design of holding all `nnodes` files open
  simultaneously for the whole run.

### Usage

```bash
roms-part NP_XI NP_ETA file1.nc [file2.nc ...]
```

Preserves `partit`'s positional-argument convention for drop-in
compatibility. Additional flags:

| Flag | Default | Meaning |
|---|---|---|
| `--n-workers N` | `min(cpu_count, 8)` | Worker-process count |
| `--deflate-level N` | `1` | 0 disables compression |
| `--shuffle` / `--no-shuffle` | on | HDF5 shuffle filter |
| `--output-dir DIR` | `.` | Where node files are written |
| `--time-batch-size N` | none (whole record dim at once) | Bounds peak memory when copying a large record/unlimited dimension |
| `--keep-going` | off | Continue with remaining files after a per-file error |
| `--dry-run` | off | Print the computed decomposition/filenames without writing |
| `-v` / `--verbose` | off | Per-node progress output |

## roms-join

### Why

`ncjoin`'s job is the inverse of `partit`'s: read N independent tiles back
into **one** shared output file. That's structurally harder to
parallelize than `partit`, because multiple writers touching the same
file need real coordination, and plain HDF5 only allows one writer to
safely touch a *compressed* dataset at a time -- confirmed empirically
here (see `roms_part/join_worker.py`'s module docstring): having two
processes each hold the same file open for writing at once doesn't
silently corrupt data, it raises `BlockingIOError: unable to lock file`.
This is exactly why the existing Fortran `ncjoin_mpi` (already
MPI-parallel!) only gets 8x/12x/16x speedup at 25/50/100 cores despite its
independent-mode *reads* scaling fine -- its writes are forced into a
synchronized "collective" HDF5 mode by a master-only-file-creation
architecture, and its own author never got independent (non-collective)
compressed writes working.

Measured on the same real 1.7GB file (4x4 tiles), on an 8-core Mac:

| | wall-clock | vs. Fortran `ncjoin` |
|---|---|---|
| Fortran `ncjoin` (serial, compressed) | ~22s | 1.0x |
| roms-join, 4 workers | ~7s | ~3.1x |
| roms-join, 8 workers | ~7-8s | ~2.9-3.1x |

verified bit-for-bit identical to both the Fortran tool's output and the
original pre-partition source file (see `benchmarks/bench_ncjoin.py
--verify`).

### Design

Parallel reconstruction, with compression fully preserved -- the key
trick that makes this possible without giving up compressed output:

1. **Read/reconstruct in parallel, one worker process per variable.**
   Each worker scatters every contributing tile's local data into a
   pre-sized whole-domain array at that tile's reconstructed placement
   offset (`roms_part/join_decomposition.py`, ported from `ncjoin.F`'s
   join-side domain reconstruction -- which, unlike `partit`'s forward
   `mpi_setup`, needs no `NP_XI`/`NP_ETA` at all: the whole-domain extent
   and every tile's edge flags are derived purely from the `partition`
   global attribute plus each tile's own local dimension sizes).
2. **Each worker writes its variable to its own independent shard file**,
   compressed with the *final* desired settings, as a single HDF5 chunk.
   This is exactly `partit`'s proven safe pattern (fully independent
   files, zero contention) -- so the actual compression work, the
   expensive part, happens fully in parallel.
3. **The real destination file is never written concurrently.** A single
   orchestrating process creates it and, once all workers finish, stitches
   the shards in using `h5py`'s low-level `read_direct_chunk`/
   `write_direct_chunk` API to copy each shard's **already-compressed
   bytes** straight into the destination's matching chunk -- no
   decompression, no recompression, just a byte copy. Measured at a few
   hundred milliseconds total, vs. tens of seconds for a naive
   single-process recompression of the same data.

This means the serial bottleneck that limits both plain `ncjoin` and
`ncjoin_mpi` -- paying the full HDF5 deflate cost once, on one core, for
the whole joined file -- only has to happen once *per variable*, and
those happen concurrently across workers, not once for the whole file
sequentially.

Parallelism here scales with variable count (12 tasks for the test file
above), so the ceiling is set by the single largest variable's own
compression time; a file with many variables (BGC restart files can have
200+) scales further than a file with one huge variable and few others.
Splitting large variables into sub-chunks across workers (mirroring
`ncjoin_mpi.F`'s own spatial chunking) is a natural future refinement, not
needed for a first working version.

### Usage

```bash
roms-join file1.NNN.nc file2.NNN.nc ... [--output out.nc]
```

Preserves `ncjoin`'s "pass every tile file" convention. Additional flags:

| Flag | Default | Meaning |
|---|---|---|
| `--output PATH` | derived from tile names | Joined output path |
| `--n-workers N` | `min(cpu_count, 8)` | Worker-process count (capped at variable count) |
| `--deflate-level N` | `1` | 0 disables compression |
| `--shuffle` / `--no-shuffle` | on | HDF5 shuffle filter |
| `-d` / `--delete` | off | Delete tile files after a verified-successful join |
| `--keep-shards` | off | Keep the temporary per-variable shard files (debugging) |
| `-v` / `--verbose` | off | Extra progress output |

## Testing

```bash
pip install -e ".[test]"
export ROMS_ROOT=/path/to/ucla-roms   # or place a checkout at ~/Projects/git/ucla-roms
pytest
```

The test suite's central claim is bit-for-bit equivalence with the real
Fortran tool, so most tests build the actual `partit` binary once per
session (from a temp copy of `Tools-Roms`, via `make partit COMPILER=gnu`
-- never touching your real ucla-roms working tree) and diff every node
file's dimensions, the `partition` global attribute, other global
attributes, and all variable data against `roms-part`'s output on small
synthetic grids. Tests requiring a ucla-roms checkout are skipped (not
failed) if `ROMS_ROOT` isn't set and no sibling checkout is found.

- `tests/test_decomposition.py` -- `mpi_setup`/`node_dim_extent` vs. real
  `partit`, across a matrix of `NP_XI`/`NP_ETA`/grid-size combinations
  including uneven-remainder cases.
- `tests/test_naming.py` -- golden filename values captured from a
  standalone Fortran harness that calls `insert_node.F` the way
  `partit.F` actually does (once per node, from a pristine filename --
  not `insert_node.F`'s own `TEST_INSERT` self-test, which repeatedly
  mutates the same string across iterations and isn't representative of
  real usage).
- `tests/test_dims.py` -- obsolete-dim remap, `part_type` classification,
  boundary-suffix strict-length-inequality guard.
- `tests/test_integration.py` -- end-to-end `roms-part` vs. real `partit`
  on synthetic files covering obsolete dims, boundary variables, extra
  netCDF dtypes (byte/short/int), and time-record dimensions (with and
  without `--time-batch-size`).
- `tests/test_join_decomposition.py` -- join-side extent/edge
  reconstruction vs. real `partit`-produced tiles (via the `make_tiles`
  fixture), across the same `NP_XI`/`NP_ETA` matrix as the partit-side
  tests, plus cross-checked against `mpi_setup`'s own ground truth.
- `tests/test_join_schema.py` -- tile-set discovery and the
  filename/`partition`-attribute cross-check, including all the error
  paths (missing node, mismatched nnodes, filename/attribute
  disagreement, duplicate node).
- `tests/test_join_integration.py` -- end-to-end `roms-join` on synthetic
  tiles, diffed against the *original pre-partition source file*
  (the cleanest correctness oracle, since no Fortran intermediate is
  needed for ground truth) across boundary variables, extra dtypes, time
  records, and uncompressed output; one test also cross-validates
  directly against real `ncjoin`'s own output.

## Benchmarking

```bash
python benchmarks/bench_partit.py \
    --partit /path/to/compiled/partit \
    --file /path/to/real_file.nc \
    --np-xi 16 --np-eta 32 \
    --workers 1 2 4 8 \
    --verify

python benchmarks/bench_ncjoin.py \
    --ncjoin /path/to/compiled/ncjoin \
    --tiles '/path/to/tiles/file.*.nc' \
    --workers 1 2 4 8 \
    --original /path/to/original_source.nc \
    --verify
```

Neither is part of the pytest suite (each needs a real compiled Fortran
binary and a real, potentially large, input file/tile set). Both sweep
`--n-workers`, report wall-clock and speedup vs. the Fortran binary, and
optionally verify bit-for-bit equivalence between the two tools' output
(`bench_ncjoin.py` can additionally check against the pre-partition
original via `--original`).

## Scope / known limitations

- Full generality is a goal (matching `partit.F`'s dimension-name
  detection, obsolete-dim handling, arbitrary variable/type sets), but
  the "classic vs. mask-aware" fork in ucla-roms' `New-tools/` (a
  separate, MPI-parallel, land-mask-aware rewrite) is explicitly out of
  scope here.
- Non-partitioned variable data is currently re-read from the source file
  independently by each worker rather than loaded once and broadcast;
  this is simple and correct, and only matters for unusually large
  non-partitioned arrays (grid/mask fields are typically small relative
  to the partitioned 3D fields).
- `roms-join`'s parallelism is per-variable, so a file dominated by one
  enormous variable and few others won't benefit as much as a file with
  many variables; sub-variable spatial chunking (as `ncjoin_mpi.F` does)
  is a natural future refinement.
- `roms-join` currently reconstructs each variable fully in memory before
  writing its shard -- fine for typical ROMS fields, but a file with an
  extremely large individual variable (relative to available RAM) would
  need chunked/streamed reconstruction, which isn't implemented yet.
- True parallel-HDF5/MPI-IO writes (mirroring `ncjoin_mpi.F`'s own
  architecture via `mpi4py` + a parallel-built `netCDF4`/`h5py`) remain a
  documented stretch-goal fallback, not needed here since the
  parallel-compress + direct-chunk-stitch design already beats both
  `ncjoin` and `ncjoin_mpi` without requiring any new, MPI-flavor-
  sensitive environment changes.

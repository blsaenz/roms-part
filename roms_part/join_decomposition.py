"""Pure port of ncjoin.F / ncjoin_mod.F's join-side domain RECONSTRUCTION:
given only the per-tile `partition` global attribute ([node, nnodes,
xi_start, eta_start], as produced by partit/decomposition.py) plus each
tile's own local dimension sizes, reconstruct the whole-domain XI_rho/
ETA_rho extent and each tile's edge flags and write-placement -- WITHOUT
ever knowing NP_XI/NP_ETA. This is genuinely simpler than partit's forward
mpi_setup (ncjoin.F lines 706-857 / ncjoin_mod.F determine_joined_file_dim_sizes
+ indentify_boundary_edges).

No I/O here. Every function is a deterministic computation over plain
values read from tile files elsewhere (schema.py-style inspection), so it
can be unit-tested against real compiled `ncjoin` output.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

from .dims import PARTITIONABLE_DIMS


@dataclass(frozen=True)
class TileInfo:
    """One tile's worth of information needed for join reconstruction --
    the Python-side equivalent of one row of ncjoin.F's per-node bookkeeping
    arrays (xi_start, eta_start, dimsize(:,node))."""

    node: int
    xi_start: int
    eta_start: int
    local_dims: Mapping[str, int]  # e.g. {"xi_rho": 172, "xi_u": 171, ...}


@dataclass(frozen=True)
class GlobalExtent:
    xi_rho: int
    eta_rho: int


def reconstruct_global_extent(tiles: Sequence[TileInfo]) -> GlobalExtent:
    """Port of ncjoin.F lines 706-783 / determine_joined_file_dim_sizes.

    XI_rho = max over tiles of (local xi_rho size + xi_start - 1).
    XI_rho, reconstructed via xi_u, must agree: for non-western tiles
    (xi_start > 1), local xi_u size + xi_start - 2, then +1 to convert back
    to the RHO extent; for the western tile (xi_start == 1), local xi_u
    size itself already equals RHO extent - 1 (no adjustment).
    Symmetric for eta_rho/eta_v with eta_start.
    """
    xi_rho = 0
    for t in tiles:
        if "xi_rho" in t.local_dims:
            xi_rho = max(xi_rho, t.local_dims["xi_rho"] + t.xi_start - 1)
        if "xi_u" in t.local_dims:
            if t.xi_start > 1:
                xi_rho = max(xi_rho, t.local_dims["xi_u"] + t.xi_start - 2 + 1)
            else:
                xi_rho = max(xi_rho, t.local_dims["xi_u"] + 1)

    eta_rho = 0
    for t in tiles:
        if "eta_rho" in t.local_dims:
            eta_rho = max(eta_rho, t.local_dims["eta_rho"] + t.eta_start - 1)
        if "eta_v" in t.local_dims:
            if t.eta_start > 1:
                eta_rho = max(eta_rho, t.local_dims["eta_v"] + t.eta_start - 2 + 1)
            else:
                eta_rho = max(eta_rho, t.local_dims["eta_v"] + 1)

    if xi_rho == 0 or eta_rho == 0:
        raise ValueError("could not reconstruct global xi_rho/eta_rho from tile set")
    return GlobalExtent(xi_rho=xi_rho, eta_rho=eta_rho)


@dataclass(frozen=True)
class TileEdges:
    western_edge: bool
    eastern_edge: bool
    southern_edge: bool
    northern_edge: bool


def reconstruct_edges(tile: TileInfo, extent: GlobalExtent) -> TileEdges:
    """Port of ncjoin.F lines 817-857 / indentify_boundary_edges: a tile is
    eastern-edge iff its RHO (or U) sub-block, placed at xi_start, reaches
    exactly to XI_rho; symmetric for the other three edges. No NP_XI/NP_ETA
    needed -- purely geometric, derived from the reconstructed extent.
    """
    western_edge = tile.xi_start == 1
    southern_edge = tile.eta_start == 1

    eastern_edge = True
    if "xi_rho" in tile.local_dims:
        eastern_edge = eastern_edge and not (
            tile.xi_start + tile.local_dims["xi_rho"] < extent.xi_rho
        )
    if "xi_u" in tile.local_dims:
        eastern_edge = eastern_edge and not (
            tile.xi_start + tile.local_dims["xi_u"] < extent.xi_rho
        )

    northern_edge = True
    if "eta_rho" in tile.local_dims:
        northern_edge = northern_edge and not (
            tile.eta_start + tile.local_dims["eta_rho"] < extent.eta_rho
        )
    if "eta_v" in tile.local_dims:
        northern_edge = northern_edge and not (
            tile.eta_start + tile.local_dims["eta_v"] < extent.eta_rho
        )

    return TileEdges(
        western_edge=western_edge,
        eastern_edge=eastern_edge,
        southern_edge=southern_edge,
        northern_edge=northern_edge,
    )


def write_placement(dim_kind: str, tile: TileInfo, extent: GlobalExtent) -> tuple[int, int]:
    """(0-based global write start, tile's own local element count) for a
    partitionable axis, matching ncjoin.F's start1 (lines ~1400-1453):

        xi_rho:  start1 = xi_start
        xi_u:    start1 = max(xi_start-1, 1)
        eta_rho: start1 = eta_start
        eta_v:   start1 = max(eta_start-1, 1)

    Returned start is converted to 0-based; the count is simply the tile's
    own local dimension size (verified empirically against real `partit`/
    `ncjoin` output: adjacent tiles' placements abut with ZERO overlap or
    gap -- e.g. for a 4-tile xi_u row, tile 0 occupies [0,167], tile 1
    occupies exactly [168,335], etc. This means a direct per-tile SCATTER
    into a pre-sized global array using this start position is correct and
    simplest; no trimming or grid-shape inference is needed. (An earlier
    version of this function assumed a 1-element seam overlap requiring a
    trim before concatenation -- that assumption was wrong, disproven by
    reconstructing real tiles and comparing against the original
    pre-partition source file; the `xi_start - 1` in the Fortran formula
    is an index-base conversion for the staggered U/V grid, not evidence
    of redundant/overlapping data between tiles.)
    """
    if dim_kind == "xi_rho":
        return tile.xi_start - 1, tile.local_dims["xi_rho"]
    elif dim_kind == "xi_u":
        return max(tile.xi_start - 1, 1) - 1, tile.local_dims["xi_u"]
    elif dim_kind == "eta_rho":
        return tile.eta_start - 1, tile.local_dims["eta_rho"]
    elif dim_kind == "eta_v":
        return max(tile.eta_start - 1, 1) - 1, tile.local_dims["eta_v"]
    raise ValueError(f"not a partitionable dim kind: {dim_kind!r}")

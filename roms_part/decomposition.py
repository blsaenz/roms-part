"""Pure port of partit.F's mpi_setup subroutine and the per-variable
read/write index math (partit.F lines ~1493-1597 and ~1213-1257).

No I/O here. Every function is a deterministic integer computation so it
can be unit-tested in isolation and compared bit-for-bit against the
compiled Fortran `partit` binary's output (dimension sizes and the
`partition` global attribute).
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class NodeInfo:
    node: int
    inode: int
    jnode: int
    xi_start: int
    xi_size: int
    eta_start: int
    eta_size: int
    western_edge: bool
    eastern_edge: bool
    southern_edge: bool
    northern_edge: bool


def mpi_setup(np_xi: int, np_eta: int, xi_rho: int, eta_rho: int) -> list[NodeInfo]:
    """Literal port of partit.F's mpi_setup (lines 1493-1597).

    xi_rho/eta_rho are the FULL rho-grid dimension sizes as they appear in
    the source netCDF file (i.e. including exactly one ghost/boundary point
    on each side of the interior). Returns one NodeInfo per node, in the
    same row-major (jnode = node // np_xi) order Fortran uses by default
    (the TRANSPOSED_MPI_NODE_ORDER branch is commented out in partit.F, so
    only this ordering needs to be supported).
    """
    nnodes = np_xi * np_eta
    llm = xi_rho - 2
    mmm = eta_rho - 2
    lm = (llm + np_xi - 1) // np_xi
    mm = (mmm + np_eta - 1) // np_eta

    nodes = []
    for node in range(nnodes):
        jnode = node // np_xi
        inode = node - jnode * np_xi

        off_xi = np_xi * lm - llm
        isw_corn = inode * lm - off_xi // 2
        if inode == 0:
            iwest = 1 + off_xi // 2
        else:
            iwest = 1
        if inode < np_xi - 1:
            ieast = lm
        else:
            ieast = lm - (off_xi + 1) // 2

        off_eta = np_eta * mm - mmm
        jsw_corn = jnode * mm - off_eta // 2
        if jnode == 0:
            jsouth = 1 + off_eta // 2
        else:
            jsouth = 1
        if jnode < np_eta - 1:
            jnorth = mm
        else:
            jnorth = mm - (off_eta + 1) // 2

        xi_size = ieast - iwest + 1
        eta_size = jnorth - jsouth + 1

        if inode == 0:
            xi_start = isw_corn + iwest
        else:
            xi_start = isw_corn + iwest + 1

        if jnode == 0:
            eta_start = jsw_corn + jsouth
        else:
            eta_start = jsw_corn + jsouth + 1

        nodes.append(
            NodeInfo(
                node=node,
                inode=inode,
                jnode=jnode,
                xi_start=xi_start,
                xi_size=xi_size,
                eta_start=eta_start,
                eta_size=eta_size,
                western_edge=(inode == 0),
                eastern_edge=(inode == np_xi - 1),
                southern_edge=(jnode == 0),
                northern_edge=(jnode == np_eta - 1),
            )
        )
    return nodes


def node_dim_extent(dim_kind: str, node: NodeInfo) -> int:
    """Output-file dimension SIZE for this node (partit.F lines 403-423)."""
    if dim_kind == "xi_rho":
        size = node.xi_size
        if node.western_edge:
            size += 1
        if node.eastern_edge:
            size += 1
        return size
    elif dim_kind == "xi_u":
        size = node.xi_size
        if node.eastern_edge:
            size += 1
        return size
    elif dim_kind == "eta_rho":
        size = node.eta_size
        if node.southern_edge:
            size += 1
        if node.northern_edge:
            size += 1
        return size
    elif dim_kind == "eta_v":
        size = node.eta_size
        if node.northern_edge:
            size += 1
        return size
    raise ValueError(f"not a partitionable dim kind: {dim_kind!r}")


def node_read_range(dim_kind: str, node: NodeInfo) -> tuple[int, int]:
    """(0-based start, count) for reading the hyperslab out of the SOURCE
    file for this node (partit.F lines 1213-1257, converted from Fortran's
    1-based start to a 0-based Python offset). The output-side write always
    starts at local origin 0 (Fortran's start1=1 always), so callers just
    assign into dst[..., 0:count, ...] -- no separate write-range needed.
    """
    if dim_kind == "xi_rho":
        start = node.xi_start - 1
        count = node.xi_size
        if node.western_edge:
            count += 1
        if node.eastern_edge:
            count += 1
        return start, count
    elif dim_kind == "xi_u":
        start = node.xi_start - 1
        count = node.xi_size
        if not node.western_edge:
            start -= 1
        if node.eastern_edge:
            count += 1
        return start, count
    elif dim_kind == "eta_rho":
        start = node.eta_start - 1
        count = node.eta_size
        if node.southern_edge:
            count += 1
        if node.northern_edge:
            count += 1
        return start, count
    elif dim_kind == "eta_v":
        start = node.eta_start - 1
        count = node.eta_size
        if not node.southern_edge:
            start -= 1
        if node.northern_edge:
            count += 1
        return start, count
    raise ValueError(f"not a partitionable dim kind: {dim_kind!r}")

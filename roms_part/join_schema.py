"""Discover and validate a complete set of partit-produced tile files ready
to be joined, mirroring ncjoin.F's file-set cross-check (lines 194-459):
each tile's `partition` global attribute is authoritative for node/nnodes/
xi_start/eta_start, but the node index embedded in the filename (via the
same convention naming.insert_node_filename implements) must agree with
it, and a complete set requires nodes 0..nnodes-1 all present.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import netCDF4

from .naming import digit_width, insert_node_filename


class TileSetError(ValueError):
    """Raised for any inconsistency in a candidate tile set (mismatched
    nnodes, missing nodes, filename/attribute disagreement, etc.)."""


@dataclass(frozen=True)
class TileFile:
    path: str
    node: int
    nnodes: int
    xi_start: int
    eta_start: int


def extract_node_and_root(filename: str, nnodes: int) -> tuple[str, int]:
    """Inverse of naming.insert_node_filename: given a bare filename (no
    directory) and the tile set's nnodes, recover (root, node) such that
    insert_node_filename(root, node, nnodes) == filename. Raises
    TileSetError if the filename doesn't match the expected convention for
    this nnodes' digit width.
    """
    width = digit_width(nnodes)
    last_dot = filename.rfind(".")
    has_real_suffix = False
    if last_dot >= 0:
        trailing = filename[last_dot + 1 :]
        if trailing and not trailing.isdigit():
            has_real_suffix = True

    if has_real_suffix:
        digit_start = last_dot - width
        if digit_start < 0 or filename[digit_start - 1 : digit_start] != ".":
            raise TileSetError(
                f"filename {filename!r} doesn't match the expected "
                f"node-digit convention for nnodes={nnodes} (width={width})"
            )
        digits_str = filename[digit_start:last_dot]
        root = filename[: digit_start - 1] + filename[last_dot:]
    else:
        digit_start = len(filename) - width
        if digit_start < 1 or filename[digit_start - 1] != ".":
            raise TileSetError(
                f"filename {filename!r} doesn't match the expected "
                f"node-digit convention for nnodes={nnodes} (width={width})"
            )
        digits_str = filename[digit_start:]
        root = filename[: digit_start - 1]

    if not digits_str.isdigit() or len(digits_str) != width:
        raise TileSetError(
            f"filename {filename!r}: expected a {width}-digit node segment, got {digits_str!r}"
        )
    node = int(digits_str)

    # Round-trip check: regenerating the filename from (root, node, nnodes)
    # must reproduce exactly what we started with.
    if insert_node_filename(root, node, nnodes) != filename:
        raise TileSetError(
            f"filename {filename!r} failed round-trip check against root={root!r}"
        )
    return root, node


def _read_partition_attr(path: Path) -> tuple[int, int, int, int]:
    with netCDF4.Dataset(path, mode="r") as ds:
        if "partition" not in ds.ncattrs():
            raise TileSetError(f"'{path}' has no 'partition' global attribute -- not a tile file")
        node, nnodes, xi_start, eta_start = (int(x) for x in ds.getncattr("partition"))
    return node, nnodes, xi_start, eta_start


def discover_tile_set(paths: list[str]) -> list[TileFile]:
    """Validate `paths` as a complete tile set and return one TileFile per
    node, sorted by node index. Raises TileSetError on any inconsistency:
    mismatched nnodes across files, a filename/attribute node-index
    disagreement, a duplicate node, or a missing node in 0..nnodes-1.
    """
    if not paths:
        raise TileSetError("no candidate tile files given")

    nnodes: Optional[int] = None
    by_node: dict[int, TileFile] = {}

    for p in paths:
        path = Path(p)
        node, this_nnodes, xi_start, eta_start = _read_partition_attr(path)
        if nnodes is None:
            nnodes = this_nnodes
        elif this_nnodes != nnodes:
            raise TileSetError(
                f"'{path}': nnodes={this_nnodes} disagrees with the tile "
                f"set's nnodes={nnodes} (established by an earlier file)"
            )

        _, node_from_name = extract_node_and_root(path.name, nnodes)
        if node_from_name != node:
            raise TileSetError(
                f"'{path}': filename encodes node {node_from_name} but the "
                f"'partition' attribute says node {node} -- refusing to join"
            )

        if node in by_node:
            raise TileSetError(f"duplicate node {node}: '{by_node[node].path}' and '{path}'")
        by_node[node] = TileFile(
            path=str(path), node=node, nnodes=nnodes, xi_start=xi_start, eta_start=eta_start
        )

    missing = [n for n in range(nnodes) if n not in by_node]
    if missing:
        raise TileSetError(f"incomplete tile set: missing node(s) {missing} of {nnodes}")

    return [by_node[n] for n in range(nnodes)]

"""Pure port of partit.F's dimension/variable classification logic:
obsolete-dimension remapping (lines 298-341), part_type classification
(lines 539-558), and boundary-suffix variable detection (lines 578-593).

No I/O here -- these operate purely on names/sizes already read from a
source file's schema.
"""
from __future__ import annotations

from enum import IntEnum
from typing import Mapping, Optional, Sequence

_OBSOLETE_REMAP = {
    "xi_psi": "xi_u",
    "xi_v": "xi_rho",
    "eta_psi": "eta_v",
    "eta_u": "eta_rho",
}

PARTITIONABLE_DIMS = ("xi_rho", "xi_u", "eta_rho", "eta_v")


def remap_dim_name(name: str) -> str:
    """Map an obsolete/redundant dimension name onto its canonical
    replacement (partit.F lines 298-306); other names pass through
    unchanged."""
    return _OBSOLETE_REMAP.get(name, name)


def resolve_partitionable_dims(dim_sizes: Mapping[str, int]) -> dict[str, int]:
    """Port of partit.F lines 298-348: locate xi_rho/xi_u/eta_rho/eta_v by
    exact name.

    NOTE on a real Fortran asymmetry, preserved here deliberately: the XI
    branch infers whichever of xi_rho/xi_u is missing from the other
    (bidirectional). The ETA branch, however, is NOT symmetric in
    partit.F's actual source (lines 337-341) -- it only fills in eta_v from
    eta_rho; the `elseif` that would fill in eta_rho from eta_v alone reads
    `id_eta_rho == 0 .and. id_eta_v == 0` (BOTH absent), not `id_eta_v /= 0`
    as the XI branch's mirror would suggest. So a file defining eta_v but
    not eta_rho fails dimension resolution in the real tool (and the file
    is skipped with a warning) even though the symmetric xi_u-only case
    succeeds. This looks like a copy/paste bug in partit.F, but bit-for-bit
    fidelity means reproducing it, not "fixing" it.
    """
    xi_rho = dim_sizes.get("xi_rho", 0)
    xi_u = dim_sizes.get("xi_u", 0)
    eta_rho = dim_sizes.get("eta_rho", 0)
    eta_v = dim_sizes.get("eta_v", 0)

    if xi_rho and not xi_u:
        xi_u = xi_rho - 1
    elif xi_u and not xi_rho:
        xi_rho = xi_u + 1

    if eta_rho and not eta_v:
        eta_v = eta_rho - 1
    # NOTE: intentionally no `elif eta_v and not eta_rho: eta_rho = eta_v + 1`
    # here -- see docstring above; partit.F's real elseif condition can
    # never fire when eta_v is present and eta_rho is not.

    if not (xi_rho and xi_u and eta_rho and eta_v):
        raise ValueError(
            "not all partitionable dimensions (xi_rho, xi_u, eta_rho, "
            "eta_v) could be resolved from the source file's dimensions"
        )
    return {"xi_rho": xi_rho, "xi_u": xi_u, "eta_rho": eta_rho, "eta_v": eta_v}


class DimAxis(IntEnum):
    NONE = 0
    XI = 1
    ETA = 2
    BOTH = 3


def classify_variable(
    dim_names_after_remap: Sequence[str], unlimited_dim: Optional[str]
) -> tuple[DimAxis, bool]:
    """Returns (part_type, has_record_dim), matching partit.F lines 546-558.

    `dim_names_after_remap` must already have obsolete dim names mapped to
    their canonical form via remap_dim_name.
    """
    part_type = DimAxis.NONE
    has_record_dim = False
    for dname in dim_names_after_remap:
        if dname in ("xi_rho", "xi_u"):
            part_type = DimAxis(part_type | DimAxis.XI)
        elif dname in ("eta_rho", "eta_v"):
            part_type = DimAxis(part_type | DimAxis.ETA)
        elif dname == unlimited_dim:
            has_record_dim = True
    return part_type, has_record_dim


def boundary_edge(varname: str) -> Optional[str]:
    """Exact port of partit.F lines 578-593's strict-length-inequality
    suffix check: a variable only counts as boundary-only if its name is
    STRICTLY longer than the suffix itself (len > 5 for _west/_east, len > 6
    for _south/_north) -- a variable literally named e.g. "_west" does not
    match. Returns 'west'/'east'/'south'/'north' or None.
    """
    lvar = len(varname)
    if lvar > 5:
        if varname.endswith("_west"):
            return "west"
        if varname.endswith("_east"):
            return "east"
    if lvar > 6:
        if varname.endswith("_south"):
            return "south"
        if varname.endswith("_north"):
            return "north"
    return None

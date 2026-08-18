"""Pure port of insert_node.F's filename-mangling convention, as it is
actually invoked from partit.F (line 370): once per node, always starting
from a fresh copy of the bare source filename (partit.F resets
fname(node)=fname(nnodes-1) before every call, line 369) -- NOT the
repeated-call-on-already-mutated-string behavior exercised by
insert_node.F's own TEST_INSERT self-test block, which is a robustness
check on the routine itself rather than partit's real usage pattern.
"""
from __future__ import annotations


def digit_width(nnodes: int) -> int:
    """Number of decimal digits used to zero-pad the node number.

    Ported from insert_node.F lines 165-188: power starts at 10 and is
    repeatedly multiplied by 10 while nnodes > power; the digit count
    written out equals log10(power). Equivalent closed form: the smallest
    d such that 10**d >= nnodes (minimum 1). insert_node.F also hard-errors
    if this would exceed 5 digits (line ~171); we raise the same way.
    """
    power = 10
    while nnodes > power:
        power *= 10
    digits = len(str(power)) - 1
    if digits >= 5:
        raise ValueError(
            f"nnodes={nnodes} would require >=5 digits to encode the node "
            "number; insert_node.F's hardcoded 'digits' parameter caps at 5 "
            "(would need a source edit + recompile in the Fortran tool too)."
        )
    return digits


def insert_node_filename(name: str, node: int, nnodes: int) -> str:
    """Insert the zero-padded node number into `name`, matching insert_node.F.

    Any leading directory components in `name` are preserved verbatim; only
    the last '.' within the final path component is considered for a
    possible extension. A "real" extension is recognized only if there is
    at least one character after that last dot AND at least one of those
    characters is non-numeric (insert_node.F lines 74-86: the scan for a
    non-digit character must actually find one; an empty tail, or a
    trailing segment that is purely digits -- e.g. an already-inserted
    node/time-index segment from a prior invocation -- does NOT count).

    - If a real extension is found: the node number is inserted right
      after the dot, before that extension, e.g. ``root.nc`` -> ``root.03.nc``.
    - Otherwise: a brand-new ``.NNN`` segment is simply appended to
      whatever the current name already is -- including a bare trailing
      dot (``file.`` -> ``file..0``, a genuine double dot) or an existing
      purely-numeric segment (``file.123`` -> ``file.123.0``), neither of
      which gets modified or reused.
    """
    sep = name.rfind("/")
    fname_start = sep + 1
    basename = name[fname_start:]

    last_dot = basename.rfind(".")
    has_real_suffix = False
    suffix = ""
    if last_dot >= 0:
        trailing = basename[last_dot + 1 :]
        if trailing and not trailing.isdigit():
            has_real_suffix = True
            suffix = trailing

    width = digit_width(nnodes)
    digits_str = str(node).zfill(width)

    if has_real_suffix:
        root = basename[:last_dot]
        new_basename = f"{root}.{digits_str}.{suffix}"
    else:
        new_basename = f"{basename}.{digits_str}"

    return name[:fname_start] + new_basename

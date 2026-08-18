"""Golden-value tests for naming.insert_node_filename, captured from a real
compiled insert_node.F harness that mirrors partit.F's actual call pattern
(one fresh insert_node() call per node, from the pristine bare source
filename -- see naming.py's module docstring for why this differs from
insert_node.F's own TEST_INSERT self-test, which repeatedly mutates the
same string across iterations and is not representative of real usage).

Each (base filename, nnodes) golden set below was independently verified
against a small standalone Fortran harness built from the real
insert_node.F + lenstr.F, calling insert_node exactly the way partit.F does.
"""
from __future__ import annotations

import pytest

from roms_part.naming import digit_width, insert_node_filename

# base filename -> {nnodes: {node: expected_result}}
GOLDEN = {
    "../dir/root_name.nc": {
        5: {0: "../dir/root_name.0.nc", 4: "../dir/root_name.4.nc"},
        23: {0: "../dir/root_name.00.nc", 22: "../dir/root_name.22.nc"},
        151: {0: "../dir/root_name.000.nc", 148: "../dir/root_name.148.nc"},
    },
    "root_name": {
        5: {0: "root_name.0", 4: "root_name.4"},
        23: {0: "root_name.00", 22: "root_name.22"},
        151: {0: "root_name.000", 148: "root_name.148"},
    },
    "file.123.nc": {
        5: {0: "file.123.0.nc", 4: "file.123.4.nc"},
        23: {0: "file.123.00.nc", 22: "file.123.22.nc"},
        151: {0: "file.123.000.nc", 148: "file.123.148.nc"},
    },
    "noext_file": {
        5: {0: "noext_file.0", 4: "noext_file.4"},
        23: {0: "noext_file.00", 22: "noext_file.22"},
        151: {0: "noext_file.000", 148: "noext_file.148"},
    },
    # trailing purely-numeric segment: NOT treated as a real extension, so
    # a fresh .NNN segment is appended after it rather than modifying it.
    "file.123": {
        5: {0: "file.123.0", 4: "file.123.4"},
    },
    # bare trailing dot: also not a real extension -> genuine double dot.
    "file.": {
        5: {0: "file..0", 4: "file..4"},
    },
}


@pytest.mark.parametrize(
    "base,nnodes,node,expected",
    [
        (base, nnodes, node, expected)
        for base, by_nnodes in GOLDEN.items()
        for nnodes, by_node in by_nnodes.items()
        for node, expected in by_node.items()
    ],
)
def test_golden_values(base, nnodes, node, expected):
    assert insert_node_filename(base, node, nnodes) == expected


@pytest.mark.parametrize(
    "nnodes,expected_width",
    [(1, 1), (5, 1), (10, 1), (11, 2), (23, 2), (100, 2), (101, 3), (151, 3), (1000, 3), (1001, 4)],
)
def test_digit_width(nnodes, expected_width):
    assert digit_width(nnodes) == expected_width


def test_digit_width_hard_error_beyond_5_digits():
    with pytest.raises(ValueError):
        digit_width(100_000)

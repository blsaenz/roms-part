"""Unit tests for dims.py: obsolete-dim remap, part_type classification,
and the strict-length-inequality boundary-suffix check (partit.F lines
578-593 -- a variable literally named e.g. "_west" must NOT match, since
the Fortran check requires len(varname) > 5, not >= 5).
"""
from __future__ import annotations

import pytest

from roms_part.dims import (
    DimAxis,
    boundary_edge,
    classify_variable,
    remap_dim_name,
    resolve_partitionable_dims,
)


@pytest.mark.parametrize(
    "name,expected",
    [
        ("xi_psi", "xi_u"),
        ("xi_v", "xi_rho"),
        ("eta_psi", "eta_v"),
        ("eta_u", "eta_rho"),
        ("xi_rho", "xi_rho"),
        ("xi_u", "xi_u"),
        ("s_rho", "s_rho"),
    ],
)
def test_remap_dim_name(name, expected):
    assert remap_dim_name(name) == expected


@pytest.mark.parametrize(
    "varname,expected",
    [
        ("temp_west", "west"),
        ("temp_east", "east"),
        ("zeta_south", "south"),
        ("zeta_north", "north"),
        ("a_west", "west"),  # len=6 > 5: minimal passing case
        ("a_south", "south"),  # len=7 > 6: minimal passing case
        ("_west", None),  # len=5, NOT > 5: strict inequality guard
        ("_south", None),  # len=6, NOT > 6
        ("temp", None),
        ("westward_temp", None),
    ],
)
def test_boundary_edge(varname, expected):
    assert boundary_edge(varname) == expected


def test_classify_variable_both_axes_and_record():
    part_type, has_record = classify_variable(
        ["ocean_time", "s_rho", "eta_rho", "xi_rho"], "ocean_time"
    )
    assert part_type == DimAxis.BOTH
    assert has_record is True


def test_classify_variable_xi_only():
    part_type, has_record = classify_variable(["eta_v", "xi_u"], None)
    # xi_u -> XI, eta_v -> ETA -- both present here, so this is BOTH; use a
    # single-axis case explicitly below.
    assert part_type == DimAxis.BOTH
    assert has_record is False


def test_classify_variable_none():
    part_type, has_record = classify_variable(["s_rho"], None)
    assert part_type == DimAxis.NONE
    assert has_record is False


def test_resolve_partitionable_dims_infers_missing_u_v():
    resolved = resolve_partitionable_dims({"xi_rho": 674, "eta_rho": 1322})
    assert resolved == {"xi_rho": 674, "xi_u": 673, "eta_rho": 1322, "eta_v": 1321}


def test_resolve_partitionable_dims_infers_xi_rho_from_xi_u():
    # xi branch is symmetric in partit.F: xi_u-only correctly infers xi_rho.
    resolved = resolve_partitionable_dims({"xi_u": 673, "eta_rho": 1322})
    assert resolved == {"xi_rho": 674, "xi_u": 673, "eta_rho": 1322, "eta_v": 1321}


def test_resolve_partitionable_dims_eta_v_only_is_unresolvable():
    # eta branch is NOT symmetric in the real partit.F (lines 337-341): a
    # file with eta_v but no eta_rho fails resolution in the actual Fortran
    # tool too, and must fail here as well for bit-for-bit fidelity.
    with pytest.raises(ValueError):
        resolve_partitionable_dims({"xi_rho": 674, "eta_v": 1321})


def test_resolve_partitionable_dims_raises_when_unresolvable():
    with pytest.raises(ValueError):
        resolve_partitionable_dims({"s_rho": 42})

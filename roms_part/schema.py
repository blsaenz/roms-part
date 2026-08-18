"""Read-only inspection of a source netCDF file's schema. No writing here."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import netCDF4


@dataclass(frozen=True)
class VarInfo:
    name: str
    dtype: object  # numpy dtype
    dim_names: tuple[str, ...]  # raw dim names, as in the source file
    fill_value: object  # None if no _FillValue


@dataclass(frozen=True)
class FileSchema:
    dim_sizes: dict[str, int]  # raw sizes as read from the source file
    unlimited_dim: Optional[str]
    variables: list[VarInfo]
    global_attrs: dict[str, object]


class AlreadyPartitionedError(ValueError):
    """Raised when the source file already carries a 'partition' global
    attribute (partit.F lines 216-223: warn and skip, don't abort)."""


def inspect_source(path: str, allow_partition: bool = False) -> FileSchema:
    with netCDF4.Dataset(path, mode="r") as ds:
        if not allow_partition and "partition" in ds.ncattrs():
            raise AlreadyPartitionedError(
                f"'{path}' already has a 'partition' global attribute -- "
                "it is already a partitioned file and cannot be partitioned "
                "further (matches partit.F's own already-partitioned check)."
            )

        # len(dim) returns the CURRENT size even for an unlimited dimension;
        # unlimited_dim (below) is what distinguishes it from a fixed dim.
        dim_sizes = {name: len(dim) for name, dim in ds.dimensions.items()}
        unlimited_dim = next((name for name, dim in ds.dimensions.items() if dim.isunlimited()), None)

        variables = []
        for name, var in ds.variables.items():
            fill_value = None
            if "_FillValue" in var.ncattrs():
                fill_value = var.getncattr("_FillValue")
            variables.append(
                VarInfo(
                    name=name,
                    dtype=var.dtype,
                    dim_names=tuple(var.dimensions),
                    fill_value=fill_value,
                )
            )

        global_attrs = {a: ds.getncattr(a) for a in ds.ncattrs()}

    return FileSchema(
        dim_sizes=dim_sizes,
        unlimited_dim=unlimited_dim,
        variables=variables,
        global_attrs=global_attrs,
    )

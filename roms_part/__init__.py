"""roms-part: a fast, parallel Python replacement for UCLA-ROMS's
Tools-Roms/partit.F netCDF domain-decomposition tool.
"""
from .decomposition import NodeInfo, mpi_setup, node_dim_extent, node_read_range
from .job import PartitionJob, VarPlan, build_job
from .naming import insert_node_filename
from .schema import AlreadyPartitionedError, FileSchema, VarInfo, inspect_source

try:
    from importlib.metadata import version as _version

    __version__ = _version("roms-part")
except Exception:  # pragma: no cover
    __version__ = "0.0.0"

__all__ = [
    "NodeInfo",
    "mpi_setup",
    "node_dim_extent",
    "node_read_range",
    "PartitionJob",
    "VarPlan",
    "build_job",
    "insert_node_filename",
    "AlreadyPartitionedError",
    "FileSchema",
    "VarInfo",
    "inspect_source",
]

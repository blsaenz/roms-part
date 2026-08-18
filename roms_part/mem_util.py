"""Small, dependency-free helpers for estimating and reporting memory
usage. Deliberately avoids adding psutil as a dependency -- POSIX
sysconf covers the platforms this tool actually targets (macOS, Linux).
"""
from __future__ import annotations

import os

# Warn when the estimated peak exceeds this fraction of detected total
# system memory. Conservative on purpose: these are long-running batch
# jobs sharing a machine with other work (the OS, other processes), not
# a dedicated single-purpose box.
WARN_FRACTION = 0.5


def total_system_memory_bytes() -> int | None:
    """Best-effort total physical memory in bytes, or None if it can't be
    determined on this platform (e.g. Windows, or a sysconf without these
    names)."""
    try:
        return os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")
    except (ValueError, OSError, AttributeError):
        return None


def format_bytes(n: int) -> str:
    value = float(n)
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if value < 1024 or unit == "TB":
            return f"{value:.1f} {unit}"
        value /= 1024
    return f"{value:.1f} TB"  # pragma: no cover


def report_memory_estimate(peak_bytes: int, n_workers: int, context: str) -> None:
    """Prints an informational estimate, and a warning if it looks likely
    to exceed a safe fraction of total system memory."""
    total = total_system_memory_bytes()
    print(
        f"Estimated peak memory ({context}, {n_workers} workers): "
        f"~{format_bytes(peak_bytes)}"
    )
    if total is None:
        print("  (could not detect total system memory to compare against)")
        return
    frac = peak_bytes / total
    if frac > WARN_FRACTION:
        print(
            f"  ### WARNING: this is ~{frac * 100:.0f}% of this machine's "
            f"{format_bytes(total)} total RAM -- consider a lower "
            f"--n-workers, or --time-batch-size for large record dimensions, "
            f"if you see swapping or an OOM kill."
        )

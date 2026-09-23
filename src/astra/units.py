"""Parsing helpers for the unit-suffixed strings emitted by nvidia-smi and sysfs."""

from __future__ import annotations

import re

KIB = 1024
MIB = 1024 * KIB
GIB = 1024 * MIB

_NUMBER = re.compile(r"^\s*(-?\d+(?:\.\d+)?)\s*([A-Za-z/%]*)")
_BYTE_UNITS = {"": 1, "b": 1, "kib": KIB, "kb": KIB, "mib": MIB, "mb": MIB, "gib": GIB, "gb": GIB}

# PCIe signalling rate (GT/s) -> generation.
_GT_TO_GEN = {2.5: 1, 5.0: 2, 8.0: 3, 16.0: 4, 32.0: 5, 64.0: 6}
# Usable per-lane throughput in GB/s after line encoding (8b/10b for Gen1/2, 128b/130b after).
GEN_LANE_GBPS = {1: 0.25, 2: 0.5, 3: 0.985, 4: 1.969, 5: 3.938, 6: 7.563}


def is_missing(text: str | None) -> bool:
    """nvidia-smi reports unsupported fields as N/A, [N/A], or deprecation notices."""
    if text is None:
        return True
    t = text.strip()
    return t == "" or "N/A" in t or "deprecated" in t.lower() or "Not Supported" in t


def parse_number(text: str | None) -> float | None:
    if is_missing(text):
        return None
    assert text is not None
    m = _NUMBER.match(text)
    return float(m.group(1)) if m else None


def parse_int(text: str | None) -> int | None:
    value = parse_number(text)
    return int(value) if value is not None else None


def parse_bytes(text: str | None) -> int | None:
    """Parse '8192 MiB' / '6 GiB' / '1024' into bytes."""
    if is_missing(text):
        return None
    assert text is not None
    m = _NUMBER.match(text)
    if not m:
        return None
    unit = m.group(2).lower()
    if unit not in _BYTE_UNITS:
        return None
    return int(float(m.group(1)) * _BYTE_UNITS[unit])


def parse_link_width(text: str | None) -> int | None:
    """'16x', 'x4' and '4' all mean a 4/16-lane link."""
    if is_missing(text):
        return None
    assert text is not None
    digits = re.search(r"\d+", text)
    return int(digits.group(0)) if digits else None


def gts_to_gen(text: str | None) -> int | None:
    """Convert a sysfs link speed such as '8.0 GT/s PCIe' into a PCIe generation."""
    speed = parse_number(text)
    if speed is None:
        return None
    return _GT_TO_GEN.get(speed)


def link_bandwidth_gbps(gen: int | None, width: int | None) -> float | None:
    """Theoretical one-direction payload bandwidth of a link in GB/s."""
    if gen is None or width is None or gen not in GEN_LANE_GBPS:
        return None
    return GEN_LANE_GBPS[gen] * width


def fmt_bytes(n: int | float | None) -> str:
    if n is None:
        return "n/a"
    value = float(n)
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if abs(value) < 1024 or unit == "TiB":
            return f"{value:.0f} {unit}" if unit in ("B", "KiB") else f"{value:.2f} {unit}"
        value /= 1024
    return f"{value:.2f} TiB"

"""GPU-to-GPU topology from ``nvidia-smi topo -m`` (works on Windows and Linux).

sysfs gives the full PCIe path on Linux only. NVIDIA's own topology matrix tells,
on any OS, how each pair of GPUs is connected, which is enough to tell whether they
sit behind a shared PCIe switch (the ASTRA chassis) or connect through the CPU.
"""

from __future__ import annotations

import re

from astra.errors import CommandError
from astra.hardware.runner import CommandRunner

_ANSI = re.compile(r"\x1b\[[0-9;]*m")
_CELL = re.compile(r"^(X|SYS|NODE|PHB|PXB|PIX|NV\d+)$")

# What each link type means for ASTRA.
LINK_MEANING = {
    "PIX": "same PCIe switch",
    "PXB": "PCIe switches, not via the CPU",
    "PHB": "through the CPU's PCIe host bridge",
    "NODE": "through the CPU (between host bridges)",
    "SYS": "through the CPU interconnect (different sockets)",
}
SWITCH_LINKS = frozenset({"PIX", "PXB"})


def parse_topology(text: str) -> dict[tuple[int, int], str]:
    """{(i, j): link} for i < j from the ``nvidia-smi topo -m`` matrix."""
    rows: dict[int, list[str]] = {}
    for raw in _ANSI.sub("", text).splitlines():
        parts = raw.split()
        if len(parts) >= 2 and re.fullmatch(r"GPU\d+", parts[0]):
            cells = []
            for cell in parts[1:]:
                if not _CELL.match(cell):
                    break  # CPU affinity / NUMA columns follow the GPU columns
                cells.append(cell)
            rows[int(parts[0][3:])] = cells
    links: dict[tuple[int, int], str] = {}
    for i, cells in rows.items():
        for j, cell in enumerate(cells):
            if i < j and cell != "X":
                links[(i, j)] = cell
    return links


_P2P_CELL = re.compile(r"^(X|OK|NS|CNS|GNS|TNS|DR|U)$")
P2P_MEANING = {
    "OK": "peer-to-peer supported",
    "NS": "not supported",
    "CNS": "chipset does not support it",
    "GNS": "GPU does not support it (GeForce driver policy)",
    "TNS": "topology does not support it",
    "DR": "disabled by registry key",
    "U": "unknown",
}


def parse_p2p(text: str) -> dict[tuple[int, int], str]:
    """{(i, j): status} for i < j from ``nvidia-smi topo -p2p r`` (read capability)."""
    status: dict[tuple[int, int], str] = {}
    for raw in _ANSI.sub("", text).splitlines():
        parts = raw.split()
        if len(parts) >= 2 and re.fullmatch(r"GPU\d+", parts[0]):
            i = int(parts[0][3:])
            cells = [c for c in parts[1:] if _P2P_CELL.match(c)]
            for j, cell in enumerate(cells):
                if i < j and cell != "X":
                    status[(i, j)] = cell
    return status


def query_p2p(runner: CommandRunner) -> dict[tuple[int, int], str]:
    try:
        return parse_p2p(runner.run(["nvidia-smi", "topo", "-p2p", "r"]))
    except CommandError:
        return {}


def describe(link: str) -> str:
    if link.startswith("NV"):
        return f"NVLink ×{link[2:]}"
    return LINK_MEANING.get(link, link)


def query(runner: CommandRunner) -> dict[tuple[int, int], str]:
    try:
        return parse_topology(runner.run(["nvidia-smi", "topo", "-m"]))
    except CommandError:
        return {}  # older drivers / restricted environments: topology is optional

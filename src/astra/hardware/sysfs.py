"""Linux PCIe topology from sysfs.

nvidia-smi only reports the link between a GPU and the port directly above it —
behind the ASTRA packet switch that is the switch's x16 downstream port, which
hides the real bottleneck: the x4 OCuLink uplink. Walking the sysfs device path
from the root complex down to the GPU exposes every hop, so validation can check
the uplink and detect the PLX switch.

Filesystem access goes through :class:`SysfsReader` so the logic is testable on
any OS (sysfs names contain ':' which Windows filesystems reject).
"""

from __future__ import annotations

import os
import re
from pathlib import Path, PurePosixPath
from typing import Protocol

from astra.hardware.models import AerCounters, GpuPath, PcieLink
from astra.units import gts_to_gen, parse_int

BDF_RE = re.compile(r"^[0-9a-f]{4}:[0-9a-f]{2}:[0-9a-f]{2}\.[0-7]$")


class SysfsReader(Protocol):
    def available(self) -> bool: ...

    def device_path(self, bdf: str) -> PurePosixPath | None:
        """Canonical sysfs path, e.g. /sys/devices/pci0000:00/0000:00:1b.4/0000:02:00.0."""

    def read(self, path: PurePosixPath) -> str | None: ...


class LocalSysfs:
    def __init__(self, root: str | Path = "/sys") -> None:
        self.root = Path(root)

    def available(self) -> bool:
        return (self.root / "bus" / "pci" / "devices").is_dir()

    def device_path(self, bdf: str) -> PurePosixPath | None:
        link = self.root / "bus" / "pci" / "devices" / bdf
        if not link.exists():
            return None
        return PurePosixPath(Path(os.path.realpath(link)).as_posix())

    def read(self, path: PurePosixPath) -> str | None:
        try:
            return Path(path).read_text(encoding="ascii", errors="replace").strip()
        except OSError:
            return None


def read_link(fs: SysfsReader, device: PurePosixPath) -> PcieLink:
    return PcieLink(
        bdf=device.name,
        vendor_id=fs.read(device / "vendor"),
        device_id=fs.read(device / "device"),
        class_code=fs.read(device / "class"),
        current_gen=gts_to_gen(fs.read(device / "current_link_speed")),
        current_width=parse_int(fs.read(device / "current_link_width")),
        max_gen=gts_to_gen(fs.read(device / "max_link_speed")),
        max_width=parse_int(fs.read(device / "max_link_width")),
    )


def _aer_total(text: str | None, total_key: str) -> int:
    if not text:
        return 0
    values: dict[str, int] = {}
    for line in text.splitlines():
        parts = line.split()
        if len(parts) == 2 and parts[1].isdigit():
            values[parts[0]] = int(parts[1])
    if total_key in values:
        return values[total_key]
    return sum(v for k, v in values.items() if not k.startswith("TOTAL"))


def read_aer(fs: SysfsReader, device: PurePosixPath) -> AerCounters:
    return AerCounters(
        correctable=_aer_total(fs.read(device / "aer_dev_correctable"), "TOTAL_ERR_COR"),
        nonfatal=_aer_total(fs.read(device / "aer_dev_nonfatal"), "TOTAL_ERR_NONFATAL"),
        fatal=_aer_total(fs.read(device / "aer_dev_fatal"), "TOTAL_ERR_FATAL"),
    )


def device_chain(device: PurePosixPath) -> list[PurePosixPath]:
    """All PCI devices from the first hop below the root complex down to ``device``."""
    chain: list[PurePosixPath] = []
    current = device
    while BDF_RE.match(current.name):
        chain.append(current)
        current = current.parent
    chain.reverse()
    return chain


def gpu_path(bdf: str, fs: SysfsReader) -> GpuPath | None:
    """Describe the PCIe path to the GPU at ``bdf`` (e.g. '0000:05:00.0'), or None."""
    device = fs.device_path(bdf.lower())
    if device is None:
        return None
    chain = device_chain(device)
    aer = AerCounters()
    for hop in chain:
        c = read_aer(fs, hop)
        aer = AerCounters(
            aer.correctable + c.correctable, aer.nonfatal + c.nonfatal, aer.fatal + c.fatal
        )
    return GpuPath(gpu_bdf=bdf.lower(), chain=tuple(read_link(fs, hop) for hop in chain), aer=aer)

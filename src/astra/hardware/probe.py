"""Assemble a full node inventory from the driver and (on Linux) sysfs."""

from __future__ import annotations

import platform
import time

from astra.hardware import nvsmi, nvtopo, sysfs
from astra.hardware.models import GpuPath, Inventory
from astra.hardware.runner import CommandRunner, SubprocessRunner


def platform_label() -> str:
    """'Windows 11' too: Python reports release '10' for Windows 11 (build >= 22000)."""
    system, release = platform.system(), platform.release()
    if system == "Windows" and release == "10":
        build = platform.version().split(".")[-1]
        if build.isdigit() and int(build) >= 22000:
            release = "11"
    return f"{system} {release}"


def probe(
    runner: CommandRunner | None = None,
    fs: sysfs.SysfsReader | None = None,
    with_topology: bool = True,
) -> Inventory:
    runner = runner or SubprocessRunner()
    fs = fs or sysfs.LocalSysfs()
    report = nvsmi.query(runner)

    paths: dict[str, GpuPath] = {}
    if with_topology and fs.available():
        for gpu in report.gpus:
            path = sysfs.gpu_path(gpu.short_bdf, fs)
            if path is not None:
                paths[gpu.uuid] = path

    links = nvtopo.query(runner) if with_topology and len(report.gpus) > 1 else {}

    return Inventory(
        gpu_links=links,
        gpus=report.gpus,
        driver_version=report.driver_version,
        cuda_version=report.cuda_version,
        platform=platform_label(),
        paths=paths,
        timestamp=time.time(),
    )

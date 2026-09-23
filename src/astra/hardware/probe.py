"""Assemble a full node inventory from the driver and (on Linux) sysfs."""

from __future__ import annotations

import platform
import time

from astra.hardware import nvsmi, sysfs
from astra.hardware.models import GpuPath, Inventory
from astra.hardware.runner import CommandRunner, SubprocessRunner


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

    return Inventory(
        gpus=report.gpus,
        driver_version=report.driver_version,
        cuda_version=report.cuda_version,
        platform=f"{platform.system()} {platform.release()}",
        paths=paths,
        timestamp=time.time(),
    )

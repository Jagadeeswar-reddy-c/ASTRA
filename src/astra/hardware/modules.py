"""ASTRA Stack bricks (ADR-0014): which installed GPU belongs to which ``[[module]]``.

Each ``[[module]]`` lists its GPUs by UUID, PCI bus id, or a product-name substring.
Exact identifiers are matched first, then names in PCI bus order, and every GPU is
assigned at most once, so ``gpus = ["RTX 3090"]`` in four modules claims four 3090s.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from astra.config import ModuleConfig
from astra.hardware.models import GpuInfo


@dataclass(frozen=True)
class ModuleAssignment:
    module: ModuleConfig
    gpus: tuple[GpuInfo, ...]
    missing: tuple[str, ...]  # selectors that matched no installed GPU


def _bus(bus_id: str) -> str:
    """'00000000:05:00.0' and '0000:05:00.0' and '05:00.0' compare equal."""
    return bus_id.lower().split(":", 1)[-1] if bus_id.count(":") == 2 else bus_id.lower()


def _exact(selector: str, gpu: GpuInfo) -> bool:
    s = selector.strip().lower()
    return s == gpu.uuid.lower() or (":" in s and _bus(s) == _bus(gpu.bus_id))


def _by_name(selector: str, gpu: GpuInfo) -> bool:
    return selector.strip().lower() in gpu.name.lower()


def assign(
    gpus: Sequence[GpuInfo], modules: Sequence[ModuleConfig]
) -> tuple[list[ModuleAssignment], dict[str, str]]:
    """Per-module GPUs and missing selectors, plus {uuid: module name}."""
    ordered = sorted(gpus, key=lambda g: g.bus_id)
    owner: dict[str, str] = {}
    picked: dict[tuple[int, int], GpuInfo] = {}
    for exact in (True, False):
        for mi, m in enumerate(modules):
            for si, selector in enumerate(m.gpus):
                if (mi, si) in picked:
                    continue
                match = _exact if exact else _by_name
                gpu = next((g for g in ordered if g.uuid not in owner and match(selector, g)), None)
                if gpu is not None:
                    picked[(mi, si)] = gpu
                    owner[gpu.uuid] = m.name
    out = []
    for mi, m in enumerate(modules):
        found = tuple(picked[(mi, si)] for si in range(len(m.gpus)) if (mi, si) in picked)
        missing = tuple(s for si, s in enumerate(m.gpus) if (mi, si) not in picked)
        out.append(ModuleAssignment(m, found, missing))
    return out, owner

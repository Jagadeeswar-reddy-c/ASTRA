"""Ray resource model for a heterogeneous GPU node.

``@ray.remote(num_gpus=1)`` from the concept document schedules onto *any* free
GPU. ASTRA chassis mix arbitrary cards (a 2 GB GT 1030 next to a 12 GB RTX 3060),
so tasks often need to say *which kind* of GPU they need. This module advertises
extra custom resources so tasks can request them:

    vram_mib        total MiB across the pool (for memory-aware packing)
    gpu_<arch>      one unit per GPU of that architecture, e.g. gpu_turing, gpu_ada_lovelace
    gpu_vram_<n>g   one unit per GPU with at least <n> GiB, e.g. gpu_vram_8g
    gpu_cc_<nn>     one unit per GPU with compute capability >= n.n, e.g. gpu_cc_75
    nvenc           one unit per GPU with a hardware video encoder

Example::

    from astra.orchestration.ray_pool import start_pool
    ray = start_pool()

    @ray.remote(num_gpus=1, resources={"gpu_ampere": 1})
    def transcode(chunk): ...

Ray is an optional dependency: ``pip install astra-node[ray]``.
"""

from __future__ import annotations

import os
from typing import Any

from astra.hardware.compat import capabilities
from astra.hardware.models import GpuInfo
from astra.units import GIB, MIB


def ray_resources(gpus: list[GpuInfo] | tuple[GpuInfo, ...]) -> dict[str, float]:
    resources: dict[str, float] = {}
    total_mib = 0

    def bump(key: str) -> None:
        resources[key] = resources.get(key, 0) + 1

    for g in gpus:
        caps = capabilities(g)
        if caps.architecture != "unknown":
            bump("gpu_" + caps.architecture.lower().replace(" ", "_"))
        cc = caps.compute_capability
        if cc is not None:
            for threshold in (61, 70, 75, 80, 86, 89, 120):
                if round(cc * 10) >= threshold:
                    bump(f"gpu_cc_{threshold}")
        if caps.nvenc:
            bump("nvenc")
        if g.memory_total_bytes:
            total_mib += g.memory_total_bytes // MIB
            gib = round(g.memory_total_bytes / GIB)
            for threshold in (2, 4, 6, 8, 12, 16, 24, 32):
                if gib >= threshold:
                    bump(f"gpu_vram_{threshold}g")
    if total_mib:
        resources["vram_mib"] = float(total_mib)
    return resources


def start_pool(gpus: list[GpuInfo] | None = None, **ray_init_kwargs: Any) -> Any:
    """Start a local Ray runtime that exposes the ASTRA GPUs with typed resources."""
    try:
        import ray
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise RuntimeError("Ray is not installed: pip install 'astra-node[ray]'") from exc

    if gpus is None:
        from astra.hardware.probe import probe

        gpus = list(probe(with_topology=False).gpus)
    os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
    ray.init(num_gpus=len(gpus), resources=ray_resources(gpus), **ray_init_kwargs)
    return ray

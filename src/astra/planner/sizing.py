"""Reverse planning: which GPUs (ASTRA Stack bricks) to buy for a model and a speed.

For every GPU model in the catalog (each VRAM variant), find the smallest number of
identical cards, optionally added to GPUs you already have, whose plan fits with
margin and reaches the speed target. Used by ``astra size`` (CR-006, S4).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from astra.errors import PlanningError
from astra.hardware.gpu_specs import SPECS, GpuSpec, arch_info, lookup
from astra.planner.models import ModelProfile
from astra.planner.split import SPEED_FILL_LIMIT, Plan, plan, select_gpus

MIN_SIZING_CC = 7.5  # Turing and newer: current drivers and CUDA 13 (see supported-gpus.md)


@dataclass(frozen=True)
class SizingOption:
    gpu: str  # catalog model, e.g. "RTX 3090"
    vram_mib: int
    count: int  # new cards to add
    plan: Plan

    @property
    def tokens_per_s(self) -> float | None:
        return self.plan.est_decode_tokens_per_s

    @property
    def rated_power_w(self) -> int:
        total = 0
        for p in self.plan.placements:
            spec = lookup(p.device.name.replace("NVIDIA GeForce ", ""), p.device.total_bytes)
            total += spec.board_power_w if spec else 0
        return total

    @property
    def pool_vram_bytes(self) -> int:
        return sum(p.device.total_bytes for p in self.plan.placements)

    def to_dict(self) -> dict[str, Any]:
        return {
            "gpu": self.gpu,
            "vram_mib": self.vram_mib,
            "count": self.count,
            "gpus_used": len(self.plan.placements),
            "est_decode_tokens_per_s": self.tokens_per_s,
            "pool_vram_bytes": self.pool_vram_bytes,
            "rated_power_w": self.rated_power_w,
        }


def candidates(include_legacy: bool = False) -> list[GpuSpec]:
    out = []
    for s in SPECS:
        arch = arch_info(s.architecture)
        if arch is None or (not include_legacy and arch.compute_capability < MIN_SIZING_CC):
            continue
        out.append(s)
    return out


def size_stack(
    model: ModelProfile,
    context: int,
    min_tps: float,
    gpu_memory_utilization: float,
    reserve_mib: int,
    have: str | None = None,
    max_count: int = 8,
    include_legacy: bool = False,
    only: str | None = None,
) -> list[SizingOption]:
    """Cheapest-first options (fewest cards, then lowest power) that meet the target."""
    from astra.planner.split import budgets_from_gpus
    from astra.pool import simulated

    if min_tps <= 0:
        raise PlanningError("--min-tps must be > 0")
    options: list[SizingOption] = []
    for spec in candidates(include_legacy):
        if only and only.lower() not in spec.model.lower():
            continue
        for count in range(1, max_count + 1):
            items = [have] if have else []
            items.append(f"{count}x {spec.model}:{spec.vram_mib}")
            gpus, _ = simulated(", ".join(items))
            budgets, _ = budgets_from_gpus(gpus, gpu_memory_utilization, reserve_mib)
            try:
                if have:  # mixed pool: let the planner choose which cards to use
                    result = select_gpus(
                        model, budgets, "llamacpp", context, gpu_memory_utilization
                    )
                else:  # identical cards: every subset of a size is the same
                    result = plan(
                        model,
                        budgets,
                        "llamacpp",
                        context,
                        gpu_memory_utilization,
                        objective="speed",
                        hint_single_gpu=False,
                    )
            except PlanningError:
                break
            worst = max(p.model_bytes / p.device.usable_bytes for p in result.placements)
            tps = result.est_decode_tokens_per_s or 0.0
            if result.fits and worst <= SPEED_FILL_LIMIT and tps >= min_tps:
                options.append(SizingOption(spec.model, spec.vram_mib, count, result))
                break
            if result.fits and tps < min_tps and worst <= SPEED_FILL_LIMIT:
                break  # more cards of this kind add memory, not speed (LIM-02)
    options.sort(key=lambda o: (o.count, o.rated_power_w, -(o.tokens_per_s or 0)))
    return options

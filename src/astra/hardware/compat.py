"""Compatibility and power analysis for an arbitrary mix of NVIDIA GPUs (ADR-0009).

Any GT / GTX / RTX card may be installed, but mixing generations has hard
constraints that are easy to miss:

* **One driver serves every GPU.** Kepler needs ≤ R470 and Maxwell/Pascal/Volta
  need ≤ R580, while new cards need a recent branch. The installed branch must
  lie inside every card's supported window.
* **Engines must be built for every compute capability** (e.g. ``61;75;86``), and
  CUDA 13 no longer targets Maxwell/Pascal/Volta, so such mixes need CUDA 12.x
  builds of llama.cpp.
* **vLLM needs compute capability ≥ 7.0**; llama.cpp runs on older cards.
* **The chassis PSU must carry the actual cards**, not the reference build's.

``analyse()`` turns GPUs (+ driver version + config) into per-GPU capabilities,
findings (fail / warn / info) and a power budget, used by ``astra compat``, the
link gate (L11, L12), the planner and the exporter.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass, field, replace
from typing import Any

from astra.config import ChassisConfig, ModuleConfig
from astra.hardware.gpu_specs import ArchInfo, GpuSpec, arch_info, lookup
from astra.hardware.models import GpuInfo, GpuPath

# Minimum compute capability per engine. llama.cpp's CUDA backend still builds for
# Maxwell, but ASTRA only qualifies it from Pascal (6.1) upward.
ENGINE_MIN_CC = {"llamacpp": 5.0, "vllm": 7.0}
LLAMACPP_QUALIFIED_CC = 6.1
SMALL_VRAM_MIB = 4096
PSU_SIZES = (450, 550, 650, 750, 850, 1000, 1200, 1600)


@dataclass(frozen=True)
class GpuCaps:
    gpu: GpuInfo
    spec: GpuSpec | None
    arch: ArchInfo | None

    @property
    def architecture(self) -> str:
        if self.arch:
            return self.arch.name
        return self.gpu.architecture or (self.spec.architecture if self.spec else "unknown")

    @property
    def compute_capability(self) -> float | None:
        return self.arch.compute_capability if self.arch else None

    @property
    def mem_bandwidth_gbps(self) -> float | None:
        return self.spec.mem_bandwidth_gbps if self.spec else None

    @property
    def board_power_w(self) -> float | None:
        """Reference board power; else the driver's max limit; else the enforced limit."""
        if self.spec:
            return float(self.spec.board_power_w)
        return self.gpu.power_max_limit_w or self.gpu.power_limit_w

    @property
    def nvenc(self) -> bool | None:
        return self.spec.nvenc if self.spec else None

    @property
    def engines(self) -> tuple[str, ...]:
        cc = self.compute_capability
        if cc is None:
            return ("llamacpp",)  # unknown: allow the permissive engine, flagged separately
        return tuple(e for e, need in ENGINE_MIN_CC.items() if cc >= need)

    @property
    def sm(self) -> str | None:
        cc = self.compute_capability
        return None if cc is None else f"{round(cc * 10)}"


def capabilities(gpu: GpuInfo) -> GpuCaps:
    spec = lookup(gpu.name, gpu.memory_total_bytes)
    arch = arch_info(gpu.architecture) or (arch_info(spec.architecture) if spec else None)
    return GpuCaps(gpu, spec, arch)


@dataclass(frozen=True)
class Finding:
    severity: str  # "fail" | "warn" | "info"
    code: str
    message: str


@dataclass(frozen=True)
class PowerBudget:
    psu_watts: int
    gpu_watts: float
    overhead_watts: int
    sustained_watts: float
    peak_watts: float
    sustained_limit_watts: float
    unknown_gpus: tuple[str, ...]
    slot_powered_watts: float  # drawn through the backplane's slot feed

    @property
    def ok(self) -> bool:
        return (
            self.sustained_watts <= self.sustained_limit_watts and self.peak_watts <= self.psu_watts
        )

    @property
    def recommended_psu_watts(self) -> int:
        need = max(
            self.sustained_watts / (self.sustained_limit_watts / self.psu_watts), self.peak_watts
        )
        return next((s for s in PSU_SIZES if s >= need), int(math.ceil(need / 100) * 100))


@dataclass(frozen=True)
class ModuleReport:
    """One ASTRA Stack brick: its GPUs and its own PSU budget (ADR-0014)."""

    config: ModuleConfig
    members: tuple[GpuCaps, ...]
    missing: tuple[str, ...]
    power: PowerBudget

    @property
    def name(self) -> str:
        return self.config.name

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "gpus": [c.gpu.uuid for c in self.members],
            "missing": list(self.missing),
            "psu_watts": self.power.psu_watts,
            "sustained_watts": self.power.sustained_watts,
            "peak_watts": self.power.peak_watts,
            "sustained_limit_watts": self.power.sustained_limit_watts,
            "ok": self.power.ok,
            "recommended_psu_watts": self.power.recommended_psu_watts,
        }


@dataclass(frozen=True)
class CompatReport:
    driver_version: str | None
    caps: tuple[GpuCaps, ...]
    chassis: tuple[GpuCaps, ...]
    chassis_basis: str
    findings: tuple[Finding, ...]
    power: PowerBudget
    cuda_architectures: str
    required_driver_window: tuple[int, int | None]  # (min branch, max branch or None)
    notes: tuple[str, ...] = field(default_factory=tuple)
    modules: tuple[ModuleReport, ...] = ()  # set when [[module]] bricks are configured

    def module_of(self, uuid: str) -> str | None:
        return next(
            (m.name for m in self.modules if any(c.gpu.uuid == uuid for c in m.members)), None
        )

    @property
    def worst(self) -> str:
        sev = {f.severity for f in self.findings}
        return "fail" if "fail" in sev else "warn" if "warn" in sev else "ok"

    def to_dict(self) -> dict[str, Any]:
        return {
            "driver_version": self.driver_version,
            "status": self.worst,
            "cuda_architectures": self.cuda_architectures,
            "required_driver_window": list(self.required_driver_window),
            "chassis_basis": self.chassis_basis,
            "gpus": [
                {
                    "index": c.gpu.index,
                    "uuid": c.gpu.uuid,
                    "name": c.gpu.name,
                    "known_model": c.spec.model if c.spec else None,
                    "architecture": c.architecture,
                    "compute_capability": c.compute_capability,
                    "vram_bytes": c.gpu.memory_total_bytes,
                    "mem_bandwidth_gbps": c.mem_bandwidth_gbps,
                    "board_power_w": c.board_power_w,
                    "nvenc": c.nvenc,
                    "engines": list(c.engines),
                    "last_driver_branch": c.arch.last_driver_branch if c.arch else None,
                    "in_chassis": c in self.chassis,
                }
                for c in self.caps
            ],
            "power": {
                "psu_watts": self.power.psu_watts,
                "gpu_watts": self.power.gpu_watts,
                "sustained_watts": self.power.sustained_watts,
                "peak_watts": self.power.peak_watts,
                "sustained_limit_watts": self.power.sustained_limit_watts,
                "ok": self.power.ok,
                "recommended_psu_watts": self.power.recommended_psu_watts,
                "unknown_gpus": list(self.power.unknown_gpus),
            },
            "findings": [f.__dict__ for f in self.findings],
            "modules": [m.to_dict() for m in self.modules],
        }


def driver_branch(version: str | None) -> int | None:
    if not version:
        return None
    head = version.strip().split(".")[0]
    return int(head) if head.isdigit() else None


def chassis_members(
    caps: Sequence[GpuCaps], paths: dict[str, GpuPath] | None, switch_vendor_ids: tuple[str, ...]
) -> tuple[tuple[GpuCaps, ...], str]:
    """Which GPUs the chassis PSU feeds: those behind the switch; else non-display GPUs."""
    if paths:
        behind = tuple(
            c for c in caps if c.gpu.uuid in paths and paths[c.gpu.uuid].switches(switch_vendor_ids)
        )
        if behind:
            return behind, "GPUs behind the PCIe switch"
    headless = tuple(c for c in caps if c.gpu.display_active is not True)
    if headless and len(headless) < len(caps):
        return headless, "GPUs without an active display (host display GPU excluded)"
    return tuple(caps), "all GPUs (chassis membership unknown)"


def power_budget(members: Sequence[GpuCaps], chassis: ChassisConfig) -> PowerBudget:
    watts, unknown, slot = 0.0, [], 0.0
    for c in members:
        w = c.board_power_w
        if w is None:
            unknown.append(c.gpu.name)
            w = 75.0  # assume at least slot power; flagged as unknown
        watts += w
        slot += min(w, 75.0)
    sustained = watts + chassis.overhead_watts
    return PowerBudget(
        psu_watts=chassis.psu_watts,
        gpu_watts=watts,
        overhead_watts=chassis.overhead_watts,
        sustained_watts=sustained,
        peak_watts=watts * chassis.transient_factor + chassis.overhead_watts,
        sustained_limit_watts=chassis.psu_watts * chassis.max_sustained_ratio,
        unknown_gpus=tuple(unknown),
        slot_powered_watts=slot,
    )


def analyse(
    gpus: Sequence[GpuInfo],
    driver_version: str | None,
    chassis: ChassisConfig,
    paths: dict[str, GpuPath] | None = None,
    switch_vendor_ids: tuple[str, ...] = ("0x10b5",),
    modules: Sequence[ModuleConfig] = (),
) -> CompatReport:
    caps = tuple(capabilities(g) for g in gpus)
    members, basis = chassis_members(caps, paths, switch_vendor_ids)
    module_reports: list[ModuleReport] = []
    if modules:
        from astra.hardware.modules import assign

        by_uuid = {c.gpu.uuid: c for c in caps}
        for a in assign(gpus, modules)[0]:
            brick = replace(
                chassis,
                psu_watts=a.module.psu_watts,
                overhead_watts=a.module.overhead_watts,
                slots=len(a.module.gpus),
            )
            m_caps = tuple(by_uuid[g.uuid] for g in a.gpus)
            module_reports.append(
                ModuleReport(a.module, m_caps, a.missing, power_budget(m_caps, brick))
            )
        in_modules = {c.gpu.uuid for m in module_reports for c in m.members}
        members = tuple(c for c in caps if c.gpu.uuid in in_modules)
        basis = "GPUs in ASTRA Stack modules ([[module]])"
    findings: list[Finding] = []

    def add(sev: str, code: str, msg: str) -> None:
        findings.append(Finding(sev, code, msg))

    # --- driver window: newest "min branch" .. oldest "last branch" -------------
    known = [c for c in caps if c.arch]
    lo = max((c.arch.min_driver_branch for c in known if c.arch), default=0)
    hi_candidates = [
        c.arch.last_driver_branch for c in known if c.arch and c.arch.last_driver_branch
    ]
    hi = min(hi_candidates) if hi_candidates else None
    branch = driver_branch(driver_version)

    for c in caps:
        label = f"GPU {c.gpu.index} {c.gpu.name}"
        if c.arch is None:
            add(
                "warn",
                "unknown_arch",
                f"{label}: architecture unknown — engine support and speed cannot be predicted",
            )
        elif c.arch.name == "Kepler":
            add(
                "fail",
                "kepler",
                f"{label}: Kepler is only supported by driver R470 and older, which cannot drive "
                "any current GPU; remove it from the chassis",
            )
        if c.spec is None:
            add(
                "info",
                "unknown_model",
                f"{label}: not in the spec table; no bandwidth/NVENC data, power from driver limit",
            )
        mib = (c.gpu.memory_total_bytes or 0) / (1024 * 1024)
        if 0 < mib < SMALL_VRAM_MIB:
            add(
                "info",
                "small_vram",
                f"{label}: only {mib / 1024:.0f} GB VRAM — after the CUDA reserve it adds little; "
                "auto planning may leave it out of pooled inference",
            )
        cc = c.compute_capability
        if cc is not None and cc < ENGINE_MIN_CC["vllm"]:
            add(
                "info",
                "no_vllm",
                f"{label}: compute capability {cc} < 7.0 — llama.cpp only (no vLLM)",
            )
        if cc is not None and cc < LLAMACPP_QUALIFIED_CC:
            add(
                "warn",
                "legacy_cc",
                f"{label}: compute capability {cc} is below ASTRA's qualified minimum 6.1",
            )

    if hi is not None and lo > hi:
        add(
            "fail",
            "driver_window_empty",
            f"no single driver supports every GPU: newest card needs ≥ R{lo}, oldest card ends at "
            f"R{hi} — remove the oldest or the newest card",
        )
    elif branch is not None:
        if branch < lo:
            add(
                "fail",
                "driver_too_old",
                f"driver R{branch} is older than R{lo} needed by the newest GPU",
            )
        if hi is not None and branch > hi:
            add(
                "fail",
                "driver_too_new",
                f"driver R{branch} has dropped support for pre-Turing GPUs (last branch R{hi}); "
                f"install the R{hi} branch — such GPUs may not appear in nvidia-smi at all",
            )
    if hi is not None and not any(f.code.startswith("driver") for f in findings):
        add(
            "warn",
            "driver_pinned",
            f"pre-Turing GPU present: pin the driver to the R{hi} branch and hold upgrades "
            "(newer branches drop Maxwell/Pascal/Volta)",
        )

    sms = sorted({c.sm for c in caps if c.sm}, key=int)
    cuda_archs = ";".join(sms)
    if any(c.arch and c.arch.max_cuda_major == 12 for c in caps):
        add(
            "warn",
            "cuda12_build",
            f"engine builds must use CUDA 12.x (CUDA 13 dropped sm < 75) with "
            f'CMAKE_CUDA_ARCHITECTURES="{cuda_archs}"',
        )
    elif len(sms) > 1:
        add(
            "info",
            "multi_arch_build",
            f'mixed architectures: engine builds need CMAKE_CUDA_ARCHITECTURES="{cuda_archs}"',
        )

    if caps and not any(c.nvenc for c in caps):
        add("info", "no_nvenc", "no NVENC-capable GPU: the partitioned transcode profile needs one")

    for m in module_reports:
        if m.missing:
            add(
                "fail",
                "module_missing_gpu",
                f"module {m.name}: no installed GPU matches {', '.join(m.missing)} "
                "(brick unpowered, cable loose, or MMIO/BAR space exhausted: R-15)",
            )
        if not m.power.ok:
            add(
                "fail",
                "module_psu_undersized",
                f"module {m.name}: PSU {m.power.psu_watts} W too small: sustained "
                f"{m.power.sustained_watts:.0f} W (limit {m.power.sustained_limit_watts:.0f} W), "
                f"peak {m.power.peak_watts:.0f} W — use ≥ {m.power.recommended_psu_watts} W",
            )

    if not module_reports and len(members) > chassis.slots:
        add(
            "fail",
            "slots",
            f"{len(members)} chassis GPUs but the backplane has {chassis.slots} slots",
        )

    power = power_budget(members, chassis)
    if power.unknown_gpus:
        add(
            "warn",
            "power_unknown",
            f"no power figure for {', '.join(power.unknown_gpus)}; 75 W assumed, measure it",
        )
    if not power.ok and not module_reports:  # bricks carry their own PSUs
        add(
            "fail",
            "psu_undersized",
            f"chassis PSU {power.psu_watts} W too small: sustained {power.sustained_watts:.0f} W "
            f"(limit {power.sustained_limit_watts:.0f} W), peak {power.peak_watts:.0f} W — "
            f"use ≥ {power.recommended_psu_watts} W",
        )
    if power.slot_powered_watts > 150:
        add(
            "info",
            "backplane_feed",
            f"backplane delivers {power.slot_powered_watts:.0f} W of slot power: connect every "
            "backplane power input",
        )

    return CompatReport(
        driver_version=driver_version,
        caps=caps,
        chassis=members,
        chassis_basis=basis,
        findings=tuple(findings),
        power=power,
        cuda_architectures=cuda_archs,
        required_driver_window=(lo, hi),
        modules=tuple(module_reports),
    )

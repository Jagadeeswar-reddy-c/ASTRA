"""Typed inventory records shared by probe, validation, planner and telemetry."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

from astra.units import link_bandwidth_gbps

# Clock-event reasons that indicate the GPU is being slowed by a fault condition
# (heat, power brake, external slowdown) rather than by idling or app settings.
FAULT_SLOWDOWN_REASONS = frozenset(
    {"hw_slowdown", "hw_thermal_slowdown", "hw_power_brake_slowdown", "sw_thermal_slowdown"}
)


@dataclass(frozen=True)
class PcieLink:
    """One PCIe device on the path from the root complex to a GPU (Linux sysfs)."""

    bdf: str
    vendor_id: str | None = None
    device_id: str | None = None
    class_code: str | None = None
    current_gen: int | None = None
    current_width: int | None = None
    max_gen: int | None = None
    max_width: int | None = None

    @property
    def is_bridge(self) -> bool:
        return bool(self.class_code and self.class_code.lower().startswith("0x0604"))

    @property
    def bandwidth_gbps(self) -> float | None:
        return link_bandwidth_gbps(self.current_gen, self.current_width)


@dataclass(frozen=True)
class AerCounters:
    correctable: int = 0
    nonfatal: int = 0
    fatal: int = 0

    @property
    def total(self) -> int:
        return self.correctable + self.nonfatal + self.fatal


@dataclass(frozen=True)
class GpuInfo:
    index: int
    uuid: str
    name: str
    bus_id: str
    architecture: str | None = None
    vbios: str | None = None
    link_gen_current: int | None = None
    link_gen_max: int | None = None
    link_width_current: int | None = None
    link_width_max: int | None = None
    memory_total_bytes: int | None = None
    memory_used_bytes: int | None = None
    memory_free_bytes: int | None = None
    utilization_pct: float | None = None
    temperature_c: float | None = None
    temperature_slowdown_c: float | None = None
    power_draw_w: float | None = None
    power_limit_w: float | None = None
    fan_pct: float | None = None
    performance_state: str | None = None
    replay_counter: int | None = None
    active_clock_events: tuple[str, ...] = ()
    power_max_limit_w: float | None = None
    display_active: bool | None = None

    @property
    def short_bdf(self) -> str:
        """Linux-style BDF ('0000:01:00.0') from nvidia-smi's 8-digit domain form."""
        bus = self.bus_id.lower()
        if len(bus.split(":")[0]) == 8:
            bus = bus[4:]
        return bus

    @property
    def fault_slowdowns(self) -> tuple[str, ...]:
        return tuple(r for r in self.active_clock_events if r in FAULT_SLOWDOWN_REASONS)


@dataclass(frozen=True)
class GpuPath:
    """PCIe path for one GPU plus the derived upstream bottleneck."""

    gpu_bdf: str
    chain: tuple[PcieLink, ...] = ()
    aer: AerCounters = field(default_factory=AerCounters)

    @property
    def bottleneck(self) -> PcieLink | None:
        """The link with the lowest bandwidth on the path (normally the OCuLink uplink)."""
        measured = [link for link in self.chain if link.bandwidth_gbps is not None]
        if not measured:
            return None
        return min(measured, key=lambda link: link.bandwidth_gbps or 0.0)

    def switches(self, vendor_ids: tuple[str, ...]) -> tuple[PcieLink, ...]:
        wanted = {v.lower() for v in vendor_ids}
        return tuple(
            link
            for link in self.chain
            if link.is_bridge and (link.vendor_id or "").lower() in wanted
        )


@dataclass(frozen=True)
class Inventory:
    gpus: tuple[GpuInfo, ...]
    driver_version: str | None = None
    cuda_version: str | None = None
    platform: str = ""
    paths: dict[str, GpuPath] = field(default_factory=dict)  # keyed by GPU uuid
    timestamp: float = 0.0

    def gpu(self, uuid: str) -> GpuInfo | None:
        return next((g for g in self.gpus if g.uuid == uuid), None)

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        for uuid, path in self.paths.items():
            bottleneck = path.bottleneck
            data["paths"][uuid]["bottleneck"] = asdict(bottleneck) if bottleneck else None
            data["paths"][uuid]["bottleneck_gbps"] = (
                bottleneck.bandwidth_gbps if bottleneck else None
            )
        return data

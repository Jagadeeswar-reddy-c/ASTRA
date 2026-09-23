"""Node configuration (``astra.toml``).

The config describes the *expected* hardware so validation can compare what the
host actually enumerated against the as-built node, plus tunables for the planner,
telemetry and validation. Every field has a safe default so the CLI works without
a config file; validation checks that need expectations are skipped in that case.
"""

from __future__ import annotations

import os
import tomllib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from astra.errors import ConfigError

DEFAULT_SEARCH_PATHS = (Path("astra.toml"), Path("/etc/astra/astra.toml"))
PLX_VENDOR_ID = "0x10b5"  # Broadcom / PLX Technology


@dataclass(frozen=True)
class ExpectedGpu:
    match: str  # case-insensitive substring of the product name, e.g. "RTX 2060"
    vram_mib: int | None = None


@dataclass(frozen=True)
class NodeConfig:
    name: str = "astra-01"
    expected_gpus: tuple[ExpectedGpu, ...] = ()


@dataclass(frozen=True)
class InterconnectConfig:
    expected_uplink_gen: int = 3
    expected_uplink_width: int = 4
    require_switch: bool = True
    switch_vendor_ids: tuple[str, ...] = (PLX_VENDOR_ID,)


@dataclass(frozen=True)
class ChassisConfig:
    psu_watts: int = 650
    slots: int = 4  # downstream slots on the switch backplane
    overhead_watts: int = 25  # switch, OCuLink receiver, fans
    transient_factor: float = 1.3  # GPU power excursions above board power
    max_sustained_ratio: float = 0.7  # sustained load ceiling as a share of PSU rating


@dataclass(frozen=True)
class ThermalConfig:
    warn_c: float = 80.0
    crit_c: float = 87.0


@dataclass(frozen=True)
class PlannerConfig:
    gpu_memory_utilization: float = 0.90
    reserve_mib: int = 768  # CUDA context + compute buffers kept free per GPU
    default_context: int = 8192
    objective: str = "speed"  # "speed": fastest GPU set; "balanced": equal memory pressure
    exclude_gpus: tuple[str, ...] = ()  # UUIDs / name substrings never auto-selected


@dataclass(frozen=True)
class FabricConfig:
    """ADR-0011: GPUs on other machines, reached through llama.cpp rpc-server."""

    node_name: str = ""  # defaults to [node].name
    bind: str = "0.0.0.0"  # restrict to the cluster LAN interface in production
    agent_port: int = 9837
    discovery_port: int = 9836
    rpc_port_base: int = 50052
    rpc_binary: str = "rpc-server"
    network_hop_ms: float = 2.0  # per token, per stage boundary between machines
    peers: tuple[str, ...] = ()  # static peer list; empty = UDP discovery


@dataclass(frozen=True)
class TelemetryConfig:
    listen: str = "0.0.0.0"
    port: int = 9835
    min_sample_interval_s: float = 2.0


@dataclass(frozen=True)
class ValidationConfig:
    max_replay_count: int = 0
    runtime_tolerance: float = 0.15
    runtime_samples: int = 10
    runtime_sample_interval_s: float = 0.5


@dataclass(frozen=True)
class AstraConfig:
    node: NodeConfig = field(default_factory=NodeConfig)
    interconnect: InterconnectConfig = field(default_factory=InterconnectConfig)
    chassis: ChassisConfig = field(default_factory=ChassisConfig)
    thermal: ThermalConfig = field(default_factory=ThermalConfig)
    planner: PlannerConfig = field(default_factory=PlannerConfig)
    fabric: FabricConfig = field(default_factory=FabricConfig)
    telemetry: TelemetryConfig = field(default_factory=TelemetryConfig)
    validation: ValidationConfig = field(default_factory=ValidationConfig)
    source: Path | None = None


def _section(data: dict[str, Any], name: str) -> dict[str, Any]:
    value = data.get(name, {})
    if not isinstance(value, dict):
        raise ConfigError(f"[{name}] must be a table")
    return value


def _build(cls: type[Any], values: dict[str, Any], section: str) -> Any:
    known = set(cls.__dataclass_fields__)
    unknown = set(values) - known
    if unknown:
        raise ConfigError(f"unknown key(s) in [{section}]: {', '.join(sorted(unknown))}")
    try:
        return cls(**values)
    except TypeError as exc:  # pragma: no cover - defensive
        raise ConfigError(f"invalid [{section}]: {exc}") from exc


def parse_config(data: dict[str, Any], source: Path | None = None) -> AstraConfig:
    node_raw = dict(_section(data, "node"))
    gpus_raw = node_raw.pop("expected_gpu", [])
    if not isinstance(gpus_raw, list):
        raise ConfigError("[[node.expected_gpu]] must be an array of tables")
    expected = []
    for i, g in enumerate(gpus_raw):
        if not isinstance(g, dict) or "match" not in g:
            raise ConfigError(f"node.expected_gpu[{i}] needs a 'match' string")
        expected.append(_build(ExpectedGpu, g, f"node.expected_gpu[{i}]"))
    node = _build(NodeConfig, {**node_raw, "expected_gpus": tuple(expected)}, "node")

    inter_raw = dict(_section(data, "interconnect"))
    if "switch_vendor_ids" in inter_raw:
        inter_raw["switch_vendor_ids"] = tuple(v.lower() for v in inter_raw["switch_vendor_ids"])

    planner_raw = dict(_section(data, "planner"))
    if "exclude_gpus" in planner_raw:
        planner_raw["exclude_gpus"] = tuple(planner_raw["exclude_gpus"])

    fabric_raw = dict(_section(data, "fabric"))
    if "peers" in fabric_raw:
        fabric_raw["peers"] = tuple(fabric_raw["peers"])

    cfg = AstraConfig(
        node=node,
        interconnect=_build(InterconnectConfig, inter_raw, "interconnect"),
        chassis=_build(ChassisConfig, _section(data, "chassis"), "chassis"),
        thermal=_build(ThermalConfig, _section(data, "thermal"), "thermal"),
        planner=_build(PlannerConfig, planner_raw, "planner"),
        fabric=_build(FabricConfig, fabric_raw, "fabric"),
        telemetry=_build(TelemetryConfig, _section(data, "telemetry"), "telemetry"),
        validation=_build(ValidationConfig, _section(data, "validation"), "validation"),
        source=source,
    )
    _check(cfg)
    return cfg


def _check(cfg: AstraConfig) -> None:
    if not 0.1 <= cfg.planner.gpu_memory_utilization <= 1.0:
        raise ConfigError("planner.gpu_memory_utilization must be within [0.1, 1.0]")
    if cfg.planner.reserve_mib < 0:
        raise ConfigError("planner.reserve_mib must be >= 0")
    if cfg.planner.objective not in ("speed", "balanced"):
        raise ConfigError("planner.objective must be 'speed' or 'balanced'")
    if cfg.chassis.psu_watts <= 0 or not 0.1 <= cfg.chassis.max_sustained_ratio <= 1.0:
        raise ConfigError("chassis.psu_watts must be > 0 and max_sustained_ratio in [0.1, 1.0]")
    if cfg.chassis.transient_factor < 1.0:
        raise ConfigError("chassis.transient_factor must be >= 1.0")
    if cfg.fabric.network_hop_ms < 0:
        raise ConfigError("fabric.network_hop_ms must be >= 0")
    if cfg.thermal.warn_c >= cfg.thermal.crit_c:
        raise ConfigError("thermal.warn_c must be lower than thermal.crit_c")
    if not 0 < cfg.validation.runtime_tolerance < 1:
        raise ConfigError("validation.runtime_tolerance must be within (0, 1)")


def load_config(path: str | Path | None = None) -> AstraConfig:
    """Load config from an explicit path, $ASTRA_CONFIG, or the default search paths."""
    candidates: list[Path]
    if path is not None:
        candidates = [Path(path)]
    elif os.environ.get("ASTRA_CONFIG"):
        candidates = [Path(os.environ["ASTRA_CONFIG"])]
    else:
        candidates = [p for p in DEFAULT_SEARCH_PATHS if p.is_file()]
        if not candidates:
            return AstraConfig()

    chosen = candidates[0]
    if not chosen.is_file():
        raise ConfigError(f"config file not found: {chosen}")
    try:
        with chosen.open("rb") as fh:
            data = tomllib.load(fh)
    except tomllib.TOMLDecodeError as exc:
        raise ConfigError(f"{chosen}: {exc}") from exc
    return parse_config(data, source=chosen)

"""Wire format shared by the agent, discovery and the head node.

Deliberately small and versioned: a node descriptor is plain JSON so any tool
(curl, a dashboard) can read it. RPC endpoints travel as *ports only*; the
client combines them with the address it reached the agent on, so agents never
have to guess which of their addresses is routable.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

from astra import __version__
from astra.errors import ParseError
from astra.hardware.models import GpuInfo, Inventory

PROTOCOL = "astra-fabric/1"
DISCOVER_MESSAGE = f"ASTRA-DISCOVER {PROTOCOL}".encode()


@dataclass(frozen=True)
class NodeGpu:
    uuid: str
    name: str
    bus_id: str
    architecture: str | None
    memory_total_bytes: int | None
    memory_used_bytes: int | None
    display_active: bool | None = None
    rpc_port: int | None = None  # set when the agent serves this GPU via rpc-server
    temperature_c: float | None = None
    utilization_pct: float | None = None
    power_draw_w: float | None = None

    def to_gpu_info(self, index: int) -> GpuInfo:
        return GpuInfo(
            index=index,
            uuid=self.uuid,
            name=self.name,
            bus_id=self.bus_id,
            architecture=self.architecture,
            memory_total_bytes=self.memory_total_bytes,
            memory_used_bytes=self.memory_used_bytes,
            display_active=self.display_active,
            temperature_c=self.temperature_c,
            utilization_pct=self.utilization_pct,
            power_draw_w=self.power_draw_w,
        )


@dataclass(frozen=True)
class NodeDescriptor:
    node: str
    api_port: int
    driver_version: str | None
    gpus: tuple[NodeGpu, ...]
    version: str = __version__
    protocol: str = PROTOCOL
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> dict[str, Any]:
        return {
            "protocol": self.protocol,
            "version": self.version,
            "node": self.node,
            "api_port": self.api_port,
            "driver_version": self.driver_version,
            "timestamp": self.timestamp,
            "gpus": [g.__dict__ for g in self.gpus],
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> NodeDescriptor:
        if not isinstance(data, dict) or data.get("protocol") != PROTOCOL:
            raise ParseError(f"not an {PROTOCOL} node descriptor")
        try:
            gpus = tuple(
                NodeGpu(
                    uuid=str(g["uuid"]),
                    name=str(g["name"]),
                    bus_id=str(g.get("bus_id", "")),
                    architecture=g.get("architecture"),
                    memory_total_bytes=g.get("memory_total_bytes"),
                    memory_used_bytes=g.get("memory_used_bytes"),
                    display_active=g.get("display_active"),
                    rpc_port=g.get("rpc_port"),
                    temperature_c=g.get("temperature_c"),
                    utilization_pct=g.get("utilization_pct"),
                    power_draw_w=g.get("power_draw_w"),
                )
                for g in data.get("gpus", [])
            )
            return cls(
                node=str(data["node"]),
                api_port=int(data["api_port"]),
                driver_version=data.get("driver_version"),
                gpus=gpus,
                version=str(data.get("version", "?")),
                timestamp=float(data.get("timestamp", 0.0)),
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise ParseError(f"malformed node descriptor: {exc}") from exc


def describe(
    node: str, api_port: int, inventory: Inventory, rpc_ports: dict[str, int] | None = None
) -> NodeDescriptor:
    rpc_ports = rpc_ports or {}
    return NodeDescriptor(
        node=node,
        api_port=api_port,
        driver_version=inventory.driver_version,
        gpus=tuple(
            NodeGpu(
                uuid=g.uuid,
                name=g.name,
                bus_id=g.bus_id,
                architecture=g.architecture,
                memory_total_bytes=g.memory_total_bytes,
                memory_used_bytes=g.memory_used_bytes,
                display_active=g.display_active,
                rpc_port=rpc_ports.get(g.uuid),
                temperature_c=g.temperature_c,
                utilization_pct=g.utilization_pct,
                power_draw_w=g.power_draw_w,
            )
            for g in inventory.gpus
        ),
    )

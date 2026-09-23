"""The GPU pool: every GPU ASTRA can plan on (local, fabric nodes or simulated),
with both a planner view (budgets) and a topology view (nodes with live stats).

Shared by the CLI (`astra plan --cluster`) and the web UI (`astra ui`).
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any

from astra.config import AstraConfig
from astra.errors import AstraError, CommandError
from astra.fabric.cluster import remote_budgets
from astra.fabric.discovery import Peer, discover, gather, parse_peers
from astra.hardware.compat import capabilities
from astra.hardware.gpu_specs import SPECS, lookup
from astra.hardware.models import GpuInfo
from astra.hardware.probe import probe
from astra.planner.split import DeviceBudget, budgets_from_gpus
from astra.units import MIB

# Named GPU mixes for --simulate. "reference" is the concept document's build.
SIMULATION_PRESETS = {
    "reference": "RTX 2060:6144,RTX 3050:8192",
    "budget-mix": "GT 1030,GTX 1660 SUPER,RTX 3060:12288",
    "mixed-gen": "GTX 1080 Ti,RTX 2070 SUPER,RTX 4060 Ti:16384",
    "fabric-demo": "RTX 3050:8192, GT 1030, @desktop RTX 3060 Ti, @laptop RTX 3060:8192",
}


def simulated(spec: str) -> tuple[list[GpuInfo], dict[str, str]]:
    """Build GPUs from 'name[:mib[:arch]],...' (or a preset) to plan before buying hardware.

    Known models only need a name ("RTX 3060", "GT 1030"); VRAM defaults to the
    largest variant of that model. Prefix "2x " to add several identical cards and
    "@<node> " to place a card on another machine of the fabric (ADR-0011), e.g.
    "RTX 3050, @desktop RTX 3060 Ti". Returns the GPUs and {uuid: node} for remote ones.
    """
    spec = SIMULATION_PRESETS.get(spec, spec)
    items: list[tuple[str | None, str]] = []
    for raw in (s.strip() for s in spec.split(",")):
        if not raw:
            continue
        node = None
        if raw.startswith("@"):
            node, _, raw = raw[1:].partition(" ")
            raw = raw.strip()
            if not node or not raw:
                raise AstraError(f"--simulate: '@{node}' must be followed by a GPU name")
        count, sep, rest = raw.partition("x ")
        if sep and count.strip().isdigit():
            items += [(node, rest.strip())] * int(count)
        else:
            items.append((node, raw))
    if not items:
        raise AstraError("--simulate needs at least one GPU")
    gpus: list[GpuInfo] = []
    remote: dict[str, str] = {}
    for i, (node, item) in enumerate(items):
        parts = [x.strip() for x in item.split(":")]
        known = lookup(parts[0])
        if len(parts) > 1 and parts[1]:
            if not parts[1].isdigit():
                raise AstraError(f"--simulate '{item}': VRAM must be MiB, e.g. 'RTX 3060:12288'")
            mib = int(parts[1])
        elif known:
            mib = max(s.vram_mib for s in SPECS if s.model == known.model)
        else:
            raise AstraError(f"--simulate: unknown GPU '{parts[0]}'; use 'name:mib[:arch]'")
        if known:
            known = lookup(parts[0], mib * MIB)
        arch = parts[2] if len(parts) > 2 and parts[2] else (known.architecture if known else None)
        gpu = GpuInfo(
            index=i,
            uuid=f"GPU-sim-{i:04d}",
            name=f"NVIDIA GeForce {known.model}" if known else parts[0],
            bus_id=f"00000000:{5 + i:02X}:00.0",
            architecture=arch,
            memory_total_bytes=mib * MIB,
            memory_used_bytes=0,
            display_active=False,
        )
        gpus.append(gpu)
        if node:
            remote[gpu.uuid] = node
    return gpus, remote


@dataclass
class PoolNode:
    name: str
    host: str
    role: str  # "head" or "worker"
    driver_version: str | None
    gpus: list[tuple[GpuInfo, DeviceBudget | None]] = field(default_factory=list)
    version: str | None = None


@dataclass
class Pool:
    nodes: list[PoolNode]
    budgets: list[DeviceBudget]
    notes: list[str]
    simulated: bool = False

    def to_dict(self) -> dict[str, Any]:
        out_nodes = []
        for n in self.nodes:
            gpus = []
            for g, b in n.gpus:
                caps = capabilities(g)
                gpus.append(
                    {
                        "index": b.index if b else None,
                        "uuid": g.uuid,
                        "name": g.name,
                        "model": caps.spec.model if caps.spec else g.name,
                        "architecture": caps.architecture,
                        "compute_capability": caps.compute_capability,
                        "memory_total_bytes": g.memory_total_bytes,
                        "memory_used_bytes": g.memory_used_bytes,
                        "temperature_c": g.temperature_c,
                        "utilization_pct": g.utilization_pct,
                        "power_draw_w": g.power_draw_w,
                        "board_power_w": caps.board_power_w,
                        "bandwidth_gbps": caps.mem_bandwidth_gbps,
                        "nvenc": caps.nvenc,
                        "display_active": g.display_active,
                        "rpc_endpoint": b.rpc_endpoint if b else None,
                        "usable": b is not None,
                    }
                )
            out_nodes.append(
                {
                    "name": n.name,
                    "host": n.host,
                    "role": n.role,
                    "driver_version": n.driver_version,
                    "version": n.version,
                    "gpus": gpus,
                }
            )
        return {"simulated": self.simulated, "nodes": out_nodes, "notes": self.notes}


def peers_for(cfg: AstraConfig, peers: str | None, timeout_s: float) -> list[Peer]:
    """Explicit peers, else [fabric].peers, else UDP discovery."""
    text = peers or ",".join(cfg.fabric.peers)
    if text:
        return parse_peers(text, cfg.fabric.agent_port)
    return discover(cfg.fabric.discovery_port, timeout_s)


def collect(
    cfg: AstraConfig,
    util: float,
    reserve_mib: int,
    simulate: str | None = None,
    fabric: bool = False,
    peers: str | None = None,
    timeout_s: float = 1.5,
) -> Pool:
    """Local GPUs first (PCI order), then GPUs on other fabric nodes, grouped by node."""
    head = cfg.fabric.node_name or cfg.node.name
    if simulate:
        gpus, remote_nodes = simulated(simulate)
        local = [g for g in gpus if g.uuid not in remote_nodes]
        budgets, sim_notes = budgets_from_gpus(local, util, reserve_mib)
        sim_nodes = {head: PoolNode(head, "localhost", "head", None)}
        by_uuid = {b.uuid: b for b in budgets}
        sim_nodes[head].gpus = [(g, by_uuid.get(g.uuid)) for g in local]
        remote = [g for g in gpus if g.uuid in remote_nodes]
        remote_b, remote_notes = budgets_from_gpus(remote, util, reserve_mib)
        port: dict[str, int] = {}
        info = {g.uuid: g for g in remote}
        for b in sorted(remote_b, key=lambda b: remote_nodes[b.uuid]):
            name = remote_nodes[b.uuid]
            port[name] = port.get(name, cfg.fabric.rpc_port_base - 1) + 1
            nb = replace(b, node=name, rpc_endpoint=f"{name}:{port[name]}", index=len(budgets))
            budgets.append(nb)
            node = sim_nodes.setdefault(name, PoolNode(name, name, "worker", None))
            node.gpus.append((info[b.uuid], nb))
        return Pool(list(sim_nodes.values()), budgets, sim_notes + remote_notes, simulated=True)

    notes: list[str] = []
    try:
        inv = probe(with_topology=False)
        local_gpus, driver = list(inv.gpus), inv.driver_version
    except CommandError as exc:
        if not fabric:
            raise
        local_gpus, driver = [], None  # a GPU-less head node that only drives remote GPUs
        notes.append(f"head node has no usable NVIDIA driver ({exc})")
    budgets, warn = budgets_from_gpus(local_gpus, util, reserve_mib)
    notes += warn
    by_uuid = {b.uuid: b for b in budgets}
    head_node = PoolNode(head, "localhost", "head", driver)
    head_node.gpus = [(g, by_uuid.get(g.uuid)) for g in local_gpus]
    nodes = [head_node]
    if fabric:
        found, errors = gather(peers_for(cfg, peers, timeout_s))
        notes += [f"fabric: {e}" for e in errors]
        if not found and not errors:
            notes.append("fabric: no agents found (run `astra agent --rpc` on each machine)")
        remote_b, remote_notes = remote_budgets(found, util, reserve_mib, len(budgets), head)
        budgets += remote_b
        notes += remote_notes
        remote_by_uuid = {b.uuid: b for b in remote_b}
        for peer, desc in sorted(found, key=lambda f: f[1].node):
            if desc.node == head:
                continue
            node = PoolNode(
                desc.node, peer.host, "worker", desc.driver_version, version=desc.version
            )
            node.gpus = [(g.to_gpu_info(-1), remote_by_uuid.get(g.uuid)) for g in desc.gpus]
            nodes.append(node)
    return Pool(nodes, budgets, notes)

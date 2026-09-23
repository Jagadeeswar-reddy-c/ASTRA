"""Turn discovered nodes into planner budgets for GPUs on other machines."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import replace

from astra.fabric.discovery import Peer
from astra.fabric.protocol import NodeDescriptor
from astra.planner.split import DeviceBudget, budgets_from_gpus


def remote_budgets(
    nodes: Sequence[tuple[Peer, NodeDescriptor]],
    gpu_memory_utilization: float,
    reserve_mib: int,
    first_index: int,
    local_node: str,
) -> tuple[list[DeviceBudget], list[str]]:
    """Budgets for every RPC-served GPU on other nodes, grouped by node.

    GPUs are numbered after the local ones so indices stay unique across the
    cluster. A GPU whose agent runs without ``--rpc`` is reported, not used.
    """
    budgets: list[DeviceBudget] = []
    notes: list[str] = []
    index = first_index
    for peer, desc in sorted(nodes, key=lambda n: n[1].node):
        if desc.node == local_node:
            continue  # the head's own agent: its GPUs are already local
        served = [g for g in desc.gpus if g.rpc_port]
        for g in desc.gpus:
            if not g.rpc_port:
                notes.append(f"{desc.node}: {g.name} not served (start `astra agent --rpc`)")
        infos = [g.to_gpu_info(0) for g in served]
        node_budgets, warnings = budgets_from_gpus(infos, gpu_memory_utilization, reserve_mib)
        notes += [f"{desc.node}: {w}" for w in warnings]
        port = {g.uuid: g.rpc_port for g in served}
        for b in node_budgets:
            budgets.append(
                replace(
                    b,
                    index=index,
                    node=desc.node,
                    rpc_endpoint=f"{peer.host}:{port[b.uuid]}",
                    display_active=b.display_active,
                )
            )
            index += 1
    return budgets, notes

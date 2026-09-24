"""ASTRA Fabric (ADR-0011): agent, discovery, cluster budgets, planning and launch
across machines."""

import json
import sys
import threading
import time
from collections.abc import Iterator
from dataclasses import replace
from http.server import ThreadingHTTPServer
from pathlib import Path

import pytest

from astra import cli, pool
from astra.config import AstraConfig, parse_config
from astra.errors import ParseError, PlanningError
from astra.fabric.agent import (
    DiscoveryResponder,
    RpcSupervisor,
    make_handler,
    rpc_server_specs,
)
from astra.fabric.cluster import remote_budgets
from astra.fabric.discovery import Peer, discover, fetch_descriptor, gather, parse_peers
from astra.fabric.protocol import NodeDescriptor, describe
from astra.hardware.models import GpuInfo, Inventory
from astra.planner import engines
from astra.planner.models import from_catalog
from astra.planner.split import budgets_from_gpus, plan, select_gpus
from astra.validation.checks import run_runtime_checks
from astra.validation.report import Status

GIB = 1 << 30


def _gpu(i: int, name: str, gib: int, arch: str) -> GpuInfo:
    return GpuInfo(
        i,
        f"GPU-{name.replace(' ', '')}-{i}",
        f"NVIDIA GeForce {name}",
        f"00000000:{1 + i:02X}:00.0",
        architecture=arch,
        memory_total_bytes=gib * GIB,
        memory_used_bytes=0,
    )


def _descriptor(node: str, gpus: list[GpuInfo], served: bool = True) -> NodeDescriptor:
    inv = Inventory(tuple(gpus), "580.95", "12.9")
    ports = {g.uuid: 50052 + i for i, g in enumerate(gpus)} if served else {}
    return describe(node, 9837, inv, ports)


# ---------------------------------------------------------------------- protocol


def test_descriptor_round_trip() -> None:
    desc = _descriptor("desktop", [_gpu(0, "RTX 3060 Ti", 8, "Ampere")])
    back = NodeDescriptor.from_dict(json.loads(json.dumps(desc.to_dict())))
    assert back.node == "desktop" and back.gpus[0].rpc_port == 50052
    assert back.gpus[0].to_gpu_info(3).index == 3


@pytest.mark.parametrize("data", [{}, {"protocol": "other"}, {"protocol": "astra-fabric/1"}])
def test_descriptor_rejects_garbage(data: dict[str, object]) -> None:
    with pytest.raises(ParseError):
        NodeDescriptor.from_dict(data)


def test_parse_peers() -> None:
    assert parse_peers("10.0.0.5, pc2:9900,", 9837) == [Peer("10.0.0.5", 9837), Peer("pc2", 9900)]
    assert Peer("pc2", 9900).api_url == "http://pc2:9900"


# ------------------------------------------------------------------------- agent


def test_rpc_server_specs_pin_one_gpu_each() -> None:
    gpus = [_gpu(1, "RTX 3060", 12, "Ampere"), _gpu(0, "GT 1030", 2, "Pascal")]
    specs = rpc_server_specs(gpus, "rpc-server", "10.0.0.2", 50100)
    assert [s.port for s in specs] == [50100, 50101]
    assert specs[0].uuid == gpus[1].uuid  # PCI order
    assert specs[0].env == {"CUDA_DEVICE_ORDER": "PCI_BUS_ID", "CUDA_VISIBLE_DEVICES": gpus[1].uuid}
    assert specs[0].argv == ("rpc-server", "--host", "10.0.0.2", "--port", "50100", "--cache")


def test_rpc_supervisor_restarts_exited_servers() -> None:
    spec = rpc_server_specs([_gpu(0, "RTX 3060", 12, "Ampere")], sys.executable)[0]
    spec = replace(spec, argv=(sys.executable, "-c", "pass"))  # exits immediately
    sup = RpcSupervisor([spec], restart_delay_s=0.05)
    spawned: list[int] = []
    original = sup._spawn

    def counting(s):  # type: ignore[no-untyped-def]
        spawned.append(1)
        original(s)

    sup._spawn = counting  # type: ignore[method-assign]
    sup.start()
    deadline = time.monotonic() + 5
    while len(spawned) < 2 and time.monotonic() < deadline:
        time.sleep(0.05)
    sup.stop()
    assert len(spawned) >= 2


@pytest.fixture
def agent() -> Iterator[tuple[Peer, NodeDescriptor]]:
    desc = _descriptor("desktop", [_gpu(0, "RTX 3060 Ti", 8, "Ampere")])
    server = ThreadingHTTPServer(("127.0.0.1", 0), make_handler(lambda: desc))
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        yield Peer("127.0.0.1", server.server_address[1]), desc
    finally:
        server.shutdown()
        server.server_close()


def test_http_descriptor_and_gather(agent: tuple[Peer, NodeDescriptor]) -> None:
    peer, desc = agent
    assert fetch_descriptor(peer).gpus == desc.gpus
    nodes, errors = gather([peer, Peer("127.0.0.1", 1)], timeout_s=1)
    assert [d.node for _, d in nodes] == ["desktop"] and len(errors) == 1


def test_udp_discovery_on_loopback() -> None:
    responder = DiscoveryResponder("desktop", 9837, "127.0.0.1", 0)
    responder.start()
    try:
        peers = discover(responder.port, timeout_s=0.8, targets=("127.0.0.1",))
    finally:
        responder.stop()
    assert peers == [Peer("127.0.0.1", 9837)]


# ----------------------------------------------------------------------- cluster


def test_remote_budgets_numbering_and_endpoints() -> None:
    nodes = [
        (Peer("10.0.0.7", 9837), _descriptor("desktop", [_gpu(0, "RTX 3060 Ti", 8, "Ampere")])),
        (Peer("10.0.0.9", 9837), _descriptor("laptop", [_gpu(0, "RTX 3050", 6, "Ampere")], False)),
        (Peer("10.0.0.1", 9837), _descriptor("astra-01", [_gpu(0, "RTX 2060", 6, "Turing")])),
    ]
    budgets, notes = remote_budgets(nodes, 0.9, 768, first_index=2, local_node="astra-01")
    assert [(b.index, b.node, b.rpc_endpoint) for b in budgets] == [
        (2, "desktop", "10.0.0.7:50052")
    ]
    assert budgets[0].bandwidth_gbps == 448 and budgets[0].is_remote
    assert any("laptop" in n and "not served" in n for n in notes)


# ---------------------------------------------------------------------- planning


def _fabric_budgets() -> list:  # type: ignore[type-arg]
    local, _ = budgets_from_gpus([_gpu(0, "RTX 3050", 8, "Ampere")], 0.9, 768)
    desk = _descriptor("desktop", [_gpu(0, "RTX 3060 Ti", 8, "Ampere")])
    remote, _ = remote_budgets([(Peer("10.0.0.7", 9837), desk)], 0.9, 768, 1, "astra-01")
    return local + remote


def test_remote_gpu_used_only_when_needed() -> None:
    budgets = _fabric_budgets()
    small = select_gpus(from_catalog("llama-3.1-8b", "q4_k_m"), budgets, "llamacpp", 4096, 0.9)
    assert all(not p.device.is_remote for p in small.placements) or len(small.placements) == 1
    big = select_gpus(from_catalog("qwen2.5-14b", "q4_k_m"), budgets, "llamacpp", 8192, 0.9)
    assert big.fits and [p.device.node for p in big.placements] == ["local", "desktop"]


def test_network_hop_costs_speed() -> None:
    budgets = _fabric_budgets()
    model = from_catalog("qwen2.5-14b", "q4_k_m")
    near = plan(model, budgets, "llamacpp", 8192, 0.9, network_hop_s=0.0)
    far = plan(model, budgets, "llamacpp", 8192, 0.9, network_hop_s=0.05)
    assert (near.est_decode_tokens_per_s or 0) > (far.est_decode_tokens_per_s or 0) * 1.5


def test_vllm_is_local_only() -> None:
    budgets = _fabric_budgets()
    model = from_catalog("llama-3.1-8b", "awq-int4")
    result = select_gpus(model, budgets, "vllm", 4096, 0.9)
    assert all(not p.device.is_remote for p in result.placements)
    assert any("local-only" in w for w in result.warnings)
    with pytest.raises(PlanningError, match="only llamacpp"):
        plan(model, budgets, "vllm", 4096, 0.9)


def test_llamacpp_launch_over_rpc() -> None:
    result = plan(from_catalog("qwen2.5-14b", "q4_k_m"), _fabric_budgets(), "llamacpp", 8192, 0.9)
    spec = engines.llamacpp(result, "/m.gguf")
    argv = list(spec.argv)
    assert argv[argv.index("--rpc") + 1] == "10.0.0.7:50052"
    assert argv[argv.index("--device") + 1] == "CUDA0,RPC0"
    assert spec.env["CUDA_VISIBLE_DEVICES"] == result.placements[0].device.uuid
    assert any("--list-devices" in n for n in spec.notes)


def test_runtime_gate_checks_only_local_gpus(reference_inventory: Inventory) -> None:
    g2060, _ = reference_inventory.gpus
    plan_doc = {
        "engine": "llamacpp",
        "planned_share": [0.4, 0.6],
        "devices": [
            {
                "index": 0,
                "uuid": g2060.uuid,
                "name": "RTX 2060",
                "baseline_used_bytes": 0,
                "capacity_bytes": 5 * GIB,
            },
            {
                "index": 1,
                "uuid": "GPU-remote",
                "name": "RTX 3060 Ti",
                "baseline_used_bytes": 0,
                "capacity_bytes": 7 * GIB,
                "rpc_endpoint": "10.0.0.7:50052",
            },
        ],
    }
    results = run_runtime_checks(
        AstraConfig(), plan_doc, lambda: reference_inventory, samples=1, sleep=lambda s: None
    )
    r = {x.id: x for x in results}
    assert r["R01"].status is Status.PASS and "+1 remote" in r["R01"].detail
    assert "plan 100%" in r["R03"].detail  # renormalised to the local GPU
    plan_doc["devices"] = plan_doc["devices"][1:]  # type: ignore[index]
    only_remote = run_runtime_checks(
        AstraConfig(), plan_doc, lambda: reference_inventory, samples=1, sleep=lambda s: None
    )
    assert only_remote[0].status is Status.SKIP


def test_fabric_config() -> None:
    cfg = parse_config(
        {"fabric": {"node_name": "mini", "peers": ["10.0.0.7"], "network_hop_ms": 1}}
    )
    assert cfg.fabric.peers == ("10.0.0.7",) and cfg.fabric.node_name == "mini"


# --------------------------------------------------------------------------- CLI


@pytest.fixture
def no_config(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("ASTRA_CONFIG", raising=False)
    return tmp_path


def test_cli_simulated_fabric(no_config: Path, capsys: pytest.CaptureFixture[str]) -> None:
    code = cli.main(
        ["plan", "--simulate", "RTX 3050:8192, @desktop RTX 3060 Ti", "--model", "qwen2.5-14b"]
    )
    out = capsys.readouterr().out
    assert code == 0 and "@desktop RTX 3060 Ti" in out
    assert "--rpc desktop:50052" in out and "RPC0" in out
    assert cli.main(["plan", "--simulate", "@desktop", "--model", "qwen2.5-14b"]) == 2


def test_cli_cluster_and_plan_with_live_agent(
    no_config: Path,
    agent: tuple[Peer, NodeDescriptor],
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    peer, _ = agent
    assert cli.main(["cluster", "--peers", f"{peer.host}:{peer.api_port}"]) == 0
    assert "rpc 127.0.0.1:50052" in capsys.readouterr().out

    # A GPU-less head node: no local driver, all compute on the fabric.
    from astra.errors import CommandError

    def no_driver(**_: object) -> Inventory:
        raise CommandError("nvidia-smi not found on PATH")

    monkeypatch.setattr(cli, "probe", no_driver)
    monkeypatch.setattr(pool, "probe", no_driver)  # plans collect GPUs through astra.pool
    code = cli.main(
        [
            "plan",
            "--peers",
            f"{peer.host}:{peer.api_port}",
            "--model",
            "llama-3.1-8b",
            "--context",
            "2048",
        ]
    )
    out = capsys.readouterr().out
    assert code == 0 and "@desktop RTX 3060 Ti" in out and "--rpc 127.0.0.1:50052" in out


def test_cli_agent_needs_rpc_server_binary(
    no_config: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert cli.main(["agent", "--rpc", "--rpc-binary", "definitely-not-installed-rpc"]) == 2
    assert "GGML_RPC" in capsys.readouterr().err


# ----------------------------------------------------------- topology in emit-config


def test_emit_config_captures_switch_topology(reference_inventory: Inventory) -> None:
    lines = cli._topology_config(
        reference_inventory, AstraConfig(), [g.uuid for g in reference_inventory.gpus]
    )
    assert "require_switch = true" in lines
    assert "expected_uplink_width = 4" in lines and "expected_uplink_gen = 3" in lines


def test_emit_config_captures_direct_slot_topology(reference_inventory: Inventory) -> None:
    from astra.hardware.models import GpuPath, PcieLink

    direct = GpuPath(
        "0000:01:00.0",
        chain=(
            PcieLink(
                "0000:00:01.0",
                "0x8086",
                class_code="0x060400",
                current_gen=4,
                current_width=16,
                max_gen=4,
                max_width=16,
            ),
            PcieLink(
                "0000:01:00.0",
                "0x10de",
                class_code="0x030000",
                current_gen=4,
                current_width=8,
                max_gen=4,
                max_width=8,
            ),
        ),
    )
    gpu = reference_inventory.gpus[1]
    inv = replace(reference_inventory, gpus=(gpu,), paths={gpu.uuid: direct})
    lines = cli._topology_config(inv, AstraConfig(), [gpu.uuid])
    assert "require_switch = false" in lines
    assert "expected_uplink_width = 8" in lines and "expected_uplink_gen = 4" in lines

import json
import xml.etree.ElementTree as ET
from dataclasses import replace
from pathlib import Path

import pytest

from astra.config import AstraConfig, ExpectedGpu, NodeConfig, load_config
from astra.hardware import nvsmi, sysfs
from astra.hardware.models import Inventory
from astra.validation.checks import LinkContext, run_link_checks, run_runtime_checks
from astra.validation.report import Report, Status
from conftest import UUID_2060, UUID_3050, DictSysfs, build_reference_sysfs, with_gpu

REPO = Path(__file__).resolve().parents[2]
CFG = load_config(REPO / "config" / "astra.example.toml")


def _by_id(results):  # type: ignore[no-untyped-def]
    return {r.id: r for r in results}


def _ctx(inv: Inventory, log: str | None = "", cfg: AstraConfig = CFG, **kw) -> LinkContext:  # type: ignore[no-untyped-def]
    return LinkContext(cfg, inv, log, **kw)


def test_healthy_reference_node_passes(reference_inventory: Inventory) -> None:
    results = _by_id(run_link_checks(_ctx(reference_inventory)))
    assert {k: r.status for k, r in results.items()} == {
        **{f"L{i:02d}": Status.PASS for i in range(1, 13)},
        "L13": Status.SKIP,  # no [[module]] bricks configured
    }
    assert "Gen3 x4" in results["L05"].detail
    assert "0000:02:00.0" in results["L04"].detail


def test_missing_gpu_fails_inventory(reference_inventory: Inventory) -> None:
    inv = replace(reference_inventory, gpus=reference_inventory.gpus[:1])
    r = _by_id(run_link_checks(_ctx(inv)))["L02"]
    assert r.status is Status.FAIL and "RTX 3050" in r.detail


def test_wrong_vram_fails_inventory(reference_inventory: Inventory) -> None:
    inv = with_gpu(reference_inventory, UUID_3050, memory_total_bytes=4096 << 20)
    assert _by_id(run_link_checks(_ctx(inv)))["L02"].status is Status.FAIL


def test_inventory_skipped_without_expectations(reference_inventory: Inventory) -> None:
    cfg = replace(CFG, node=NodeConfig(name="x"))
    assert _by_id(run_link_checks(_ctx(reference_inventory, cfg=cfg)))["L02"].status is Status.SKIP


def test_extra_host_gpu_is_tolerated(reference_inventory: Inventory) -> None:
    cfg = replace(CFG, node=NodeConfig(expected_gpus=(ExpectedGpu("RTX 2060"),)))
    r = _by_id(run_link_checks(_ctx(reference_inventory, cfg=cfg)))["L02"]
    assert r.status is Status.PASS and "other GPU" in r.detail


def _inventory_with_sysfs(reference_xml: str, root: DictSysfs) -> Inventory:
    report = nvsmi.parse_xml(reference_xml)
    paths = {g.uuid: p for g in report.gpus if (p := sysfs.gpu_path(g.short_bdf, root))}
    return Inventory(report.gpus, report.driver_version, report.cuda_version, "Linux", paths)


def test_uplink_x1_fails(reference_xml: str) -> None:
    inv = _inventory_with_sysfs(reference_xml, build_reference_sysfs(uplink_width="1"))
    r = _by_id(run_link_checks(_ctx(inv)))["L05"]
    assert r.status is Status.FAIL and "x1" in r.detail


def test_uplink_idle_downshift_warns(reference_xml: str) -> None:
    inv = _inventory_with_sysfs(reference_xml, build_reference_sysfs(uplink_speed="2.5 GT/s PCIe"))
    assert _by_id(run_link_checks(_ctx(inv)))["L05"].status is Status.WARN


def test_uplink_gen_capped_fails(reference_xml: str) -> None:
    root = build_reference_sysfs(uplink_speed="5.0 GT/s PCIe", uplink_max_speed="5.0 GT/s PCIe")
    inv = _inventory_with_sysfs(reference_xml, root)
    assert _by_id(run_link_checks(_ctx(inv)))["L05"].status is Status.FAIL


def test_aer_counters(reference_xml: str) -> None:
    inv = _inventory_with_sysfs(reference_xml, build_reference_sysfs(aer_correctable=2))
    assert _by_id(run_link_checks(_ctx(inv)))["L08"].status is Status.WARN
    inv = _inventory_with_sysfs(reference_xml, build_reference_sysfs(aer_fatal=1))
    assert _by_id(run_link_checks(_ctx(inv)))["L08"].status is Status.FAIL


def test_no_topology_skips_linux_only_checks(reference_inventory: Inventory) -> None:
    inv = replace(reference_inventory, paths={})
    results = _by_id(
        run_link_checks(_ctx(inv, None, kernel_log_error="Linux-only", topology_expected=False))
    )
    for cid in ("L04", "L05", "L08", "L09"):
        assert results[cid].status is Status.SKIP
    assert results["L01"].status is Status.PASS


def test_no_switch_found(reference_inventory: Inventory) -> None:
    cfg = replace(CFG, interconnect=replace(CFG.interconnect, switch_vendor_ids=("0x1234",)))
    assert _by_id(run_link_checks(_ctx(reference_inventory, cfg=cfg)))["L04"].status is Status.FAIL
    cfg = replace(cfg, interconnect=replace(cfg.interconnect, require_switch=False))
    assert _by_id(run_link_checks(_ctx(reference_inventory, cfg=cfg)))["L04"].status is Status.WARN


def test_degraded_gpu_slot_width(reference_inventory: Inventory) -> None:
    inv = with_gpu(reference_inventory, UUID_2060, link_width_current=4)
    assert _by_id(run_link_checks(_ctx(inv)))["L06"].status is Status.FAIL


def test_replays_fail(reference_inventory: Inventory) -> None:
    inv = with_gpu(reference_inventory, UUID_3050, replay_counter=17)
    r = _by_id(run_link_checks(_ctx(inv)))["L07"]
    assert r.status is Status.FAIL and "17" in r.detail


def test_kernel_log_findings(reference_inventory: Inventory) -> None:
    log = "pcieport 0000:02:00.0: AER: Corrected error received\n"
    assert _by_id(run_link_checks(_ctx(reference_inventory, log)))["L09"].status is Status.WARN
    log += "NVRM: Xid (PCI:0000:05:00): 79, GPU has fallen off the bus.\n"
    assert _by_id(run_link_checks(_ctx(reference_inventory, log)))["L09"].status is Status.FAIL


@pytest.mark.parametrize(
    ("changes", "status"),
    [
        ({"temperature_c": 82.0}, Status.WARN),
        ({"temperature_c": 90.0}, Status.FAIL),
        ({"active_clock_events": ("hw_thermal_slowdown",)}, Status.FAIL),
    ],
)
def test_thermals(
    reference_inventory: Inventory, changes: dict[str, object], status: Status
) -> None:
    inv = with_gpu(reference_inventory, UUID_2060, **changes)
    assert _by_id(run_link_checks(_ctx(inv)))["L10"].status is status


def test_crashing_check_is_contained(reference_inventory: Inventory) -> None:
    inv = replace(
        reference_inventory, gpus=(replace(reference_inventory.gpus[0], memory_total_bytes="x"),)
    )  # type: ignore[arg-type]
    results = run_link_checks(_ctx(inv))
    assert any(r.status is Status.FAIL and "crashed" in r.detail for r in results)
    assert len(results) == 13


# --------------------------------------------------------------------------- runtime


def _plan_dict(engine: str = "llamacpp") -> dict[str, object]:
    return {
        "engine": engine,
        "planned_share": [0.41, 0.59],
        "devices": [
            {
                "index": 0,
                "uuid": UUID_2060,
                "name": "RTX 2060",
                "baseline_used_bytes": 200 << 20,
                "capacity_bytes": int(6144 * 0.9) << 20,
            },
            {
                "index": 1,
                "uuid": UUID_3050,
                "name": "RTX 3050",
                "baseline_used_bytes": 200 << 20,
                "capacity_bytes": int(8192 * 0.9) << 20,
            },
        ],
    }


def _sampler(inv: Inventory, used_mib: tuple[int, int], replays: list[int] | None = None):  # type: ignore[no-untyped-def]
    state = {"n": 0}

    def sample() -> Inventory:
        state["n"] += 1
        out = with_gpu(inv, UUID_2060, memory_used_bytes=used_mib[0] << 20)
        out = with_gpu(out, UUID_3050, memory_used_bytes=used_mib[1] << 20)
        if replays:
            out = with_gpu(
                out, UUID_3050, replay_counter=replays[min(state["n"], len(replays)) - 1]
            )
        return out

    return sample


def _runtime(
    inv: Inventory,
    used: tuple[int, int],
    engine: str = "llamacpp",
    replays: list[int] | None = None,
):  # type: ignore[no-untyped-def]
    return _by_id(
        run_runtime_checks(
            CFG,
            _plan_dict(engine),
            _sampler(inv, used, replays),
            samples=3,
            interval_s=0,
            sleep=lambda s: None,
        )
    )


def test_runtime_split_matches(reference_inventory: Inventory) -> None:
    r = _runtime(reference_inventory, (200 + 2050, 200 + 2950))  # 41% : 59%
    assert all(x.status is Status.PASS for x in r.values()), r


def test_runtime_vllm_expects_capacity_ratio(reference_inventory: Inventory) -> None:
    # Concept doc success metric: ~5.4 GB on GPU 0 and ~7.2 GB on GPU 1.
    r = _runtime(reference_inventory, (200 + 5400, 200 + 7200), engine="vllm")
    assert r["R03"].status is Status.PASS


def test_runtime_detects_idle_gpu(reference_inventory: Inventory) -> None:
    r = _runtime(reference_inventory, (200, 200 + 5000))
    assert r["R02"].status is Status.FAIL
    assert r["R03"].status is Status.FAIL


def test_runtime_detects_skewed_split(reference_inventory: Inventory) -> None:
    r = _runtime(reference_inventory, (200 + 4000, 200 + 1000))
    assert r["R03"].status is Status.FAIL


def test_runtime_detects_replay_growth(reference_inventory: Inventory) -> None:
    r = _runtime(reference_inventory, (2250, 3150), replays=[0, 3, 9])
    assert r["R05"].status is Status.FAIL and "+9" in r["R05"].detail


def test_runtime_missing_gpu(reference_inventory: Inventory) -> None:
    inv = replace(reference_inventory, gpus=reference_inventory.gpus[:1])
    r = run_runtime_checks(CFG, _plan_dict(), lambda: inv, samples=1, sleep=lambda s: None)
    assert r[0].status is Status.FAIL and len(r) == 1


# ---------------------------------------------------------------------------- report


def test_report_renderers(reference_inventory: Inventory) -> None:
    rep = Report(node="astra-01", phase="link", results=run_link_checks(_ctx(reference_inventory)))
    rep.finished = rep.started + 1.5
    assert rep.status is Status.PASS and rep.exit_code == 0
    assert json.loads(rep.to_json())["counts"]["PASS"] == 12
    suite = ET.fromstring(rep.to_junit())
    assert suite.get("tests") == "13" and suite.get("failures") == "0"
    assert suite.get("skipped") == "1"  # L13 without bricks
    assert rep.to_markdown().count("\n| L") == 13
    assert "Result: PASS" in rep.to_table()


def test_report_status_precedence() -> None:
    from astra.validation.report import CheckResult

    rep = Report(
        "n",
        "link",
        [CheckResult("A", "a", Status.WARN, ""), CheckResult("B", "b", Status.SKIP, "")],
    )
    assert rep.status is Status.WARN and rep.exit_code == 0
    rep.results.append(CheckResult("C", "c", Status.FAIL, "boom"))
    assert rep.status is Status.FAIL and rep.exit_code == 1
    assert ET.fromstring(rep.to_junit()).find("testcase/failure") is not None

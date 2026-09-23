"""Acceptance checks.

Link gate (Phase 2, TC-L*) — run after assembly, on every boot, and in CI on the
bench: GPU inventory, PCIe switch/uplink, signal integrity, thermals.

Runtime gate (Phase 3, TC-R*) — run while an inference engine is serving: every
planned GPU carries its share of the model and nothing throttles or errors.

Requirement IDs refer to docs/01-requirements/PRD.md.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from astra.config import AstraConfig
from astra.errors import AstraError
from astra.hardware.compat import CompatReport, analyse
from astra.hardware.kernel_log import Severity, scan
from astra.hardware.models import GpuInfo, Inventory
from astra.units import MIB, fmt_bytes

from .report import CheckResult, Status

MIN_ENGINE_ALLOCATION = 256 * MIB


@dataclass(frozen=True)
class LinkContext:
    config: AstraConfig
    inventory: Inventory
    kernel_log: str | None  # None when it could not be read
    kernel_log_error: str | None = None
    topology_expected: bool = True  # False off-Linux: sysfs checks are skipped


def _r(cid: str, title: str, status: Status, detail: str, *reqs: str) -> CheckResult:
    return CheckResult(cid, title, status, detail, tuple(reqs))


# --------------------------------------------------------------------------- link gate


def check_driver(ctx: LinkContext) -> CheckResult:
    inv = ctx.inventory
    if not inv.gpus:
        return _r(
            "L01", "NVIDIA driver sees GPUs", Status.FAIL, "driver loaded but no GPUs", "FR-01"
        )
    return _r(
        "L01",
        "NVIDIA driver sees GPUs",
        Status.PASS,
        f"{len(inv.gpus)} GPU(s), driver {inv.driver_version}, CUDA {inv.cuda_version}",
        "FR-01",
    )


def _matches(gpu: GpuInfo, match: str, vram_mib: int | None) -> bool:
    if match.lower() not in gpu.name.lower():
        return False
    if vram_mib is None or gpu.memory_total_bytes is None:
        return True
    return abs(gpu.memory_total_bytes / MIB - vram_mib) <= 0.05 * vram_mib


def check_inventory(ctx: LinkContext) -> CheckResult:
    expected = ctx.config.node.expected_gpus
    title = "GPU inventory matches as-built"
    if not expected:
        return _r("L02", title, Status.SKIP, "no [[node.expected_gpu]] in config", "FR-01", "FR-02")
    remaining = list(ctx.inventory.gpus)
    missing = []
    for exp in expected:
        hit = next((g for g in remaining if _matches(g, exp.match, exp.vram_mib)), None)
        if hit is None:
            missing.append(exp.match + (f" {exp.vram_mib} MiB" if exp.vram_mib else ""))
        else:
            remaining.remove(hit)
    if missing:
        return _r("L02", title, Status.FAIL, "missing: " + ", ".join(missing), "FR-01", "FR-02")
    extra = f"; {len(remaining)} other GPU(s) (e.g. host display GPU)" if remaining else ""
    return _r(
        "L02",
        title,
        Status.PASS,
        f"all {len(expected)} expected GPU(s) present{extra}",
        "FR-01",
        "FR-02",
    )


def check_identity(ctx: LinkContext) -> CheckResult:
    uuids = [g.uuid for g in ctx.inventory.gpus]
    buses = [g.bus_id for g in ctx.inventory.gpus]
    title = "GPU identities unique"
    if len(set(uuids)) != len(uuids) or len(set(buses)) != len(buses):
        return _r("L03", title, Status.FAIL, "duplicate UUID or PCI bus id", "FR-01")
    return _r(
        "L03",
        title,
        Status.PASS,
        ", ".join(f"{g.index}:{g.bus_id}" for g in ctx.inventory.gpus),
        "FR-01",
    )


def _node_gpus(ctx: LinkContext) -> list[GpuInfo]:
    """GPUs that belong to the ASTRA chassis (behind the switch), or all if unknown."""
    vendors = ctx.config.interconnect.switch_vendor_ids
    behind = [
        g
        for g in ctx.inventory.gpus
        if g.uuid in ctx.inventory.paths and ctx.inventory.paths[g.uuid].switches(vendors)
    ]
    return behind or list(ctx.inventory.gpus)


def _switch_from_nvtopo(ctx: LinkContext) -> CheckResult | None:
    """Without sysfs (Windows), use nvidia-smi topo -m: PIX/PXB between every pair."""
    from astra.hardware.nvtopo import SWITCH_LINKS

    links = ctx.inventory.gpu_links
    if not links:
        return None
    title = "PCIe packet switch present"
    shared = sorted(k for k, v in links.items() if v in SWITCH_LINKS)
    if len(shared) == len(links):
        return _r(
            "L04",
            title,
            Status.PASS,
            f"all {len(ctx.inventory.gpus)} GPUs share a PCIe switch (nvidia-smi topo)",
            "FR-02",
        )
    kinds = ", ".join(f"GPU{i}-GPU{j}:{v}" for (i, j), v in sorted(links.items()))
    status = Status.FAIL if ctx.config.interconnect.require_switch else Status.PASS
    detail = f"not all GPUs behind one switch ({kinds}; nvidia-smi topo)"
    if not ctx.config.interconnect.require_switch:
        detail = f"no switch required by config ({kinds})"
    return _r("L04", title, status, detail, "FR-02")


def check_switch(ctx: LinkContext) -> CheckResult:
    title = "PCIe packet switch present"
    if not ctx.inventory.paths:
        fallback = _switch_from_nvtopo(ctx)
        if fallback is not None:
            return fallback
    if not ctx.topology_expected or not ctx.inventory.paths:
        return _r(
            "L04", title, Status.SKIP, "PCIe topology unavailable (needs Linux sysfs)", "FR-02"
        )
    vendors = ctx.config.interconnect.switch_vendor_ids
    upstream_ports = set()
    for gpu in ctx.inventory.gpus:
        path = ctx.inventory.paths.get(gpu.uuid)
        switches = path.switches(vendors) if path else ()
        if switches:
            upstream_ports.add(switches[0].bdf)
    if not upstream_ports:
        status = Status.FAIL if ctx.config.interconnect.require_switch else Status.WARN
        return _r(
            "L04",
            title,
            status,
            f"no bridge from vendor(s) {', '.join(vendors)} above any GPU",
            "FR-02",
        )
    if len(upstream_ports) > 1:
        return _r(
            "L04",
            title,
            Status.WARN,
            f"GPUs sit behind different switches: {', '.join(sorted(upstream_ports))}",
            "FR-02",
        )
    port = next(iter(upstream_ports))
    return _r(
        "L04",
        title,
        Status.PASS,
        f"{len(_node_gpus(ctx))} GPU(s) behind switch upstream port {port}",
        "FR-02",
    )


def check_uplink(ctx: LinkContext) -> CheckResult:
    title = "Host uplink bandwidth"
    ic = ctx.config.interconnect
    want = f"Gen{ic.expected_uplink_gen} x{ic.expected_uplink_width}"
    if not ctx.topology_expected or not ctx.inventory.paths:
        return _r(
            "L05",
            title,
            Status.SKIP,
            "PCIe topology unavailable (needs Linux sysfs)",
            "FR-02",
            "NFR-05",
        )
    worst = None
    for gpu in _node_gpus(ctx):
        path = ctx.inventory.paths.get(gpu.uuid)
        link = path.bottleneck if path else None
        if link and (worst is None or (link.bandwidth_gbps or 0) < (worst.bandwidth_gbps or 0)):
            worst = link
    if worst is None:
        return _r("L05", title, Status.SKIP, "no link speed attributes in sysfs", "FR-02", "NFR-05")
    got = (
        f"Gen{worst.current_gen} x{worst.current_width} at {worst.bdf} "
        f"(~{worst.bandwidth_gbps:.1f} GB/s)"
    )
    if (worst.current_width or 0) < ic.expected_uplink_width:
        return _r(
            "L05",
            title,
            Status.FAIL,
            f"{got}; expected {want} — check cable seating/adapter",
            "FR-02",
            "NFR-05",
        )
    if (worst.current_gen or 0) < ic.expected_uplink_gen:
        if (worst.max_gen or 0) >= ic.expected_uplink_gen:
            return _r(
                "L05",
                title,
                Status.WARN,
                f"{got}; capable of Gen{worst.max_gen} — idle link power saving, "
                "re-run while a GPU workload is active",
                "FR-02",
                "NFR-05",
            )
        return _r("L05", title, Status.FAIL, f"{got}; expected {want}", "FR-02", "NFR-05")
    return _r("L05", title, Status.PASS, got, "FR-02", "NFR-05")


def check_gpu_links(ctx: LinkContext) -> CheckResult:
    title = "GPU slot links at full width"
    degraded = [
        f"GPU {g.index} x{g.link_width_current}/x{g.link_width_max}"
        for g in ctx.inventory.gpus
        if g.link_width_current and g.link_width_max and g.link_width_current < g.link_width_max
    ]
    summary = ", ".join(
        f"GPU {g.index} Gen{g.link_gen_current}/{g.link_gen_max} x{g.link_width_current}"
        for g in ctx.inventory.gpus
    )
    if degraded:
        return _r("L06", title, Status.FAIL, "degraded: " + ", ".join(degraded), "FR-02")
    return _r("L06", title, Status.PASS, summary, "FR-02")


def check_replays(ctx: LinkContext) -> CheckResult:
    title = "PCIe replay counters"
    limit = ctx.config.validation.max_replay_count
    counts = {g.index: g.replay_counter for g in ctx.inventory.gpus}
    known = {i: c for i, c in counts.items() if c is not None}
    if not known:
        return _r("L07", title, Status.SKIP, "driver does not expose replay counters", "NFR-01")
    bad = {i: c for i, c in known.items() if c > limit}
    detail = ", ".join(f"GPU {i}: {c}" for i, c in known.items())
    if bad:
        return _r(
            "L07",
            title,
            Status.FAIL,
            f"{detail} (limit {limit}) — marginal signal integrity",
            "NFR-01",
        )
    return _r("L07", title, Status.PASS, detail, "NFR-01")


def check_aer(ctx: LinkContext) -> CheckResult:
    title = "PCIe AER error counters"
    if not ctx.topology_expected or not ctx.inventory.paths:
        return _r(
            "L08", title, Status.SKIP, "AER counters unavailable (needs Linux sysfs)", "NFR-01"
        )
    cor = sum(p.aer.correctable for p in ctx.inventory.paths.values())
    unc = sum(p.aer.nonfatal + p.aer.fatal for p in ctx.inventory.paths.values())
    detail = f"correctable {cor}, uncorrectable {unc} (summed over every hop)"
    if unc:
        return _r("L08", title, Status.FAIL, detail, "NFR-01")
    if cor:
        return _r("L08", title, Status.WARN, detail, "NFR-01")
    return _r("L08", title, Status.PASS, detail, "NFR-01")


def check_kernel_log(ctx: LinkContext) -> CheckResult:
    title = "Kernel log clean"
    if ctx.kernel_log is None:
        return _r(
            "L09",
            title,
            Status.SKIP,
            ctx.kernel_log_error or "kernel log not read",
            "NFR-01",
            "NFR-10",
        )
    findings = scan(ctx.kernel_log)
    if not findings:
        return _r(
            "L09",
            title,
            Status.PASS,
            "no AER / Xid / BAR / link-down messages this boot",
            "NFR-01",
            "NFR-10",
        )
    keys: dict[str, int] = {}
    for f in findings:
        keys[f.key] = keys.get(f.key, 0) + 1
    worst = Status.FAIL if any(f.severity is Severity.FAIL for f in findings) else Status.WARN
    hint = next(f.hint for f in findings if (f.severity is Severity.FAIL) == (worst is Status.FAIL))
    summary = ", ".join(f"{k}×{n}" for k, n in keys.items())
    return _r("L09", title, worst, f"{summary}. {hint}", "NFR-01", "NFR-10")


def check_thermals(ctx: LinkContext) -> CheckResult:
    th = ctx.config.thermal
    return _thermal_result(
        "L10", "Thermals and clock slowdowns", list(ctx.inventory.gpus), th.warn_c, th.crit_c
    )


def _thermal_result(
    cid: str, title: str, gpus: list[GpuInfo], warn_c: float, crit_c: float
) -> CheckResult:
    temps = {g.index: g.temperature_c for g in gpus if g.temperature_c is not None}
    slowdowns = {g.index: g.fault_slowdowns for g in gpus if g.fault_slowdowns}
    detail = ", ".join(f"GPU {i}: {t:.0f}°C" for i, t in temps.items()) or "no temperature data"
    if slowdowns:
        s = "; ".join(f"GPU {i}: {', '.join(r)}" for i, r in slowdowns.items())
        return _r(cid, title, Status.FAIL, f"{detail}; active slowdown — {s}", "NFR-02")
    hottest = max(temps.values(), default=0.0)
    if hottest >= crit_c:
        return _r(cid, title, Status.FAIL, f"{detail} (critical ≥ {crit_c:.0f}°C)", "NFR-02")
    if hottest >= warn_c:
        return _r(cid, title, Status.WARN, f"{detail} (warning ≥ {warn_c:.0f}°C)", "NFR-02")
    return _r(cid, title, Status.PASS, detail, "NFR-02")


def _compat(ctx: LinkContext) -> CompatReport:
    return analyse(
        ctx.inventory.gpus,
        ctx.inventory.driver_version,
        ctx.config.chassis,
        ctx.inventory.paths,
        ctx.config.interconnect.switch_vendor_ids,
    )


def check_compatibility(ctx: LinkContext) -> CheckResult:
    title = "GPU / driver compatibility"
    report = _compat(ctx)
    relevant = [
        f
        for f in report.findings
        if f.severity in ("fail", "warn")
        and f.code != "psu_undersized"
        and f.code != "power_unknown"
    ]
    archs = f"CUDA archs {report.cuda_architectures or 'n/a'}"
    if any(f.severity == "fail" for f in relevant):
        msg = "; ".join(f.message for f in relevant if f.severity == "fail")
        return _r("L11", title, Status.FAIL, msg, "FR-13")
    if relevant:
        return _r("L11", title, Status.WARN, "; ".join(f.message for f in relevant), "FR-13")
    return _r(
        "L11",
        title,
        Status.PASS,
        f"driver {ctx.inventory.driver_version} supports all GPUs; {archs}",
        "FR-13",
    )


def check_power_budget(ctx: LinkContext) -> CheckResult:
    title = "Chassis PSU budget"
    report = _compat(ctx)
    pw = report.power
    detail = (
        f"{len(report.chassis)} chassis GPU(s) [{report.chassis_basis}]: sustained "
        f"{pw.sustained_watts:.0f} W / limit {pw.sustained_limit_watts:.0f} W, peak "
        f"{pw.peak_watts:.0f} W / PSU {pw.psu_watts} W"
    )
    if not pw.ok:
        return _r(
            "L12",
            title,
            Status.FAIL,
            f"{detail} — fit a PSU of at least {pw.recommended_psu_watts} W",
            "FR-15",
            "NFR-03",
        )
    if pw.unknown_gpus:
        return _r(
            "L12",
            title,
            Status.WARN,
            f"{detail}; no power data for {', '.join(pw.unknown_gpus)} (75 W assumed)",
            "FR-15",
            "NFR-03",
        )
    return _r("L12", title, Status.PASS, detail, "FR-15", "NFR-03")


LINK_CHECKS: tuple[Callable[[LinkContext], CheckResult], ...] = (
    check_driver,
    check_inventory,
    check_identity,
    check_switch,
    check_uplink,
    check_gpu_links,
    check_replays,
    check_aer,
    check_kernel_log,
    check_thermals,
    check_compatibility,
    check_power_budget,
)


def run_link_checks(ctx: LinkContext) -> list[CheckResult]:
    results = []
    for check in LINK_CHECKS:
        try:
            results.append(check(ctx))
        except Exception as exc:  # one broken check must not hide the others
            name = check.__name__.removeprefix("check_")
            results.append(_r("L??", name, Status.FAIL, f"check crashed: {exc!r}"))
    return results


# ------------------------------------------------------------------------ runtime gate


def run_runtime_checks(
    config: AstraConfig,
    plan: dict[str, Any],
    sample: Callable[[], Inventory],
    samples: int | None = None,
    interval_s: float | None = None,
    sleep: Callable[[float], None] = time.sleep,
) -> list[CheckResult]:
    """Sample the node while the engine serves and compare against the saved plan."""
    samples = samples or config.validation.runtime_samples
    interval_s = config.validation.runtime_sample_interval_s if interval_s is None else interval_s
    all_devices = plan.get("devices", [])
    if not all_devices:
        raise AstraError("plan file has no devices; create it with `astra plan --output plan.json`")
    # GPUs on other machines are validated by `astra validate` on their own node.
    devices = [d for d in all_devices if not d.get("rpc_endpoint")]
    remote = len(all_devices) - len(devices)
    if not devices:
        return [
            _r(
                "R01",
                "Planned GPUs present",
                Status.SKIP,
                f"all {remote} planned GPU(s) are remote; run the runtime gate on their nodes",
                "FR-05",
            )
        ]

    snapshots: list[Inventory] = []
    for i in range(samples):
        snapshots.append(sample())
        if i + 1 < samples:
            sleep(interval_s)

    results: list[CheckResult] = []
    planned_uuids = [d["uuid"] for d in devices]
    last = snapshots[-1]
    missing = [u for u in planned_uuids if last.gpu(u) is None]
    if missing:
        results.append(
            _r(
                "R01",
                "Planned GPUs present",
                Status.FAIL,
                "missing: " + ", ".join(missing),
                "FR-05",
            )
        )
        return results
    remote_note = f" (+{remote} remote, checked on their nodes)" if remote else ""
    results.append(
        _r(
            "R01",
            "Planned GPUs present",
            Status.PASS,
            f"{len(planned_uuids)} GPU(s){remote_note}",
            "FR-05",
        )
    )

    peak: dict[str, int] = {u: 0 for u in planned_uuids}
    for snap in snapshots:
        for u in planned_uuids:
            gpu = snap.gpu(u)
            if gpu and gpu.memory_used_bytes is not None:
                peak[u] = max(peak[u], gpu.memory_used_bytes)
    delta = {d["uuid"]: max(0, peak[d["uuid"]] - d.get("baseline_used_bytes", 0)) for d in devices}

    idle = [d for d in devices if delta[d["uuid"]] < MIN_ENGINE_ALLOCATION]
    if idle:
        names = ", ".join(f"GPU {d['index']} {d['name']}" for d in idle)
        results.append(
            _r(
                "R02",
                "Engine allocated on every GPU",
                Status.FAIL,
                f"no engine allocation on {names} — is the engine running with this plan?",
                "FR-05",
            )
        )
    else:
        results.append(
            _r(
                "R02",
                "Engine allocated on every GPU",
                Status.PASS,
                ", ".join(f"GPU {d['index']}: +{fmt_bytes(delta[d['uuid']])}" for d in devices),
                "FR-05",
            )
        )

    total_delta = sum(delta.values()) or 1
    if plan.get("engine") == "vllm":
        # vLLM pre-allocates its whole budget on every GPU: expect the capacity ratio.
        cap_total = sum(d["capacity_bytes"] for d in devices)
        expected = {d["uuid"]: d["capacity_bytes"] / cap_total for d in devices}
    else:
        # Shares are over all planned GPUs; renormalise to the ones on this node.
        shares = dict(
            zip((d["uuid"] for d in all_devices), plan.get("planned_share", []), strict=False)
        )
        local_total = sum(shares.get(u, 0.0) for u in planned_uuids) or 1.0
        expected = {u: shares.get(u, 0.0) / local_total for u in planned_uuids}
    tol = config.validation.runtime_tolerance
    parts, off = [], False
    for d in devices:
        u = d["uuid"]
        actual = delta[u] / total_delta
        want = expected.get(u, 0.0)
        off = off or abs(actual - want) > tol
        parts.append(f"GPU {d['index']} {actual:.0%} (plan {want:.0%})")
    status = Status.FAIL if off or idle else Status.PASS
    results.append(
        _r(
            "R03",
            "Memory split matches plan",
            status,
            ", ".join(parts) + f"; tolerance ±{tol:.0%}",
            "FR-03",
            "FR-05",
        )
    )

    th = config.thermal
    severity = [Status.PASS, Status.SKIP, Status.WARN, Status.FAIL]
    worst: CheckResult | None = None
    for snap in snapshots:
        gpus = [g for u in planned_uuids if (g := snap.gpu(u)) is not None]
        result = _thermal_result("R04", "No throttling under load", gpus, th.warn_c, th.crit_c)
        if worst is None or severity.index(result.status) > severity.index(worst.status):
            worst = result
    assert worst is not None
    results.append(worst)

    first, final = snapshots[0], snapshots[-1]
    grew = []
    for u in planned_uuids:
        a, b = first.gpu(u), final.gpu(u)
        if (
            a
            and b
            and a.replay_counter is not None
            and b.replay_counter is not None
            and b.replay_counter > a.replay_counter
        ):
            grew.append(f"GPU {b.index} +{b.replay_counter - a.replay_counter}")
    results.append(
        _r(
            "R05",
            "No PCIe replays under load",
            Status.FAIL if grew else Status.PASS,
            ", ".join(grew) if grew else f"stable over {samples} samples",
            "NFR-01",
        )
    )
    return results

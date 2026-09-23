"""``astra`` command-line interface.

    astra probe                 inventory of GPUs and PCIe paths
    astra validate              link gate (Phase 2) — run after assembly and on every boot
    astra validate --phase runtime --plan plan.json
                                runtime gate (Phase 3) — run while the engine serves
    astra compat                what the installed GPUs can do: driver window, engines,
                                CUDA build archs, NVENC, chassis power budget
    astra plan ...              pick GPUs + heterogeneous VRAM split + engine launch command
    astra launch ...            plan, then exec the inference engine
    astra agent [--rpc]         fabric node: advertise GPUs, serve them to the cluster
    astra cluster               list fabric nodes and GPUs (UDP discovery or --peers)
    astra plan --cluster ...    plan across local + remote GPUs (llama.cpp RPC)
    astra ui                    web console: topology map, planner, health, chat
    astra exporter              Prometheus metrics endpoint
    astra gpus                  supported NVIDIA GPUs and support tiers
    astra models                built-in model catalog and quantizations

Exit codes: 0 success, 1 validation failed / plan does not fit, 2 usage or runtime error.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import platform
import subprocess
import sys
import time
from dataclasses import replace
from pathlib import Path
from typing import Any

from astra import __version__, pool
from astra.config import AstraConfig, load_config
from astra.errors import AstraError, CommandError
from astra.fabric.discovery import Peer, gather
from astra.hardware import kernel_log, sysfs
from astra.hardware.compat import CompatReport, analyse
from astra.hardware.models import GpuInfo, Inventory
from astra.hardware.probe import probe
from astra.hardware.runner import SubprocessRunner
from astra.planner import engines
from astra.planner.models import CATALOG, QUANTS, ModelProfile, from_catalog, from_gguf
from astra.planner.split import DeviceBudget, Plan, plan, select_gpus
from astra.units import MIB, fmt_bytes
from astra.validation.checks import LinkContext, run_link_checks, run_runtime_checks
from astra.validation.report import Report


def _simulated(spec: str) -> tuple[list[GpuInfo], dict[str, str]]:
    return pool.simulated(spec)


def _simulated_gpus(spec: str) -> list[GpuInfo]:
    return pool.simulated(spec)[0]


def _select(budgets: list[DeviceBudget], selector: str) -> list[DeviceBudget]:
    chosen = []
    for token in selector.split(","):
        token = token.strip()
        match = next((b for b in budgets if token in (str(b.index), b.uuid, b.bus_id)), None)
        if match is None:
            raise AstraError(f"--gpus: no GPU matches '{token}'")
        chosen.append(match)
    return chosen


def _profile(args: argparse.Namespace) -> ModelProfile:
    if args.gguf:
        return from_gguf(args.gguf, args.kv_type)
    if not args.model:
        raise AstraError("choose a model: --model <catalog key> or --gguf <file>")
    quant = args.quant or ("awq-int4" if args.engine == "vllm" else "q4_k_m")
    if quant in QUANTS and args.engine not in QUANTS[quant].engines:
        raise AstraError(f"quantization '{quant}' is not supported by {args.engine}")
    return from_catalog(args.model, quant, args.kv_type)


def _peers(args: argparse.Namespace, cfg: AstraConfig) -> list[Peer]:
    return pool.peers_for(cfg, getattr(args, "peers", None), getattr(args, "timeout", 1.5))


def _budgets(
    args: argparse.Namespace, cfg: AstraConfig, util: float, reserve: int
) -> tuple[list[DeviceBudget], list[str]]:
    result = pool.collect(
        cfg,
        util,
        reserve,
        simulate=args.simulate,
        fabric=bool(getattr(args, "cluster", False) or getattr(args, "peers", None)),
        peers=getattr(args, "peers", None),
        timeout_s=getattr(args, "timeout", 1.5),
    )
    return result.budgets, result.notes


def _make_plan(args: argparse.Namespace, cfg: AstraConfig) -> Plan:
    util = args.util if args.util is not None else cfg.planner.gpu_memory_utilization
    reserve = args.reserve_mib if args.reserve_mib is not None else cfg.planner.reserve_mib
    context = args.context or cfg.planner.default_context
    objective = args.objective or cfg.planner.objective
    hop = cfg.fabric.network_hop_ms / 1000.0
    model = _profile(args)
    budgets, warnings = _budgets(args, cfg, util, reserve)
    if not budgets:
        raise AstraError("no GPUs found locally or on the fabric")
    selector = (args.gpus or "auto").strip().lower()
    if selector == "auto":
        return select_gpus(
            model,
            budgets,
            args.engine,
            context,
            util,
            warnings,
            objective,
            cfg.planner.exclude_gpus,
            network_hop_s=hop,
        )
    chosen = budgets if selector == "all" else _select(budgets, args.gpus)
    return plan(model, chosen, args.engine, context, util, warnings, objective, network_hop_s=hop)


def _launch_spec(args: argparse.Namespace, result: Plan) -> engines.LaunchSpec:
    if result.engine == "llamacpp":
        model_path = args.gguf or "<path/to/model.gguf>"
        return engines.llamacpp(
            result, model_path, args.host, args.port or 8080, args.binary or "llama-server"
        )
    return engines.vllm(result, args.model_ref, args.host, args.port or 8000, args.binary or "vllm")


def _pct(values: tuple[float, ...]) -> str:
    return " : ".join(f"{v:.0%}" for v in values)


def _format_plan(result: Plan, spec: engines.LaunchSpec | None) -> str:
    m = result.model
    tps = result.est_decode_tokens_per_s
    speed = f"~{tps:.0f} tok/s single-stream (estimate)" if tps else "speed estimate n/a"
    out = [
        f"Plan: {m.name} via {result.engine}, context {result.context}, "
        f"objective {result.objective}",
        f"  weights {fmt_bytes(m.weights_bytes)}, {m.n_layers} layers, source {m.source}",
        f"  VRAM ratio: {_pct(result.vram_ratio)}   planned share: "
        f"{_pct(result.planned_share)}   {speed}",
        "",
        f"  {'GPU':<4}{'NAME':<30}{'LAYERS':<14}{'WEIGHTS':>12}{'KV':>12}{'FIXED':>12}"
        f"{'RESERVE':>12}{'REQUIRED':>12}{'CAPACITY':>12}{'HEADROOM':>12}",
    ]
    for p in result.placements:
        layers = f"{p.first_layer}-{p.first_layer + p.n_layers - 1} ({p.n_layers})"
        label = p.device.name.replace("NVIDIA GeForce ", "")
        if p.device.is_remote:
            label = f"@{p.device.node} {label}"
        out.append(
            f"  {p.device.index:<4}{label[:29]:<30}{layers:<14}"
            f"{fmt_bytes(p.weight_bytes):>12}"
            f"{fmt_bytes(p.kv_bytes):>12}{fmt_bytes(p.fixed_bytes):>12}{fmt_bytes(p.device.reserve_bytes):>12}"
            f"{fmt_bytes(p.required_bytes):>12}{fmt_bytes(p.device.capacity_bytes):>12}"
            f"{fmt_bytes(p.headroom_bytes):>12}"
        )
    out += [
        "",
        f"  Fits: {'YES' if result.fits else 'NO'}   "
        f"max context on this pool: {result.max_context}",
    ]
    for w in result.warnings:
        out.append(f"  ! {w}")
    if len(result.alternatives) > 1:
        out += ["", "  GPU sets considered (best first):"]
        for c in result.alternatives:
            est = f"~{c.est_decode_tokens_per_s:.0f} tok/s" if c.est_decode_tokens_per_s else "n/a"
            names = " + ".join(
                f"{i}:{n.replace('NVIDIA GeForce ', '')}"
                for i, n in zip(c.gpu_indices, c.names, strict=True)
            )
            fit = "fits" if c.fits else "no  "
            out.append(f"    {fit}  {est:>12}  mem {c.worst_utilisation:5.0%}  {names}")
    if spec is not None:
        out += ["", "Launch (bash):", f"  {spec.shell(posix=True)}"]
        out += ["Launch (PowerShell):", f"  {spec.shell(posix=False)}"]
        out += [f"  note: {n}" for n in spec.notes]
    return "\n".join(out)


def _format_inventory(inv: Inventory, cfg: AstraConfig) -> str:
    out = [
        f"ASTRA node {cfg.node.name} — driver {inv.driver_version}, "
        f"CUDA {inv.cuda_version}, {inv.platform}",
        "",
        f"  {'IDX':<4}{'NAME':<30}{'ARCH':<9}{'VRAM':>11}{'USED':>11}{'TEMP':>6}{'POWER':>9}"
        f"  {'GPU LINK cur/max':<18}{'BUS':<14}",
    ]
    for g in inv.gpus:
        link = (
            f"Gen{g.link_gen_current}/{g.link_gen_max} x{g.link_width_current}/{g.link_width_max}"
        )
        temp = f"{g.temperature_c:.0f}C" if g.temperature_c is not None else "n/a"
        power = f"{g.power_draw_w:.0f}W" if g.power_draw_w is not None else "n/a"
        out.append(
            f"  {g.index:<4}{g.name[:29]:<30}{(g.architecture or '?'):<9}"
            f"{fmt_bytes(g.memory_total_bytes):>11}"
            f"{fmt_bytes(g.memory_used_bytes):>11}{temp:>6}{power:>9}  {link:<18}{g.bus_id:<14}"
        )
    if inv.paths:
        out += ["", "  PCIe paths (root port -> GPU):"]
        for g in inv.gpus:
            path = inv.paths.get(g.uuid)
            if not path:
                continue
            hops = " -> ".join(
                f"{h.bdf}{'[PLX]' if (h.vendor_id or '').lower() == '0x10b5' else ''}"
                f"(Gen{h.current_gen} x{h.current_width})"
                for h in path.chain
            )
            bn = path.bottleneck
            tail = (
                f"; bottleneck {bn.bdf} ~{bn.bandwidth_gbps:.1f} GB/s"
                if bn and bn.bandwidth_gbps
                else ""
            )
            out.append(f"    GPU {g.index}: {hops}{tail}")
    elif platform.system() != "Linux":
        out += ["", "  (PCIe path / uplink detail requires Linux sysfs)"]
    return "\n".join(out)


def _write_or_print(text: str, output: str | None) -> None:
    if output:
        Path(output).write_text(text, encoding="utf-8")
        print(f"wrote {output}", file=sys.stderr)
    else:
        print(text)


# ---------------------------------------------------------------------------- commands


def _emit_config(inv: Inventory, cfg: AstraConfig) -> str:
    """astra.toml that freezes the currently installed chassis GPUs as the as-built node."""
    report = analyse(
        inv.gpus, inv.driver_version, cfg.chassis, inv.paths, cfg.interconnect.switch_vendor_ids
    )
    lines = [
        f"# generated by `astra probe --emit-config` ({report.chassis_basis})",
        "[node]",
        f'name = "{cfg.node.name}"',
        "",
    ]
    for c in report.chassis:
        model = c.spec.model if c.spec else c.gpu.name.replace("NVIDIA GeForce ", "")
        lines += ["[[node.expected_gpu]]", f'match = "{model}"']
        if c.gpu.memory_total_bytes:
            lines.append(f"vram_mib = {c.gpu.memory_total_bytes // MIB}")
        lines.append("")
    lines += [
        "[chassis]",
        f"psu_watts = {cfg.chassis.psu_watts}  # these GPUs need >= "
        f"{report.power.recommended_psu_watts} W",
        f"slots = {cfg.chassis.slots}",
        "",
    ]
    lines += _topology_config(inv, cfg, [c.gpu.uuid for c in report.chassis])
    return "\n".join(lines) + "\n"


def _topology_config(inv: Inventory, cfg: AstraConfig, uuids: list[str]) -> list[str]:
    """[interconnect] matching what is plugged in: switch or not, narrowest link."""
    paths = [inv.paths[u] for u in uuids if u in inv.paths]
    if not paths:
        return [
            "[interconnect]",
            "# PCIe topology not visible here (needs Linux sysfs); defaults kept.",
            f"require_switch = {str(cfg.interconnect.require_switch).lower()}",
        ]
    vendors = cfg.interconnect.switch_vendor_ids
    has_switch = any(p.switches(vendors) for p in paths)
    hops = [p.bottleneck for p in paths if p.bottleneck is not None]
    width = min((h.current_width or 16) for h in hops) if hops else 16
    # The link can only train as fast as the slowest device on it (e.g. a Gen3 switch).
    gens = [h.max_gen for p in paths for h in p.chain if h.max_gen]
    gen = min(gens) if gens else 3
    kind = "PCIe switch" if has_switch else "direct slots / bifurcation / adapters"
    return [
        "[interconnect]",
        f"# detected: {kind}; narrowest link Gen{gen} x{width}",
        f"require_switch = {str(has_switch).lower()}",
        f"expected_uplink_gen = {gen}",
        f"expected_uplink_width = {width}",
    ]


def _format_compat(report: CompatReport) -> str:
    lo, hi = report.required_driver_window
    out = [
        f"Driver {report.driver_version or 'n/a'}; these GPUs need a driver branch "
        f"R{lo} to {f'R{hi}' if hi else 'current'}; "
        f"CUDA build archs: {report.cuda_architectures or 'n/a'}",
        "",
        f"  {'IDX':<4}{'MODEL':<22}{'ARCH':<14}{'CC':>5}{'VRAM':>11}{'BW GB/s':>9}{'POWER':>8}"
        f"{'NVENC':>7}  {'ENGINES':<18}{'CHASSIS':<8}",
    ]
    for c in report.caps:
        model = c.spec.model if c.spec else c.gpu.name.replace("NVIDIA GeForce ", "")
        cc = f"{c.compute_capability:.1f}" if c.compute_capability else "?"
        bw = f"{c.mem_bandwidth_gbps:.0f}" if c.mem_bandwidth_gbps else "?"
        watts = f"{c.board_power_w:.0f}W" if c.board_power_w else "?"
        nvenc = {True: "yes", False: "no", None: "?"}[c.nvenc]
        out.append(
            f"  {c.gpu.index:<4}{model[:21]:<22}{c.architecture[:13]:<14}{cc:>5}"
            f"{fmt_bytes(c.gpu.memory_total_bytes):>11}{bw:>9}{watts:>8}{nvenc:>7}  "
            f"{', '.join(c.engines):<18}{'yes' if c in report.chassis else 'no':<8}"
        )
    pw = report.power
    out += [
        "",
        f"  Chassis power ({report.chassis_basis}): GPUs {pw.gpu_watts:.0f} W + overhead "
        f"{pw.overhead_watts} W = {pw.sustained_watts:.0f} W sustained, "
        f"{pw.peak_watts:.0f} W peak",
        f"  PSU {pw.psu_watts} W (sustained limit {pw.sustained_limit_watts:.0f} W): "
        f"{'OK' if pw.ok else 'TOO SMALL'}; recommended >= {pw.recommended_psu_watts} W",
    ]
    if report.findings:
        out.append("")
        out += [f"  [{f.severity.upper():<4}] {f.message}" for f in report.findings]
    return "\n".join(out)


def cmd_compat(args: argparse.Namespace, cfg: AstraConfig) -> int:
    chassis = cfg.chassis
    if args.psu_watts:
        chassis = replace(chassis, psu_watts=args.psu_watts)
    if args.simulate:
        gpus, driver, paths = _simulated_gpus(args.simulate), args.driver, None
    else:
        inv = probe()
        gpus, driver, paths = list(inv.gpus), args.driver or inv.driver_version, inv.paths
    report = analyse(gpus, driver, chassis, paths, cfg.interconnect.switch_vendor_ids)
    print(json.dumps(report.to_dict(), indent=2) if args.json else _format_compat(report))
    return 1 if report.worst == "fail" else 0


def cmd_probe(args: argparse.Namespace, cfg: AstraConfig) -> int:
    inv = probe()
    if args.emit_config:
        print(_emit_config(inv, cfg), end="")
    elif args.json:
        print(json.dumps(inv.to_dict(), indent=2))
    else:
        print(_format_inventory(inv, cfg))
    return 0


def cmd_validate(args: argparse.Namespace, cfg: AstraConfig) -> int:
    report = Report(node=cfg.node.name, phase=args.phase)
    if args.phase == "link":
        runner = SubprocessRunner()
        inv = probe(runner)
        log_text, log_err = None, None
        if platform.system() == "Linux":
            try:
                log_text = kernel_log.read_kernel_log(runner)
            except CommandError as exc:
                log_err = str(exc)
        else:
            log_err = "kernel log scan is Linux-only"
        ctx = LinkContext(
            cfg, inv, log_text, log_err, topology_expected=sysfs.LocalSysfs().available()
        )
        report.results = run_link_checks(ctx)
    else:
        if not args.plan:
            raise AstraError(
                "--phase runtime needs --plan <plan.json> (from `astra plan --output`)"
            )
        plan_data: dict[str, Any] = json.loads(Path(args.plan).read_text(encoding="utf-8"))
        report.results = run_runtime_checks(
            cfg, plan_data, lambda: probe(with_topology=False), samples=args.samples
        )
    report.finished = time.time()
    _write_or_print(report.render(args.format), args.output)
    if args.output and args.format != "table":
        print(report.to_table(), file=sys.stderr)
    return report.exit_code


def cmd_plan(args: argparse.Namespace, cfg: AstraConfig) -> int:
    result = _make_plan(args, cfg)
    spec = _launch_spec(args, result)
    if args.json or args.output:
        data = result.to_dict()
        data["launch"] = {"argv": list(spec.argv), "env": spec.env, "notes": list(spec.notes)}
        text = json.dumps(data, indent=2)
        if args.output:
            _write_or_print(text, args.output)
            print(_format_plan(result, spec), file=sys.stderr)
        else:
            print(text)
    else:
        print(_format_plan(result, spec))
    if args.env_file:
        Path(args.env_file).write_text(compose_env(result, args), encoding="utf-8")
        print(f"wrote {args.env_file}", file=sys.stderr)
    return 0 if result.fits else 1


def compose_env(result: Plan, args: argparse.Namespace) -> str:
    """KEY=VALUE lines consumed by deploy/compose/compose.yaml (see docs/06-deployment)."""
    model = Path(args.gguf).name if args.gguf else (args.model_ref or result.model.hf_repo or "")
    values = {
        "ASTRA_ENGINE": result.engine,
        "ASTRA_GPU_UUIDS": ",".join(p.device.uuid for p in result.placements),
        "ASTRA_LAYER_SPLIT": ",".join(str(n) for n in result.layer_counts),
        "ASTRA_PIPELINE_STAGES": str(len(result.placements)),
        "ASTRA_N_GPU_LAYERS": str(result.model.n_layers + 1),
        "ASTRA_CONTEXT": str(result.context),
        "ASTRA_GPU_UTIL": f"{result.gpu_memory_utilization:.2f}",
        "ASTRA_MODEL": model,
    }
    header = f"# generated by `astra plan` for {result.model.name}; fits={result.fits}\n"
    return header + "".join(f"{k}={v}\n" for k, v in values.items())


def cmd_launch(args: argparse.Namespace, cfg: AstraConfig) -> int:
    if args.simulate:
        raise AstraError("launch runs on real hardware; drop --simulate")
    result = _make_plan(args, cfg)
    spec = _launch_spec(args, result)
    print(_format_plan(result, spec), file=sys.stderr)
    if not result.fits:
        print("refusing to launch: plan does not fit (see above)", file=sys.stderr)
        return 1
    if result.engine == "llamacpp" and not args.gguf:
        raise AstraError("llama.cpp launch needs --gguf <file>")
    if args.dry_run:
        return 0
    env = {**os.environ, **spec.env}
    if os.name == "posix":
        os.execvpe(
            spec.argv[0], list(spec.argv), env
        )  # replace process: clean signals under systemd
    return subprocess.call(list(spec.argv), env=env)


def cmd_agent(args: argparse.Namespace, cfg: AstraConfig) -> int:
    from astra.fabric.agent import RpcSupervisor, rpc_server_specs, serve_agent
    from astra.fabric.protocol import describe

    fab = cfg.fabric
    node = fab.node_name or cfg.node.name
    port = args.port or fab.agent_port
    bind = args.bind or fab.bind
    supervisor, rpc_ports = None, {}
    if args.rpc:
        from astra.fabric.agent import find_rpc_server

        binary = find_rpc_server(args.rpc_binary or fab.rpc_binary)
        if binary is None:
            raise AstraError(
                f"'{args.rpc_binary or fab.rpc_binary}' not found (nor ggml-rpc-server): "
                "build llama.cpp with -DGGML_CUDA=ON -DGGML_RPC=ON and put it on PATH, "
                "or pass --rpc-binary"
            )
        specs = rpc_server_specs(
            probe(with_topology=False).gpus,
            binary,
            bind,
            fab.rpc_port_base,
        )
        rpc_ports = {s.uuid: s.port for s in specs}
        supervisor = RpcSupervisor(specs)
        for s in specs:
            print(f"serving GPU {s.uuid} on rpc port {s.port}", file=sys.stderr)
    serve_agent(
        lambda: describe(node, port, probe(with_topology=False), rpc_ports),
        node,
        bind,
        port,
        None if args.no_discovery else fab.discovery_port,
        supervisor,
    )
    return 0


def cmd_cluster(args: argparse.Namespace, cfg: AstraConfig) -> int:
    nodes, errors = gather(_peers(args, cfg))
    if args.json:
        doc = {"nodes": [{"host": p.host, **d.to_dict()} for p, d in nodes], "errors": errors}
        print(json.dumps(doc, indent=2))
        return 0 if nodes else 1
    print(f"ASTRA fabric: {len(nodes)} node(s)")
    for peer, desc in nodes:
        print(
            f"\n  {desc.node}  ({peer.api_url}, astra {desc.version}, driver {desc.driver_version})"
        )
        for g in desc.gpus:
            served = f"rpc {peer.host}:{g.rpc_port}" if g.rpc_port else "not served (no --rpc)"
            name = g.name.replace("NVIDIA GeForce ", "")
            print(
                f"    {name:<22}{fmt_bytes(g.memory_total_bytes):>11}  "
                f"{(g.architecture or '?'):<13}{served}"
            )
    for e in errors:
        print(f"  ! {e}")
    if not nodes and not errors:
        print("  no agents answered; start `astra agent --rpc` on each machine or pass --peers")
    return 0 if nodes else 1


def cmd_ui(args: argparse.Namespace, cfg: AstraConfig) -> int:
    from astra.ui.server import UiState, serve

    state = UiState(
        cfg,
        simulate=args.simulate,
        fabric=args.cluster,
        peers=args.peers,
        engine_url=args.engine_url.rstrip("/"),
    )
    if args.plan:
        state.active_plan = json.loads(Path(args.plan).read_text(encoding="utf-8"))
    serve(state, args.host, args.port)
    return 0


def cmd_exporter(args: argparse.Namespace, cfg: AstraConfig) -> int:
    from astra.telemetry.exporter import Collector, serve

    collector = Collector(lambda: probe(), cfg)
    serve(collector, args.host or cfg.telemetry.listen, args.port or cfg.telemetry.port)
    return 0


def gpu_support_rows() -> list[dict[str, Any]]:
    """One row per catalog model (VRAM variants merged), newest architecture first."""
    from astra.hardware.compat import ENGINE_MIN_CC
    from astra.hardware.gpu_specs import ARCHITECTURES, SPECS, arch_info, support_tier

    order = list(ARCHITECTURES)
    rows: dict[str, dict[str, Any]] = {}
    for s in SPECS:
        arch = arch_info(s.architecture)
        assert arch is not None
        row = rows.setdefault(
            s.model,
            {
                "model": s.model,
                "architecture": arch.name,
                "cc": arch.compute_capability,
                "vram_gb": [],
                "bandwidth_gbps": [],
                "board_power_w": [],
                "nvenc": s.nvenc,
                "lanes": s.pcie_lanes,
                "engines": [e for e, cc in ENGINE_MIN_CC.items() if arch.compute_capability >= cc],
                "tier": support_tier(arch),
                "max_driver": arch.last_driver_branch,
            },
        )
        row["vram_gb"].append(round(s.vram_mib / 1024))
        row["bandwidth_gbps"].append(s.mem_bandwidth_gbps)
        row["board_power_w"].append(s.board_power_w)
    return sorted(
        rows.values(),
        key=lambda r: (-order.index(r["architecture"].lower()), r["model"]),
    )


def _span(values: list[float], unit: str = "") -> str:
    lo, hi = min(values), max(values)
    return f"{lo:g}{unit}" if lo == hi else f"{lo:g}/{hi:g}{unit}"


def gpu_support_markdown() -> str:
    out = [
        "| Model | Architecture | cc | VRAM (GB) | Mem BW (GB/s) | Board power (W) | NVENC "
        "| PCIe | Engines | Support |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for r in gpu_support_rows():
        out.append(
            f"| {r['model']} | {r['architecture']} | {r['cc']:.1f} | "
            f"{'/'.join(str(v) for v in sorted(set(r['vram_gb'])))} | "
            f"{_span(r['bandwidth_gbps'])} | {_span(r['board_power_w'])} | "
            f"{'yes' if r['nvenc'] else '**no**'} | x{r['lanes']} | "
            f"{', '.join(r['engines'])} | {r['tier']} |"
        )
    return "\n".join(out) + "\n"


def cmd_gpus(args: argparse.Namespace, cfg: AstraConfig) -> int:
    rows = gpu_support_rows()
    if args.markdown:
        print(gpu_support_markdown(), end="")
        return 0
    if args.json:
        print(json.dumps(rows, indent=2))
        return 0
    arch = None
    for r in rows:
        if r["architecture"] != arch:
            arch = r["architecture"]
            print(f"\n{arch} (cc {r['cc']:.1f}) - {r['tier']}")
        vram = "/".join(str(v) for v in sorted(set(r["vram_gb"])))
        print(
            f"  {r['model']:<20} {vram + ' GB':<9} {_span(r['bandwidth_gbps'], ' GB/s'):<14}"
            f"{_span(r['board_power_w'], ' W'):<10}{'NVENC' if r['nvenc'] else 'no NVENC':<10}"
            f"x{r['lanes']:<4}{', '.join(r['engines'])}"
        )
    print(
        "\nNot listed but NVIDIA: works through its architecture (no speed estimate)."
        "\nKepler (GTX 6xx/7xx) and older: unsupported."
    )
    return 0


def cmd_models(args: argparse.Namespace, cfg: AstraConfig) -> int:
    print("Models (use with --model):")
    for key, a in CATALOG.items():
        print(f"  {key:<14} {a.display:<28} {a.n_params / 1e9:5.1f}B params, {a.n_layers} layers")
    print("\nQuantizations (use with --quant):")
    for key, q in QUANTS.items():
        print(f"  {key:<10} ~{q.layer_bits:.2f} bits/weight   engines: {', '.join(q.engines)}")
    return 0


# ------------------------------------------------------------------------------ parser


def _add_plan_args(p: argparse.ArgumentParser) -> None:
    src = p.add_argument_group("model")
    src.add_argument("--model", choices=sorted(CATALOG), help="catalog model (estimate)")
    src.add_argument("--quant", choices=list(QUANTS), help="quantization for --model")
    src.add_argument("--gguf", help="GGUF file: exact per-layer sizes (llama.cpp)")
    src.add_argument(
        "--model-ref", help="Hugging Face id/path passed to vLLM (default: catalog repo)"
    )
    p.add_argument("--engine", choices=["llamacpp", "vllm"], default="llamacpp")
    p.add_argument("--context", type=int, help="context length in tokens")
    p.add_argument(
        "--kv-type", choices=["f16", "q8_0", "q4_0"], default="f16", help="KV cache dtype"
    )
    p.add_argument(
        "--gpus",
        help="'auto' (default: fastest GPU set that fits), 'all', or a comma list of "
        "index/UUID/bus id in pipeline order",
    )
    p.add_argument(
        "--objective",
        choices=["speed", "balanced"],
        help="speed: favour fast GPUs; balanced: equal memory pressure (default from config)",
    )
    p.add_argument("--util", type=float, help="fraction of each GPU the engine may use")
    p.add_argument("--reserve-mib", type=int, help="per-GPU reserve for CUDA context/buffers")
    p.add_argument(
        "--simulate",
        metavar="SPEC",
        help="hypothetical GPUs: a preset (" + ", ".join(pool.SIMULATION_PRESETS) + ") or a list "
        "like 'RTX 3060, GT 1030, 2x GTX 1660 SUPER' (name[:mib[:arch]])",
    )
    p.add_argument("--host", default="127.0.0.1", help="engine listen address (0.0.0.0 exposes it)")
    p.add_argument("--port", type=int)
    p.add_argument("--binary", help="engine executable (default llama-server / vllm)")
    fab = p.add_argument_group("fabric (GPUs on other machines, ADR-0011)")
    fab.add_argument("--cluster", action="store_true", help="include GPUs from discovered agents")
    fab.add_argument("--peers", help="agents to use instead of discovery: 'host[:port],...'")
    fab.add_argument("--timeout", type=float, default=1.5, help="discovery wait in seconds")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="astra", description="ASTRA external GPU node control plane"
    )
    parser.add_argument(
        "--config", help="path to astra.toml (default: ./astra.toml, /etc/astra/astra.toml)"
    )
    parser.add_argument("-v", "--verbose", action="store_true")
    parser.add_argument("--version", action="version", version=f"astra {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("probe", help="show GPUs and PCIe paths")
    p.add_argument("--json", action="store_true")
    p.add_argument(
        "--emit-config",
        action="store_true",
        help="print an astra.toml that freezes the installed GPUs as the as-built node",
    )
    p.set_defaults(func=cmd_probe)

    p = sub.add_parser("compat", help="GPU/driver compatibility and chassis power budget")
    p.add_argument("--simulate", metavar="SPEC", help="hypothetical GPUs (see `plan --help`)")
    p.add_argument("--driver", help="driver version to evaluate against, e.g. 580.95.05")
    p.add_argument("--psu-watts", type=int, help="override chassis.psu_watts")
    p.add_argument("--json", action="store_true")
    p.set_defaults(func=cmd_compat)

    p = sub.add_parser("validate", help="run acceptance checks")
    p.add_argument("--phase", choices=["link", "runtime"], default="link")
    p.add_argument("--plan", help="plan JSON for --phase runtime")
    p.add_argument("--samples", type=int, help="runtime samples (default from config)")
    p.add_argument("--format", choices=["table", "json", "md", "junit"], default="table")
    p.add_argument("--output", help="write the report to a file")
    p.set_defaults(func=cmd_validate)

    p = sub.add_parser("plan", help="compute a VRAM-pooled split and launch command")
    _add_plan_args(p)
    p.add_argument("--json", action="store_true")
    p.add_argument("--output", help="write plan JSON (input for validate --phase runtime)")
    p.add_argument("--env-file", help="write a docker compose .env for the planned split")
    p.set_defaults(func=cmd_plan)

    p = sub.add_parser("launch", help="plan and start the inference engine")
    _add_plan_args(p)
    p.add_argument("--dry-run", action="store_true")
    p.set_defaults(func=cmd_launch)

    p = sub.add_parser("agent", help="fabric node agent: advertise GPUs, serve them over RPC")
    p.add_argument("--rpc", action="store_true", help="run one llama.cpp rpc-server per GPU")
    p.add_argument("--rpc-binary", help="rpc-server executable (default from [fabric])")
    p.add_argument("--bind", help="address to listen on (default from [fabric])")
    p.add_argument("--port", type=int, help="agent API port (default 9837)")
    p.add_argument("--no-discovery", action="store_true", help="do not answer UDP discovery")
    p.set_defaults(func=cmd_agent)

    p = sub.add_parser("cluster", help="list fabric nodes and their GPUs")
    p.add_argument("--peers", help="'host[:port],...' instead of UDP discovery")
    p.add_argument("--timeout", type=float, default=1.5)
    p.add_argument("--json", action="store_true")
    p.set_defaults(func=cmd_cluster)

    p = sub.add_parser("ui", help="web console: topology, plans, health and chat")
    p.add_argument("--host", default="127.0.0.1", help="listen address (default: localhost only)")
    p.add_argument("--port", type=int, default=9838)
    p.add_argument("--simulate", metavar="SPEC", help="hypothetical GPUs, e.g. fabric-demo")
    p.add_argument("--cluster", action="store_true", help="include GPUs from fabric agents")
    p.add_argument("--peers", help="fabric agents 'host[:port],...' instead of discovery")
    p.add_argument(
        "--engine-url",
        default="http://127.0.0.1:8080",
        help="OpenAI-compatible engine the Chat tab talks to",
    )
    p.add_argument("--plan", help="plan JSON to show as the active plan")
    p.set_defaults(func=cmd_ui)

    p = sub.add_parser("exporter", help="serve Prometheus metrics")
    p.add_argument("--host")
    p.add_argument("--port", type=int)
    p.set_defaults(func=cmd_exporter)

    p = sub.add_parser("gpus", help="list supported NVIDIA GPUs and their support tier")
    p.add_argument("--markdown", action="store_true", help="table for docs")
    p.add_argument("--json", action="store_true")
    p.set_defaults(func=cmd_gpus)

    p = sub.add_parser("models", help="list catalog models and quantizations")
    p.set_defaults(func=cmd_models)
    return parser


def main(argv: list[str] | None = None) -> int:
    for stream in (sys.stdout, sys.stderr):
        # Reports contain non-ASCII (°C, ±); keep them intact when piped on Windows (cp1252).
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8")
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    try:
        cfg = load_config(args.config)
        return int(args.func(args, cfg))
    except AstraError as exc:
        print(f"astra: error: {exc}", file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        return 130

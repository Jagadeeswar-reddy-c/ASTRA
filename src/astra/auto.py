"""``astra auto``: one command from bare hardware to a verified, serving GPU pool (ADR-0013).

Nothing has to be declared by hand. It works out how many GPUs there are, which ones,
and how they are connected:

1. **Detect**: every NVIDIA GPU via ``nvidia-smi -q -x`` (model, VRAM, PCIe link),
   GPU-to-GPU topology via ``nvidia-smi topo -m``, and ASTRA agents on the LAN.
2. **Configure**: write the as-built ``astra.toml`` (per-user location) the first
   time; when the hardware changes later, write ``astra.toml.new`` and say so.
3. **Check**: driver window, engine support, PSU budget (``astra compat``).
4. **Recommend**: the largest catalog model and best quantization that fits the
   detected pool at the target context with margin and a usable speed.
5. **Prepare**: download llama.cpp (the right CUDA build for these GPUs) and the
   model's GGUF if missing.
6. **Launch and verify**: start llama-server with the planned command, benchmark it,
   run the runtime gate, then serve the web console until Ctrl+C.
"""

from __future__ import annotations

import contextlib
import json
import os
import re
import shutil
import socket
import subprocess
import threading
import time
import urllib.request
import zipfile
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from astra import asbuilt, pool
from astra.config import AstraConfig, user_config_path
from astra.errors import AstraError, CommandError, PlanningError
from astra.hardware.compat import CompatReport, GpuCaps, analyse
from astra.hardware.models import Inventory
from astra.hardware.probe import probe
from astra.planner import engines
from astra.planner.models import (
    CATALOG,
    DRAFT_QUANT,
    QUANTS,
    ModelProfile,
    from_catalog,
    from_gguf,
    gguf_download,
    gguf_files,
)
from astra.planner.split import SPEC_MAX_TPS, DeviceBudget, Plan, select_gpus
from astra.units import MIB, fmt_bytes
from astra.validation.checks import run_runtime_checks
from astra.validation.report import Status

AUTO_QUANTS = ("q8_0", "q6_k", "q5_k_m", "q4_k_m")
AUTO_KV = ("f16", "q8_0")  # a q8_0 KV cache halves its memory for ~1 % speed (FT-SPEC-01)
DRAFT_KEEP_GAIN = 1.05  # keep speculative decoding only if it measures >= 5 % faster
PREFILL_PROMPT = ("The quick brown fox jumps over the lazy dog. " * 170).strip()  # ~1.7k tokens
MIN_HEADROOM = 256 * MIB  # recommendations keep this much free on every GPU
RELEASES_API = "https://api.github.com/repos/ggml-org/llama.cpp/releases?per_page=10"

Echo = Callable[[str], None]


# ------------------------------------------------------------------------ recommend


@dataclass(frozen=True)
class Recommendation:
    model: str
    quant: str
    params_b: float
    plan: Plan
    kv: str = "f16"
    draft: str | None = None  # catalog key of the speculative-decoding draft

    @property
    def tokens_per_s(self) -> float | None:
        return self.plan.est_decode_tokens_per_s

    @property
    def gpus(self) -> str:
        return " + ".join(
            (f"@{p.device.node} " if p.device.is_remote else "")
            + p.device.name.replace("NVIDIA GeForce ", "")
            for p in self.plan.placements
        )


def recommend(
    budgets: list[DeviceBudget],
    cfg: AstraConfig,
    context: int,
    min_tps: float = 8.0,
    models: Sequence[str] | None = None,
    quants: Sequence[str] = AUTO_QUANTS,
    use_draft: str = "auto",
) -> list[Recommendation]:
    """Every downloadable catalog model/quant that fits with margin, best first.

    Best = most parameters (a larger model at Q4 beats a smaller one at Q8), then the
    higher-precision quantization, then speed. A q8_0 KV cache is used only when the
    f16 cache does not fit. Pools slower than SPEC_MAX_TPS get the model's draft for
    speculative decoding when it fits too (kept at launch only if it measures faster);
    ``use_draft='on'`` tries it at any speed, ``'off'`` never.
    """
    util = cfg.planner.gpu_memory_utilization
    hop = cfg.fabric.network_hop_ms / 1000

    def attempt(profile: ModelProfile, draft: ModelProfile | None = None) -> Plan | None:
        try:
            p = select_gpus(
                profile,
                budgets,
                "llamacpp",
                context,
                util,
                [],
                cfg.planner.objective,
                cfg.planner.exclude_gpus,
                network_hop_s=hop,
                draft=draft,
            )
        except PlanningError:
            return None
        if not p.fits or min(pl.headroom_bytes for pl in p.placements) < MIN_HEADROOM:
            return None
        return p

    out = []
    for key in models or CATALOG:
        for quant in quants:
            if gguf_download(key, quant) is None:
                continue
            for kv in AUTO_KV:
                p = attempt(from_catalog(key, quant, kv))
                if p is not None:
                    break
            if p is None:
                continue
            tps = p.est_decode_tokens_per_s
            if tps is not None and tps < min_tps:
                continue
            draft_key = CATALOG[key].draft
            slow = tps is not None and tps <= SPEC_MAX_TPS
            if draft_key and (use_draft == "on" or (use_draft == "auto" and slow)):
                with_draft = attempt(
                    from_catalog(key, quant, kv), from_catalog(draft_key, DRAFT_QUANT, kv)
                )
                if with_draft is not None and with_draft.draft is not None:
                    out.append(
                        Recommendation(
                            key, quant, CATALOG[key].n_params / 1e9, with_draft, kv, draft_key
                        )
                    )
                    continue
            out.append(Recommendation(key, quant, CATALOG[key].n_params / 1e9, p, kv))
    out.sort(
        key=lambda r: (
            -r.params_b,
            -QUANTS[r.quant].layer_bits,
            r.kv != "f16",
            -(r.tokens_per_s or 0),
        )
    )
    return out


# --------------------------------------------------------------- engine build choice


def pick_cuda_major(caps: Sequence[GpuCaps]) -> int:
    """CUDA major version of the llama.cpp build that runs on all these GPUs.

    Prebuilt CUDA 12 builds don't target Blackwell (sm_120); CUDA 13 no longer
    targets Maxwell/Pascal/Volta (sm < 75).
    """
    ccs = [c.compute_capability for c in caps if c.compute_capability]
    blackwell = any(cc >= 12.0 for cc in ccs)
    legacy = any(cc < 7.5 for cc in ccs)
    if blackwell and legacy:
        raise AstraError(
            "no prebuilt llama.cpp covers Blackwell and pre-Turing GPUs together: build it "
            "from source with CUDA 12.8+ and CMAKE_CUDA_ARCHITECTURES from `astra compat`, "
            "then pass --binary"
        )
    return 13 if blackwell else 12


def _download(url: str, dest: Path, echo: Echo) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_suffix(dest.suffix + ".part")
    req = urllib.request.Request(url, headers={"User-Agent": "astra-auto"})
    with urllib.request.urlopen(req, timeout=60) as resp, part.open("wb") as fh:
        total = int(resp.headers.get("Content-Length") or 0)
        done, next_report = 0, 0.0
        while chunk := resp.read(1 << 20):
            fh.write(chunk)
            done += len(chunk)
            if total and done / total >= next_report:
                echo(f"      {done / total:4.0%}  {fmt_bytes(done)} / {fmt_bytes(total)}")
                next_report += 0.2
    part.replace(dest)


def find_llama_server(lab: Path, binary: str | None) -> str | None:
    if binary:
        return shutil.which(binary) or (binary if Path(binary).is_file() else None)
    exe = "llama-server.exe" if os.name == "nt" else "llama-server"
    local = lab / "llama" / exe
    if local.is_file():
        return str(local)
    return shutil.which("llama-server")


def ensure_llama_server(
    lab: Path,
    caps: Sequence[GpuCaps],
    binary: str | None,
    echo: Echo,
    windows: bool = os.name == "nt",
) -> str:
    found = find_llama_server(lab, binary)
    if found:
        return found
    if not windows:
        raise AstraError(
            "llama-server not found. On Linux build llama.cpp with -DGGML_CUDA=ON -DGGML_RPC=ON "
            "(see docs/06-deployment) or use the Compose 'pooled' profile, then pass --binary"
        )
    major = pick_cuda_major(caps)
    with urllib.request.urlopen(RELEASES_API, timeout=60) as resp:
        releases = json.loads(resp.read())
    pattern = re.compile(rf"^llama-b\d+-bin-win-cuda-{major}\.\d+-x64\.zip$")
    for release in releases:
        assets = {a["name"]: a["browser_download_url"] for a in release.get("assets", [])}
        main = next((n for n in assets if pattern.match(n)), None)
        if not main:
            continue
        cuda_ver = main.split("cuda-")[1].split("-x64")[0]
        echo(f"    llama.cpp {release['tag_name']} (Windows, CUDA {cuda_ver})")
        for name in (main, f"cudart-llama-bin-win-cuda-{cuda_ver}-x64.zip"):
            if name in assets:
                zpath = lab / "dl" / name
                _download(assets[name], zpath, echo)
                with zipfile.ZipFile(zpath) as zf:
                    zf.extractall(lab / "llama")
        for exe in ("llama-server.exe", "llama-server"):
            if (lab / "llama" / exe).is_file():
                return str(lab / "llama" / exe)
    raise AstraError(f"no Windows CUDA {major}.x llama.cpp build found in the latest releases")


def ensure_model(lab: Path, model: str, quant: str, echo: Echo) -> Path:
    """Download every part of a catalog GGUF that is missing; return the file to load."""
    files = gguf_files(model, quant)
    if not files:
        raise AstraError(f"no download source for {model} {quant}")
    for name, url in files:
        path = lab / "models" / name
        if not path.is_file():
            echo(f"    downloading {name}")
            _download(url, path, echo)
    return lab / "models" / files[0][0]


# ---------------------------------------------------------------------------- config


def sync_config(inv: Inventory, cfg: AstraConfig, echo: Echo) -> Path:
    """Write the as-built config the first time; never silently overwrite a user's file."""
    rendered = asbuilt.render_config(inv, cfg)
    target = cfg.source or user_config_path()
    if not target.exists():
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(rendered, encoding="utf-8")
        echo(f"    created {target}")
        return target
    report = analyse(
        inv.gpus, inv.driver_version, cfg.chassis, inv.paths, cfg.interconnect.switch_vendor_ids
    )
    detected = sorted(
        (c.spec.model if c.spec else c.gpu.name, (c.gpu.memory_total_bytes or 0) // MIB)
        for c in report.chassis
    )
    expected = sorted((e.match, e.vram_mib or 0) for e in cfg.node.expected_gpus)
    if detected == expected or not expected:
        echo(f"    {target} matches the installed GPUs")
        return target
    new = target.with_name(target.name + ".new")
    new.write_text(rendered, encoding="utf-8")
    echo(f"    ! installed GPUs differ from {target}: expected {expected}, found {detected}")
    echo(f"      wrote {new}; review it and replace {target.name} to accept the new hardware")
    return target


# ----------------------------------------------------------------------------- run


@dataclass
class AutoOptions:
    lab: Path
    context: int
    model: str | None = None
    quant: str | None = None
    min_tps: float = 8.0
    binary: str | None = None
    port: int = 8080
    ui_port: int = 9838
    cluster: bool = True
    launch: bool = True
    ui: bool = True
    draft: str = "auto"  # auto | on | off


def _post(url: str, body: dict[str, Any], timeout: float = 300) -> dict[str, Any]:
    req = urllib.request.Request(
        url, json.dumps(body).encode(), {"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        data: dict[str, Any] = json.loads(resp.read())
        return data


def _wait_healthy(
    url: str, proc: subprocess.Popen[bytes], log: Path, timeout_s: float = 300
) -> None:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if proc.poll() is not None:
            tail = log.read_text(encoding="utf-8", errors="replace").splitlines()[-12:]
            raise AstraError("llama-server exited while loading:\n      " + "\n      ".join(tail))
        try:
            with urllib.request.urlopen(f"{url}/health", timeout=2) as r:
                if json.loads(r.read()).get("status") == "ok":
                    return
        except (OSError, ValueError):
            pass
        time.sleep(1.5)
    raise AstraError(f"llama-server not healthy after {timeout_s:.0f} s (log: {log})")


def _port_in_use(port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.settimeout(0.5)
        return s.connect_ex(("127.0.0.1", port)) == 0


def _summary_line(cap: GpuCaps) -> str:
    g = cap.gpu
    link = (
        f"PCIe Gen{g.link_gen_max} x{g.link_width_current}"
        if g.link_width_current
        else "PCIe link unknown"
    )
    model = cap.spec.model if cap.spec else g.name
    return (
        f"GPU{g.index}  {model:<18} {fmt_bytes(g.memory_total_bytes):>10}  "
        f"{cap.architecture:<13} {link}"
    )


def run(cfg: AstraConfig, opts: AutoOptions, echo: Echo = print) -> int:
    started = time.strftime("%Y%m%d-%H%M%S")
    out = opts.lab / f"auto-{started}"

    # 1. detect ---------------------------------------------------------------
    echo("[1/6] Detecting GPUs (nvidia-smi) and the network fabric")
    try:
        inv = probe()
    except CommandError as exc:
        echo(f"    no NVIDIA GPU found ({exc}).")
        echo(
            "    Install the NVIDIA driver (Windows: nvidia.com/drivers; Ubuntu: "
            "sudo ubuntu-drivers install), reboot, and check that `nvidia-smi` lists your GPU."
        )
        return 1
    if not inv.gpus:
        echo(
            "    the NVIDIA driver is loaded but reports no GPUs; "
            "check that the card is seated and powered"
        )
        return 1
    report: CompatReport = analyse(
        inv.gpus, inv.driver_version, cfg.chassis, inv.paths, cfg.interconnect.switch_vendor_ids
    )
    echo(f"    {len(inv.gpus)} NVIDIA GPU(s), driver {inv.driver_version}, {inv.platform}")
    for cap in report.caps:
        echo("    " + _summary_line(cap))
    if inv.gpu_links:
        from astra.hardware.nvtopo import describe

        for (i, j), link in sorted(inv.gpu_links.items()):
            echo(f"    GPU{i}<->GPU{j}: {link} ({describe(link)})")
    util, reserve = cfg.planner.gpu_memory_utilization, cfg.planner.reserve_mib
    the_pool = pool.collect(cfg, util, reserve, fabric=opts.cluster, timeout_s=1.5)
    remote = [n for n in the_pool.nodes if n.role == "worker"]
    for node in remote:
        echo(
            f"    + node {node.name} ({node.host}): "
            + ", ".join(g.name.replace("NVIDIA GeForce ", "") for g, _ in node.gpus)
        )
    if opts.cluster and not remote:
        echo("    (no other ASTRA machines found on the network)")

    # 2. configure ------------------------------------------------------------
    echo("[2/6] Configuration")
    sync_config(inv, cfg, echo)

    # 3. check ----------------------------------------------------------------
    echo("[3/6] Compatibility and power")
    lo, hi = report.required_driver_window
    echo(
        f"    driver branch needed: R{lo} to {f'R{hi}' if hi else 'current'}; "
        f"CUDA archs {report.cuda_architectures or '?'}; "
        f"PSU ok for {report.power.sustained_watts:.0f} W sustained: "
        f"{'yes' if report.power.ok else 'NO'}"
    )
    for f in report.findings:
        if f.severity in ("fail", "warn"):
            echo(f"    [{f.severity.upper()}] {f.message}")
    if report.worst == "fail":
        echo("    stopping: fix the FAIL findings above first")
        return 1

    # 4. recommend ------------------------------------------------------------
    echo(f"[4/6] Choosing a model for this pool (context {opts.context})")
    models = [opts.model] if opts.model else None
    quants = [opts.quant] if opts.quant else AUTO_QUANTS
    recs = recommend(
        the_pool.budgets, cfg, opts.context, opts.min_tps, models, quants, use_draft=opts.draft
    )
    if not recs:
        echo("    nothing in the catalog fits with margin; try a smaller --context")
        return 1
    for r in recs[:6]:
        tps = f"~{r.tokens_per_s:.0f} tok/s" if r.tokens_per_s else "speed n/a"
        extras = "".join(
            [f", KV {r.kv}" if r.kv != "f16" else "", f", draft {r.draft}" if r.draft else ""]
        )
        echo(f"    {r.model:<13} {r.quant:<7} {tps:>12}  on {r.gpus}{extras}")
    best = recs[0]
    echo(f"    -> {CATALOG[best.model].display} {best.quant.upper()}")
    if not opts.launch:
        return 0

    # 5. prepare --------------------------------------------------------------
    echo("[5/6] Preparing llama.cpp and the model")
    server = ensure_llama_server(opts.lab, report.caps, opts.binary, echo)
    gguf = ensure_model(opts.lab, best.model, best.quant, echo)
    draft_gguf = ensure_model(opts.lab, best.draft, DRAFT_QUANT, echo) if best.draft else None
    echo(f"    engine {server}")
    echo(f"    model  {gguf}")
    if draft_gguf:
        echo(f"    draft  {draft_gguf}")
    profile = from_gguf(gguf, best.kv)
    draft_profile = from_gguf(draft_gguf, best.kv) if draft_gguf else None

    def exact_plan(draft: ModelProfile | None) -> Plan:
        return select_gpus(
            profile,
            the_pool.budgets,
            "llamacpp",
            opts.context,
            util,
            the_pool.notes,
            cfg.planner.objective,
            cfg.planner.exclude_gpus,
            network_hop_s=cfg.fabric.network_hop_ms / 1000,
            draft=draft,
        )

    base = exact_plan(None)
    if not base.fits:
        raise AstraError("the downloaded model does not fit after exact sizing; lower --context")
    variants: list[Plan] = [base]
    if draft_profile is not None:
        with_draft = exact_plan(draft_profile)
        if with_draft.fits and with_draft.draft is not None:
            variants.append(with_draft)
    style = engines.spec_flag_style(_help_text(server))
    out.mkdir(parents=True, exist_ok=True)

    # 6. launch + verify ------------------------------------------------------
    echo("[6/6] Launching and verifying")
    if _port_in_use(opts.port):
        raise AstraError(
            f"port {opts.port} is already in use (another engine?): stop it or pass --port"
        )
    url = f"http://127.0.0.1:{opts.port}"
    runs: list[tuple[Plan, dict[str, Any], Bench]] = []
    proc: subprocess.Popen[bytes] | None = None
    try:
        for i, variant in enumerate(variants):
            spec = engines.llamacpp(
                variant,
                str(gguf),
                "127.0.0.1",
                opts.port,
                server,
                draft_path=str(draft_gguf) if variant.draft else None,
                spec_style=style,
            )
            doc = variant.to_dict()
            doc["launch"] = {"argv": list(spec.argv), "env": spec.env, "notes": list(spec.notes)}
            label = "with speculative decoding" if variant.draft else "baseline"
            if proc is not None:
                _stop(proc)
            proc = _start(spec, out / f"llama-server-{i}.log", url)
            bench = _bench(url)
            echo(
                f"    {label:<26} {bench.decode_tps:6.1f} tok/s, "
                f"time to first token for 2k prompt ~{bench.ttft_2k_s:.2f} s"
                + (f", draft acceptance {bench.acceptance:.0%}" if bench.acceptance else "")
            )
            runs.append((variant, doc, bench))
        final, plan_doc, bench = runs[0]
        if len(runs) > 1 and runs[1][2].decode_tps >= runs[0][2].decode_tps * DRAFT_KEEP_GAIN:
            final, plan_doc, bench = runs[1]
            echo(
                f"    keeping speculative decoding "
                f"(+{(runs[1][2].decode_tps / runs[0][2].decode_tps - 1) * 100:.0f} %)"
            )
        elif len(runs) > 1:
            echo("    speculative decoding did not pay off here; back to the baseline")
            assert proc is not None
            _stop(proc)
            base_spec = plan_doc["launch"]
            proc = _start(
                engines.LaunchSpec(tuple(base_spec["argv"]), base_spec["env"]),
                out / "llama-server.log",
                url,
            )
        (out / "plan.json").write_text(json.dumps(plan_doc, indent=2), encoding="utf-8")
        echo(f"    engine healthy at {url}")
        measured = bench.decode_tps
        estimate = final.est_decode_tokens_per_s or 0
        if final.draft is None:
            delta = (measured - estimate) / estimate * 100 if estimate else 0.0
            echo(
                f"    speed: {measured:.1f} tok/s measured vs ~{estimate:.0f} estimated "
                f"({delta:+.0f} %)"
            )
        else:
            # The estimate is for plain decoding; compare the baseline run against it.
            base_measured = runs[0][2].decode_tps
            delta = (base_measured - estimate) / estimate * 100 if estimate else 0.0
            echo(
                f"    speed: {measured:.1f} tok/s with speculative decoding; baseline "
                f"{base_measured:.1f} vs ~{estimate:.0f} estimated ({delta:+.0f} %)"
            )

        def background_load() -> None:
            # Keeps the GPU busy while the gate samples; the engine may be stopped mid-request.
            with contextlib.suppress(OSError):
                _post(
                    f"{url}/completion",
                    {"prompt": "Write a long essay about GPUs.", "n_predict": 800},
                )

        load = threading.Thread(target=background_load, daemon=True)
        load.start()
        time.sleep(2)
        checks = run_runtime_checks(
            cfg, plan_doc, lambda: probe(with_topology=False), samples=6, interval_s=0.5
        )
        gate_ok = all(c.status is not Status.FAIL for c in checks)
        echo("    runtime gate: " + ", ".join(f"{c.id} {c.status.value}" for c in checks))
        result = {
            "gpus": len(inv.gpus),
            "remote_nodes": [n.name for n in remote],
            "model": profile.name,
            "quant": best.quant,
            "kv_cache": best.kv,
            "draft": final.draft.name if final.draft else None,
            "context": opts.context,
            "layer_counts": list(final.layer_counts),
            "estimate_tok_s": round(estimate, 1),
            "measured_tok_s": round(measured, 1),
            "delta_percent": round(delta, 1),
            "prefill_tok_s": round(bench.prefill_tps, 1),
            "ttft_2k_prompt_s": round(bench.ttft_2k_s, 2),
            "runs": [{"draft": v.draft is not None, **b.to_dict()} for v, _, b in runs],
            "runtime_gate": "PASS" if gate_ok else "FAIL",
        }
        (out / "results.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
        verdict = "PASS" if gate_ok and abs(delta) <= 30 else "FAIL"
        echo(f"    RESULT: {verdict}  (details in {out})")

        if opts.ui:
            from astra.ui.server import UiState, serve

            echo(
                f"    console: http://127.0.0.1:{opts.ui_port}/  "
                "(Ctrl+C stops the engine and console)"
            )
            serve(
                UiState(cfg, fabric=bool(remote), engine_url=url, active_plan=plan_doc),
                "127.0.0.1",
                opts.ui_port,
            )
        return 0 if verdict == "PASS" else 1
    finally:
        if proc is not None:
            _stop(proc)


@dataclass(frozen=True)
class Bench:
    decode_tps: float
    prefill_tps: float
    acceptance: float | None  # accepted / drafted tokens, with speculative decoding

    @property
    def ttft_2k_s(self) -> float:
        """Time to first token for a 2,000-token prompt."""
        return 2000 / self.prefill_tps if self.prefill_tps else 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "decode_tok_s": round(self.decode_tps, 1),
            "prefill_tok_s": round(self.prefill_tps, 1),
            "draft_acceptance": None if self.acceptance is None else round(self.acceptance, 2),
        }


BENCH_PROMPTS = (
    "Explain how several GPUs can share one model.",
    "Write a Python function that parses a CSV file and returns the average of each column.",
)


def _bench(url: str) -> Bench:
    """Decode speed over two different prompts, prefill speed over a ~1.7k-token prompt."""
    speeds, drafted, accepted = [], 0, 0
    for prompt in BENCH_PROMPTS:
        t = _post(f"{url}/completion", {"prompt": prompt, "n_predict": 200, "cache_prompt": False})[
            "timings"
        ]
        speeds.append(float(t["predicted_per_second"]))
        drafted += int(t.get("draft_n") or 0)
        accepted += int(t.get("draft_n_accepted") or 0)
    t = _post(
        f"{url}/completion", {"prompt": PREFILL_PROMPT, "n_predict": 1, "cache_prompt": False}
    )["timings"]
    return Bench(
        sum(speeds) / len(speeds),
        float(t.get("prompt_per_second") or 0.0),
        accepted / drafted if drafted else None,
    )


def _help_text(server: str) -> str:
    try:
        done = subprocess.run(
            [server, "--help"], capture_output=True, text=True, timeout=30, errors="replace"
        )
    except (OSError, subprocess.SubprocessError):
        return ""
    return done.stdout + done.stderr


def _start(spec: engines.LaunchSpec, log: Path, url: str) -> subprocess.Popen[bytes]:
    with log.open("wb") as log_fh:
        proc = subprocess.Popen(
            list(spec.argv), env={**os.environ, **spec.env}, stdout=log_fh, stderr=subprocess.STDOUT
        )
    try:
        _wait_healthy(url, proc, log)
    except BaseException:
        _stop(proc)
        raise
    return proc


def _stop(proc: subprocess.Popen[bytes]) -> None:
    if proc.poll() is None:
        proc.terminate()
        try:
            proc.wait(timeout=15)
        except subprocess.TimeoutExpired:
            proc.kill()


def default_lab() -> Path:
    return Path(os.environ.get("ASTRA_LAB") or Path.home() / "astra-lab")

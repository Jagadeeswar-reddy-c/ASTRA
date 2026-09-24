# Changelog

All notable changes are recorded here. Format: [Keep a Changelog](https://keepachangelog.com/en/1.1.0/);
versioning: [SemVer](https://semver.org/).

## [Unreleased]

## [0.7.0] — 2026-09-24 — ASTRA Stack bricks in software (CR-006 S1-S3, S5)

### Added
- `[[module]]` config: one ASTRA Stack brick per entry (its GPUs by UUID, bus id or name,
  its own PSU, expected link). `astra probe --emit-config --stack` writes them.
- Power budget per brick: `astra compat` lists bricks; L12 checks every brick's PSU;
  exporter `astra_module_{psu,power_limit,power,rated_power}_watts`,
  `astra_module_missing_gpus`, `module` label on `astra_gpu_info`; alerts
  `AstraModulePowerHigh`, `AstraModuleGpuMissing`; runbook RB-08.
- Link check L13: every brick present, on its expected Gen/width, one switch level
  (star, ADR-0014).
- Console: GPUs labelled with their brick on the map and in the GPU table.
- Deployment guide §12, TC-SW-29, TC-HW-13, FR-21 traced.

### Fixed
- Config files written by Windows PowerShell 5.1 (`astra probe --emit-config > astra.toml`
  is UTF-16; `Out-File` adds a BOM) now load.

## [0.6.0] — 2026-09-24 — Bigger models, faster answers, stack sizing (CR-006)

Measured on the RTX 3060 Ti: docs/05-testing/reports/performance-field-test.md.

### Added
- Catalog: Qwen2.5-32B, Llama-3.3-70B (split GGUF parts downloaded and read together),
  Qwen2.5-0.5B/1.5B and Llama-3.2-1B; each large model names its draft model.
- Speculative decoding (ADR-0015): `astra plan --draft auto|<model>|--draft-gguf`; the draft
  sits on the first local GPU and its memory is planned (within 3 % of measured).
  `astra auto --draft auto|on|off` benchmarks with and without the draft and keeps it only
  if it is >= 5 % faster (+21-38 % measured on slow pools, -15 % on a fast single GPU).
- `astra size --model M --min-tps N [--have ...]`: which GPUs / ASTRA Stack bricks reach a
  speed target (FR-22, CR-006 S4).
- `astra auto` reports prefill speed and time to first token; picks a q8_0 KV cache only
  when the f16 cache does not fit.
- ADR-0014 (ASTRA Stack bricks, CR-006 approved), ADR-0015, FR-21-23, TC-SW-26-28, TC-RT-12.

### Fixed
- `--kv-type q8_0/q4_0` was planned but never passed to llama-server or the Compose stack
  (now `--cache-type-k/v` with `--flash-attn on`; `ASTRA_KV_TYPE`).
- A fabric CLI test failed on machines with a real GPU (stubbed the wrong probe).

### Added
- docs/01-requirements/limitations-and-backlog.md: known limitations (hard limits and
  fixable ones) and the ordered backlog BL-01…BL-19.
- ASTRA Stack study (ASTRA-ARC-004) and CR-006 (proposed): DGX Spark-style stackable
  1-GPU bricks on a PCIe star, compared with DGX Spark; risk R-15.

## [0.5.1] — 2026-09-24

New-user acceptance test PASS from a fresh clone (report: docs/05-testing/reports/new-user-test.md).

### Added
- docs/GETTING-STARTED.md: new-user guide (install, run, check, use via the OpenAI API, run
  permanently, add GPUs, update/uninstall, troubleshooting).
- `scripts/install.ps1` / `scripts/install.sh`: one-step install (+ `astra auto`).

### Changed
- `astra auto` explains a missing NVIDIA driver and refuses to start on a busy port.
- `stage0-test.ps1` downloads atomically (`.part` then rename); platform shows Windows 11.

## [0.5.0] — 2026-09-24 — CR-005: zero-input automation

### Added
- `astra auto`: detects every NVIDIA GPU (`nvidia-smi -q -x`, `nvidia-smi topo -m`) and
  ASTRA agents on the LAN, writes the as-built config (per-user location; `.new` file on
  hardware changes), checks compatibility/power, recommends the best model that fits,
  downloads llama.cpp (CUDA 12/13 chosen from the GPUs) and the GGUF, launches,
  benchmarks, runs the runtime gate and serves the console. Verified on an RTX 3060 Ti:
  Qwen2.5-7B Q4_K_M, 74.4 tok/s vs 73 estimated, gate 5/5.
- `hardware/nvtopo.py`: GPU-to-GPU topology (PIX/PXB/PHB/NODE/SYS/NVLink) on any OS;
  shown by `astra probe`, used by L04 and `--emit-config` on Windows.
- Verified GGUF download sources for every catalog model × Q4_K_M/Q5_K_M/Q6_K/Q8_0.
- Per-user config location (`%APPDATA%\astra\astra.toml`, `~/.config/astra/astra.toml`).
- ADR-0013, CR-005, FR-20, TC-SW-25, TC-RT-10.

## [0.4.3] — 2026-09-24

### Added
- `scripts/stage0-test.ps1`: one-command field test on Windows (downloads llama.cpp +
  model, plans, starts the engine, measures speed, runtime gate, PASS/FAIL report).
  Verified on an RTX 3060 Ti: 139.7 tok/s measured vs 149.6 estimated, gate 5/5.
- docs/03-solution-design/connecting-gpus.md: every way to connect GPUs (x16/x8/x4/x1,
  risers, M.2, OCuLink, bifurcation, switch, Thunderbolt, network), bandwidth needs,
  example mixed builds, the single-PSU rule, step-by-step install and troubleshooting.
- Console GPU table and fabric descriptors show each GPU's PCIe link (Gen/width).
- `astra probe --emit-config` records the detected topology (switch or not, narrowest link).

## [0.4.2] — 2026-09-24 — Stage-0 field test fixes

Field test on real hardware (RTX 3060 Ti, llama.cpp b11149): planner speed within
2–7 %, runtime gate 5/5, fabric over llama.cpp RPC working end-to-end.
Report: docs/05-testing/reports/stage0-field-test.md.

### Fixed
- FT-01: fabric launch used `RPC[host:port]` device names; llama.cpp uses `RPC0, RPC1, …`.
- FT-02: agent finds the renamed `ggml-rpc-server` binary automatically.
- FT-03: generated engine commands bind to 127.0.0.1 by default (was 0.0.0.0).
- FT-04: console form shows plans made from a GGUF file.

## [0.4.1] — 2026-09-23

### Added
- `astra gpus [--markdown|--json]`: supported NVIDIA GPUs with support tiers
  (recommended / supported ≤ R580 / best effort / unsupported).
- docs/03-solution-design/supported-gpus.md (generated table, kept in sync by a unit test).
- docs/05-testing/field-test-plan.md: staged real-hardware testing (today's PC →
  two PCs → two GPUs in one desktop → the ASTRA node) with head-PC specs and a shopping list.

### Changed
- Dev tooling pinned (ruff 0.16.8, mypy 2.3.1) so CI matches developer machines.

## [0.4.0] — 2026-09-23 — CR-004: web console

### Added
- `astra ui`: exo-style web console (ADR-0012). Topology map of machines and GPUs with
  the active plan's pipeline stages and layer ranges; live GPU table; Plan, Chat
  (streamed through a proxy) and Health tabs; light/dark; `--simulate`, `--cluster`,
  `--peers`, `--engine-url`, `--plan`. Self-contained page, localhost by default.
- `astra.pool`: shared pool collection for the CLI and console; `fabric-demo` preset.
- Fabric descriptors carry live temperature, utilisation and power.
- `astra-ui.service`; deployment guide §11; README screenshot.

### Fixed
- Speed estimate: a remote first/last stage costs a network hop (the head sends the
  input and reads the logits), and remote→remote boundaries cost two (llama.cpp RPC
  relays via the head). Plans now prefer local GPUs at the pipeline ends.

## [0.3.0] — 2026-09-23 — CR-003: ASTRA Fabric

### Added
- ASTRA Fabric (ADR-0011): exo-style pooling of NVIDIA GPUs across machines.
  - `astra agent [--rpc]`: node agent (HTTP `/v1/node`, UDP discovery responder) that
    supervises one llama.cpp `rpc-server` per GPU, pinned by UUID, with restart and `--cache`.
  - `astra cluster [--peers]`: discover and list fabric nodes and GPUs.
  - `astra plan|launch --cluster|--peers`: plan across local + remote GPUs; network-hop
    cost per machine boundary; GPU-less head node; generated `llama-server --rpc --device`.
  - `--simulate "RTX 3050, @desktop RTX 3060 Ti"` for remote GPUs.
  - `[fabric]` config, `astra-agent.service`, firewall guidance (NFR-12).
- vMerge feasibility study (ASTRA-ARC-003) and CR-003.

### Changed
- The runtime gate validates only the GPUs of the node it runs on; remote GPUs are
  checked on their own nodes.

## [0.2.0] — 2026-09-23 — CR-001: any NVIDIA GPU mix

### Added
- GPU capability catalog (`hardware/gpu_specs.py`): GeForce GT/GTX/RTX from Maxwell to
  Blackwell (bandwidth, board power, NVENC, lanes), plus architecture facts.
- Compatibility analysis (`hardware/compat.py`) and `astra compat [--simulate] [--driver]`:
  driver-branch window (Pascal and older end at R580, Kepler at R470), engine eligibility,
  CUDA build architectures, NVENC, slot count, PSU budget with a recommended size.
- Link-gate checks L11 (GPU/driver compatibility) and L12 (chassis PSU budget).
- Planner: `speed` objective (bandwidth-aware placement with a 92 % fill cap) and
  automatic GPU-set selection (`--gpus auto`, default) with ranked alternatives;
  `--objective`, `planner.objective`, `planner.exclude_gpus`.
- `--simulate` accepts plain model names (`"RTX 3060, GT 1030, 2x GTX 1660 SUPER"`) and
  presets (`reference`, `budget-mix`, `mixed-gen`).
- `astra probe --emit-config` writes the as-built `[[node.expected_gpu]]` list.
- Exporter: `astra_chassis_{psu,power_limit,power,rated_power}_watts`, plus
  compute capability and chassis membership on `astra_gpu_info`.
  Alert `AstraChassisUndersized`; `AstraChassisPowerHigh` is now relative to the PSU.
- Ray resources `gpu_cc_<nn>` and `nvenc`.
- Docs: CR-001, ADR-0009, ADR-0010, design review DR-16…20, PRD v1.1 (FR-13…15),
  power-budget formula, runbook RB-07, deployment guide §8a, TC-SW-16…19, TC-HW-12.

### Changed
- `[chassis]` config section (psu_watts, slots, overhead, transient factor, sustained ratio).
- The vLLM dtype decision uses compute capability (< 8.0 → float16) instead of names.

## [0.1.0] — 2026-09-23

### Added
- Governance: project charter, RACI, phase gates G0–G7, risk register, gate log.
- Requirements: PRD with FR-01…12 and NFR-01…11.
- Architecture: system architecture, design review of the concept document
  (DR-01…15), ADR-0001…0008.
- Solution design: hardware design (power budget, sequencing, grounding, BIOS,
  assembly SOP), software design, corrected BOM.
- `astra` control plane:
  - `probe`: GPU inventory via `nvidia-smi -q -x` (R535 and R610 schemas) and PCIe
    path/uplink bottleneck/AER via sysfs.
  - `validate`: link gate L01–L10 and runtime gate R01–R05, with table, JSON,
    Markdown and JUnit output.
  - `plan` / `launch`: min-max pipeline split across heterogeneous GPUs from the
    model catalog or exact GGUF tensor tables; llama.cpp and vLLM launch specs; UUID
    pinning; simulation mode; Compose `.env` output.
  - `exporter`: Prometheus metrics with a rate-limited collector and `/healthz`.
  - Optional Ray typed-resource pool.
- Deployment: container image, Compose profiles (pooled / vllm / partitioned),
  Prometheus alert rules, Grafana dashboard, systemd units (self-test-gated
  inference), host provisioning script, CI pipeline.
- Test strategy, test cases (TC-SW/HW/RT), traceability matrix, monitoring guide and
  runbooks.

# ASTRA — Test Cases

Result columns are filled in during execution: **P** = pass, **F** = fail (with a
defect link), **B** = blocked.

## A. Software (automated — CI)

| ID | Title | Automated by | Requirements | Result (v0.1.0) |
|---|---|---|---|---|
| TC-SW-01 | nvidia-smi XML parsing: current (R610) and legacy (R535) schemas, malformed input | `tests/unit/test_hardware.py::test_parse_*`, `test_query_*` | FR-01 | P |
| TC-SW-02 | PCIe chain walk, uplink bottleneck, switch detection, AER sum | `test_hardware.py::test_gpu_path_*`, `test_aer_*`, `test_probe_*` | FR-01, FR-02 | P |
| TC-SW-03 | Kernel-log signatures (Xid, AER, BAR, link down); journalctl→dmesg fallback | `test_hardware.py::test_scan_*`, `test_read_kernel_log_*` | FR-02 | P |
| TC-SW-04 | Link gate L01–L10: healthy node passes; each fault fails or warns as specified | `tests/unit/test_validation.py` (link section) | FR-02, NFR-01/02/05 | P |
| TC-SW-05 | Runtime gate R01–R05: split match (llama.cpp and vLLM), idle GPU, skew, replay growth, missing GPU | `test_validation.py` (runtime section) | FR-05 | P |
| TC-SW-06 | Planner: concept VRAM ratio; contiguous split; fit/no-fit; 14B needs the pool; single-GPU hint; max context | `tests/unit/test_planner.py` | FR-03, FR-11 | P |
| TC-SW-07 | GGUF reader: header, tensor sizes, mixed quant blocks, tied embeddings, rejects bad files | `test_planner.py::test_gguf_*` | FR-12 | P |
| TC-SW-08 | Launch specs: llama.cpp layer split; vLLM PP (not TP), partition env, fp16, AWQ; UUID env | `test_planner.py::test_*_launch*` | FR-04 | P |
| TC-SW-09 | Exporter: metric set, label escaping, integer precision, rate limiting, error survival, HTTP endpoints | `tests/unit/test_telemetry_cli.py` | FR-06, NFR-08 | P |
| TC-SW-10 | CLI contract and exit codes (plan fits/doesn't fit/usage error; validate; env-file) | `test_telemetry_cli.py::test_cli_*` | FR-03, FR-04, FR-11 | P |
| TC-SW-11 | Config loading, defaults, search order, rejection of unknown/invalid keys | `tests/unit/test_units_and_config.py` | NFR-07 | P |
| TC-SW-12 | Real driver: probe, link checks don't crash, exporter, planner | `tests/integration/test_real_hardware.py` | FR-01, FR-06 | P (RTX 3060 Ti, driver 610.88, Windows 11) |
| TC-SW-13 | Container image builds, `--version`, simulated plan; `probe` with `--gpus all` | CI `deploy-config` job | NFR-06 | P (local Docker, WSL2 GPU) |
| TC-SW-14 | Deployment artefacts: compose config (all profiles), promtool rules, shellcheck | CI `deploy-config` job | FR-07, FR-08 | P (compose, promtool, shellcheck, Ubuntu 24.04 dry run — local Docker) |
| TC-SW-15 | Ray typed resources (architecture, cc, VRAM, NVENC) | `test_telemetry_cli.py::test_ray_resources`, `test_compat.py::test_ray_resources_typed_by_capability` | FR-09 | P |
| TC-SW-16 | GPU catalog lookup: longest-name match, VRAM variants, unknown cards, architecture facts | `tests/unit/test_compat.py` (spec table) | FR-13 | P |
| TC-SW-17 | Compatibility: driver window (Pascal on R610 fails, R580 pins), Kepler, Blackwell minimum, CUDA archs, NVENC, slots; L11 | `test_compat.py` (compatibility, link gate) | FR-13 | P |
| TC-SW-18 | Power budget: small cards, 2 × RTX 3090 needs 1200 W, unknown power, chassis membership (topology / display); L12; exporter power series | `test_compat.py` (power, exporter) | FR-15, NFR-03 | P |
| TC-SW-19 | Speed-aware placement and auto-selection: slow card left out or given only the spill; pooling when a single card is too tight; vLLM skips cc < 7; exclude list; nothing-fits; speed estimate; CLI compat/simulate/emit-config | `test_compat.py` (planning, CLI) | FR-11, FR-14 | P |
| TC-SW-20 | Fabric protocol, agent HTTP API, UDP discovery (loopback), peer parsing, rpc-server specs and supervisor restart | `tests/unit/test_fabric.py` (protocol, agent) | FR-16 | P |
| TC-SW-21 | Fabric planning: remote budgets, network-hop cost, remote only when needed, vLLM local-only, llama.cpp `--rpc`/`--device` launch, runtime gate on local subset | `test_fabric.py` (cluster, planning) | FR-17 | P |
| TC-SW-22 | CLI: simulated `@node` GPUs, `astra cluster` and `plan --peers` against a live agent, GPU-less head, missing rpc-server binary | `test_fabric.py` (CLI) | FR-16, FR-17, FR-18 | P |
| TC-SW-23 | Console API: self-contained page, topology/state, plan becomes active, input validation, models, chat streamed through a proxy to a fake engine, engine offline → 503 | `tests/unit/test_ui.py` | FR-19, NFR-13 | P |
| TC-SW-24 | Console visual check: light and dark themes, 1440 px and narrow widths, single-node and 3-node fabric with a 3-stage plan | Headless browser screenshots (`docs/images/`) | FR-19 | P (manual review) |
| TC-SW-25 | `astra auto`: recommendation ranking, CUDA build choice, llama.cpp release selection and extraction, config create/match/change, nvidia-smi topo parsing and L04 fallback, dry run | `tests/unit/test_auto.py` | FR-20 | P |

## B. Hardware acceptance (reference node)

| ID | Title | Procedure | Pass criteria | Req. | Result |
|---|---|---|---|---|---|
| TC-HW-01 | Mechanical inspection | Check GPU seating, brackets, slot gap ≥ 1, strain relief, cable bend radius | All items OK | NFR-02, NFR-04 | |
| TC-HW-02 | PSU cold check | Chassis PSU on the tester, before any GPU power-up | 12 V / 5 V / 3.3 V within ±5 %; PWR_OK asserted | NFR-04 | |
| TC-HW-03 | Power sync | Host on/off × 5; film or observe the fans | Chassis on and off with the host, skew ≤ 1 s | NFR-03 | |
| TC-HW-04 | Enumeration | `lspci -tvv`; `lspci -d 10b5:` | PLX switch plus both GPUs present behind it | FR-01 | |
| TC-HW-05 | Link gate | `astra validate --format junit --output hw05.xml` with the as-built config (`probe --emit-config`) | L01–L12 PASS (L05 WARN acceptable only for idle downshift; re-run under load → PASS) | FR-02, NFR-01, NFR-05 | |
| TC-HW-06 | Uplink bandwidth | `nvbandwidth` (or `bandwidthTest --device=N`) per GPU, then both at once | H2D and D2H ≥ 3.0 GB/s per GPU alone; aggregate ≤ uplink (report the value) | NFR-05 | |
| TC-HW-07 | Thermal soak | `gpu-burn 1800` on both GPUs at the measured ambient; `astra validate` during the run | < 83 °C sustained; no `hw_*`/`*_thermal_*` events; L10 PASS | NFR-02 | |
| TC-HW-08 | Power draw | Sum `astra_gpu_power_watts` during TC-HW-07; optional wall meter | ≤ 455 W sustained | NFR-03 | |
| TC-HW-09 | Sleep disabled | `systemctl status sleep.target suspend.target` | Both masked | NFR-03 (DR-05) | |
| TC-HW-10 | Cold-boot repeatability | 10 × full power-off (30 s) → boot → `systemctl status astra-selftest` | 10/10 PASS | NFR-10 | |
| TC-HW-12 | GPU change procedure (CR-001) | Power off → swap/add a card → `astra compat` → `astra probe --emit-config` → review/commit the config → `astra validate` | L11 and L12 PASS; L02 matches the new as-built; re-plan and re-run the runtime gate | FR-13, FR-15 | |
| TC-HW-11 | Earthing and single source | Multimeter: frame ↔ PSU earth pin; check each GPU's 8-pin and backplane feed trace to the chassis PSU; both PSUs on one strip | < 0.1 Ω; no host-PSU lead into the chassis except relay sense | NFR-04 | |

## C. Runtime acceptance (reference node)

| ID | Title | Procedure | Pass criteria | Req. | Result |
|---|---|---|---|---|---|
| TC-RT-01 | Pooled llama.cpp, 14B | `astra plan --gguf Qwen2.5-14B-Instruct-Q4_K_M.gguf --output plan.json --env-file deploy/compose/.env`; `docker compose --profile pooled up -d`; send load; `astra validate --phase runtime --plan plan.json` | Plan fits; R01–R05 PASS; completions correct | FR-03, FR-04, FR-05, FR-08 | |
| TC-RT-02 | Pooled vLLM, 8B AWQ | `astra plan --engine vllm --model llama-3.1-8b --output plan.json --env-file …`; `--profile vllm`; runtime gate | R03 shows ~5.4 GB / ~7.2 GB (±15 %); R04/R05 PASS. Record the vLLM version as the pinned version | FR-04, FR-05 (DR-01/02) | |
| TC-RT-03 | Partitioned isolation | `--profile partitioned` with ASTRA_GPU_INFER/TRANSCODE; run `nvidia-smi -L` inside each container | Each container sees exactly its UUID; NVENC job runs on the transcode GPU only | FR-08 | |
| TC-RT-04 | 24 h soak | TC-RT-01 under a sustained request load for 24 h; Prometheus retained | No Xid/AER in `astra validate`; `increase(astra_gpu_pcie_replay_total[24h]) == 0`; no slowdown alerts | NFR-01, NFR-02, NFR-10 | |
| TC-RT-05 | Engine recovery | `kill -9` the engine (systemd variant) or `docker kill` (compose) | Serving again within 60 s with no manual action | NFR-10 | |
| TC-RT-06 | Self-test gate | Set a wrong `expected_gpu` in `/etc/astra/astra.toml`; reboot | `astra-selftest` fails and `astra-inference` does not start; restore the config → starts | FR-10 | |
| TC-RT-00 | Stage 0 on a single PC (field-test-plan) | Real llama.cpp + GGUF; planned vs measured speed and memory; runtime gate; console chat; fabric over RPC loopback | Speed within ±30 %, R01–R05 PASS, chat works | FR-03, FR-04, FR-05, FR-17, FR-19 | **P** (2026-09-24, [report](reports/stage0-field-test.md)) |
| TC-RT-10 | `astra auto` on a real machine | Run `astra auto` with no arguments | Detects all GPUs, writes config, recommends, downloads if needed, launches, speed within ±30 %, gate PASS | FR-20 | **P** (2026-09-24, RTX 3060 Ti: Qwen2.5-7B Q4_K_M, 74.4 vs 73 tok/s, 5/5) |
| TC-RT-08 | Fabric: two machines | Machine B: `astra agent --rpc`; head: `astra cluster` then `astra plan --cluster --gguf <14B> --output plan.json`; launch; chat completions; `astra validate --phase runtime` on each node | Both nodes listed; plan uses GPUs on both; completions correct; per-token latency within 20 % of the estimate | FR-16, FR-17 | |
| TC-RT-09 | Fabric resilience | Kill a node's rpc-server | Agent restarts it within 10 s; llama-server reports an error rather than wrong output; service recovers after restart | NFR-10 | |
| TC-RT-07 | Ray elastic pool | `start_pool()`; submit 20 tasks with `resources={"gpu_ampere":1}` and 20 with `num_gpus=1` | Ampere tasks run only on the 3050; others spread across both | FR-09 | |

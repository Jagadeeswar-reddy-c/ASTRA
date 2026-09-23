# ADR-0013: Zero-input automation (`astra auto`)

* Status: Accepted (CR-005)
* Deciders: Solution Architect, Enterprise Architect

## Context
Every previous workflow asked the operator to say what hardware exists: `--simulate`
lists, `[[node.expected_gpu]]` entries, `--model`/`--quant`, engine downloads, launch
commands. The sponsor asked for ASTRA to work out itself how many GPUs there are and
what they are, using NVIDIA's tools, and act on it.

## Decision
`astra auto` runs six automatic steps with no required input:

| Step | Source of truth | Result |
|---|---|---|
| 1 Detect | `nvidia-smi -q -x` (GPUs, VRAM, link), `nvidia-smi topo -m` (GPU↔GPU paths: PIX/PXB = shared switch, PHB/NODE/SYS = via CPU), sysfs on Linux, UDP discovery of ASTRA agents | Inventory of every local and network GPU |
| 2 Configure | as-built renderer (`astra.asbuilt`) | Per-user `astra.toml` created on first run (`%APPDATA%\astra` / `~/.config/astra`). On later hardware changes it writes `astra.toml.new` and reports the difference; it never silently overwrites |
| 3 Check | `compat.analyse` | Stops on FAIL findings (driver window, PSU) |
| 4 Recommend | planner over the catalog × {Q8_0, Q6_K, Q5_K_M, Q4_K_M} | Largest model that fits at the target context with ≥ 256 MiB headroom per GPU and ≥ 8 tok/s; then the higher-precision quant; then speed |
| 5 Prepare | GitHub releases (llama.cpp), Hugging Face (GGUF, verified URLs) | The CUDA build chosen from the detected GPUs: CUDA 12 for pre-Turing, CUDA 13 for Blackwell, and an error with build instructions when both are present |
| 6 Launch and verify | planner on the downloaded GGUF (exact sizes), llama-server, runtime gate | Benchmark vs estimate, R01–R05, then the web console until Ctrl+C |

`nvidia-smi topo -m` also lets check L04 and `probe --emit-config` recognise a shared
PCIe switch on Windows, where sysfs is unavailable.

## Consequences
* \+ One command from a fresh machine to a verified, serving pool; the same command
  after adding or removing GPUs.
* \+ Verified on the field-test PC: detected 1 × RTX 3060 Ti, chose Qwen2.5-7B Q4_K_M,
  measured 74.4 tok/s vs 73 estimated, gate 5/5.
* − Downloads are large (a 14B Q4 model is ~9 GB). `--no-launch` shows the choice
  first; `--model/--quant` override it.
* − Automatic llama.cpp download is Windows-only (upstream publishes CUDA builds only
  for Windows). On Linux, `auto` uses an installed `llama-server` or `--binary`.

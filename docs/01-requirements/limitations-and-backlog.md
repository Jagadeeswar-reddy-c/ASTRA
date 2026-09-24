# ASTRA — Known Limitations and Product Backlog

| | |
|---|---|
| Document ID | ASTRA-REQ-003 |
| Version | 1.0 (2026-09-24), state after v0.5.1 |
| Owner | Product Manager · Reviewers: Enterprise Architect, Solution Architect, QA Lead |
| Purpose | One honest list of what ASTRA cannot do yet, why, and the backlog item that fixes it |

Priority: **P1** = blocks the product goal or a phase gate · **P2** = limits real use ·
**P3** = polish. "Hard limit" means physics or vendor policy: no software fixes it, the
design works around it.

## 1. Hard limits (design around them)

| ID | Limitation | Why | How ASTRA lives with it |
|---|---|---|---|
| LIM-01 | The pooled GPUs are **not one CUDA device**. Apps that expect one big GPU (PyTorch scripts, ComfyUI, games, Resolve) see separate cards | GeForce has P2P disabled; a kernel cannot be split across GPUs; remote memory is 35–100× slower ([vMerge study](../02-architecture/studies/vmerge-feasibility.md)) | Split the **model**, not the kernel: one endpoint, one console, one pool. vMerge layer 1 (BL-01) brings this to PyTorch apps |
| LIM-02 | **Adding GPUs adds memory, not single-user speed.** Pipeline decode speed ≈ the bandwidth-weighted average of the cards used, minus hop costs | Pipeline parallelism runs the stages one after another for each token. Tensor parallelism would add speed, but it needs fast GPU-to-GPU links that GeForce does not have | The planner keeps slow cards out unless their memory is needed. Concurrent users do scale (BL-10) |
| LIM-03 | Mixing generations is limited by **one driver per host**: Pascal/Maxwell cap it at R580, Kepler is out, Blackwell needs ≥ R570 | NVIDIA driver branch support | `astra compat` / L11 before purchase; RB-07 |
| LIM-04 | **NVIDIA only** (no AMD, Intel or Apple GPUs) | Detection, validation and launch are built on nvidia-smi and CUDA | Out of scope (PRD). llama.cpp RPC could carry a Vulkan/ROCm node later (BL-19) |
| LIM-05 | **No PCIe hot-plug**: GPUs and chassis are added with the power off | Consumer boards and BIOSes do not support PCIe hot-plug | Power off, add, boot, `astra auto` re-plans |

## 2. Product limitations (software can fix them)

| ID | Limitation | Impact | Backlog | Prio |
|---|---|---|---|---|
| LIM-10 | vMerge layer 1 (PyTorch/ComfyUI interposer) not built | Only LLM serving uses the pool, not image, video or audio models | BL-01 | **P1** |
| LIM-11 | The model catalog tops out at **14B** | Cannot plan or auto-download 32B/70B models, the ones a pool (or a DGX Spark) exists for; other models need `--gguf <file>` | BL-02 | **P1** |
| LIM-12 | The chassis (OCuLink + PEX8747 + second PSU) was **never built or tested**. TC-HW-01…12 are open | G5 cannot pass; the power sync, earthing and link-stability design is unproven | BL-03 (hardware) | **P1** |
| LIM-13 | Fabric was tested only over **loopback** on one PC; TC-RT-08/09 (two machines) are open | Real LAN latency and a node dropping out are unmeasured | BL-04 | **P1** |
| LIM-14 | Fabric has **no fault tolerance**: if one node or rpc-server dies, the request fails and the whole model must be reloaded | One flaky machine takes the pool down | BL-05 re-plan on node loss | P2 |
| LIM-15 | Fabric traffic (llama.cpp RPC) is **unauthenticated and unencrypted**, TCP only | Trusted LAN only (NFR-12); never on Wi-Fi guest nets or the internet | BL-06 mutual-TLS tunnel or WireGuard recipe | P2 |
| LIM-16 | Remote→remote stage boundaries **relay through the head** (2 hops) | Each extra machine costs ~2–4 ms per token | Upstream llama.cpp; planner already minimises it | P3 |
| LIM-17 | Remote model loading sends the weights over the network unless `--cache` is warm | 14B Q4 (8.5 GiB) over 1 GbE ≈ 75 s first load | BL-07 pre-seed the rpc cache from the GGUF | P3 |
| LIM-18 | The speed model covers **decode only**; prompt processing (prefill, compute-bound) is not estimated | Long prompts / RAG feel slower than the tok/s number suggests | BL-08 prefill model from FP16 TFLOPS | P2 |
| LIM-19 | **vLLM path not field-tested** (TC-RT-02); only llama.cpp is proven on hardware | vLLM pinned version unknown; AWQ splits unverified | BL-09 | P2 |
| LIM-20 | **Linux production path not run on real hardware**: Compose stack, systemd units, sysfs/AER checks, `setup-host.sh` (only a Docker dry run) | The target OS (Ubuntu 24.04, ADR-0006) is the least field-tested | BL-09 | **P1** |
| LIM-21 | On Linux, `astra auto` cannot download llama.cpp (no official CUDA Linux build); users compile it | Extra 10–20 minutes and a CUDA toolkit for new Linux users | BL-11 ship a CUDA container / build script | P2 |
| LIM-22 | **FT-05**: on Windows (WDDM), nvidia-smi overstates used memory by ~1.3 GB | Runtime check R03 and the free-memory planning are pessimistic on Windows | BL-12 NVML per-process memory | P3 |
| LIM-23 | **Single head node**: the endpoint, console and plan live on one machine | Head down = service down | BL-13 standby head (later) | P3 |
| LIM-24 | **No request scheduling or auth** on the OpenAI endpoint; binds to 127.0.0.1 | Sharing with a team needs a reverse proxy with auth | BL-14 API key + LAN bind option | P2 |
| LIM-25 | Power figures are **rated board power**, not measured at the wall; per-chassis only (one PSU) | Cannot budget several PSUs (modules) | BL-15 per-module PSU budget (CR-006) | P2 |
| LIM-26 | Concurrency not planned: KV cache is sized for one context; no `--parallel` slots in the plan | Several users share one context window | BL-10 plan N slots | P2 |
| LIM-27 | Recommendation uses **free memory right now** on a display GPU (NU-03) | The model choice changes with what else is open | By design; `--model/--quant` pins it | — |
| LIM-28 | Governance: G0/G1 sign-offs, CR-002 (bifurcation path) and the mini-PC specs are pending | Hardware BOM cannot freeze | Sponsor decision | **P1** |

## 3. Backlog (ordered)

| ID | Item | Closes | Size | Depends on |
|---|---|---|---|---|
| BL-02 | Add 32B / 70B models (Qwen2.5-32B, Llama-3.3-70B, …) with verified GGUF sources, incl. multi-part GGUF downloads | LIM-11 | S | — |
| BL-04 | Stage-1 field test: two PCs over the LAN (TC-RT-08/09) | LIM-13 | S | a second PC with a GPU |
| BL-09 | Linux field test: Ubuntu 24.04 on real hardware; Compose + systemd + vLLM (TC-RT-01/02/05/06) | LIM-19, LIM-20 | M | a Linux box or dual boot |
| BL-03 | Build and qualify the chassis (TC-HW-01…12) | LIM-12 | L | mini-PC specs, CR-002, BOM freeze |
| BL-01 | vMerge layer 1: PyTorch/diffusers interposer (`astra run -- python app.py`, ComfyUI launcher) that splits models across the pool | LIM-10 | L | — |
| BL-16 | **ASTRA Stack modules** (CR-006, [study](../02-architecture/studies/astra-stack-modules.md)): module identity, per-module power, cascade link checks, console grouping | LIM-25 | M (software) + L (hardware) | CR-006 decision |
| BL-05 | Fabric self-healing: detect a lost node, re-plan without it, reload | LIM-14 | M | BL-04 |
| BL-10 | Concurrency: plan KV for N parallel slots; report aggregate throughput | LIM-26 | S | — |
| BL-08 | Prefill (time-to-first-token) estimate | LIM-18 | S | — |
| BL-06 | Encrypted fabric (WireGuard recipe first, built-in TLS later) | LIM-15 | S / M | — |
| BL-14 | API key and LAN bind for the endpoint and console | LIM-24 | S | — |
| BL-11 | Linux llama.cpp: container image or `scripts/build-llama.sh` | LIM-21 | S | — |
| BL-07 | Pre-seed the rpc-server cache | LIM-17 | S | — |
| BL-12 | NVML-based memory accounting on Windows | LIM-22 | S | — |
| BL-13 | Standby head node | LIM-23 | L | BL-05 |
| BL-19 | Non-NVIDIA nodes via llama.cpp RPC (Vulkan/ROCm/Metal) | LIM-04 | M | PRD change |

Sizes: S ≤ 2 days, M ≤ 2 weeks, L > 2 weeks (including tests and docs).

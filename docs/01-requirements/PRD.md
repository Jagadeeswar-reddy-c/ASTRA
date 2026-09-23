# ASTRA — Product Requirements Document

| | |
|---|---|
| Document ID | ASTRA-REQ-001 |
| Version | 1.3 (2026-09-23): CR-001 any GPU mix; CR-003 fabric; CR-004 console |
| Status | Draft — baselined at Gate G1 |
| Owner | Product / PM · Reviewers: Enterprise Architect, QA |

## 1. Problem statement

A single desktop cannot hold or power several discrete GPUs. Cards of different
models also cannot be combined with SLI/NVLink. The goal is to take **whatever NVIDIA
GPUs are to hand** (any mix of GT, GTX and RTX cards; the concept's RTX 2060 6 GB +
RTX 3050 8 GB is the reference example) and use them as one pooled target for LLM,
vision-language and multimodal inference, or as separately assignable devices for
other workloads such as transcoding. It must work without replacing the host
motherboard and without relying on BIOS bifurcation. The set of GPUs is an **input
that changes over time** (CR-001).

## 2. Users and use cases

| Persona | Goal |
|---|---|
| ML practitioner | Serve a model that does not fit any single GPU (e.g. 14B Q4) through an OpenAI-compatible endpoint |
| Media engineer | Run NVENC transcoding on one GPU while inference runs on another, isolated |
| Operator (SRE) | Know within minutes if a GPU, the link or the thermals degrade; follow a runbook |
| Builder (HW eng.) | Assemble the chassis safely and prove it works before it goes into service |

| UC | Use case |
|---|---|
| UC-1 | Assemble chassis → verify link → put in service |
| UC-2 | Plan a model split before buying or loading anything ("will it fit?") |
| UC-3 | Serve a pooled model across all GPUs |
| UC-4 | Partition GPUs between isolated services |
| UC-5 | Dispatch parallel jobs to whichever GPU has capacity (Ray) |
| UC-6 | Monitor, alert, diagnose and recover |

## 3. Functional requirements

Priority: **M** = must, **S** = should, **C** = could. Verification: T = automated
test, I = inspection, D = demonstration, A = analysis.

| ID | Requirement | Pri | Verify |
|---|---|---|---|
| FR-01 | The system shall discover every NVIDIA GPU (UUID, model, architecture, VRAM, PCI address) and, on Linux, the full PCIe path from root port to GPU including any packet switch | M | T |
| FR-02 | The system shall run an automated **link gate** that checks the GPU inventory against the as-built configuration, switch presence, uplink generation/width, slot link width, replay and AER counters, kernel-log fault signatures and thermals, and exit non-zero on failure | M | T, D |
| FR-03 | The system shall compute a pipeline-parallel placement of a model's transformer blocks across GPUs of different sizes, accounting for weights, KV cache, fixed tensors and a per-GPU reserve, and report whether it fits and the maximum context that fits | M | T |
| FR-04 | The system shall generate launch commands for llama.cpp and vLLM that implement the placement, addressing GPUs by UUID in PCI-bus order | M | T |
| FR-05 | The system shall run an automated **runtime gate** that samples a serving node and verifies every planned GPU holds its planned share of memory (±15 %), with no throttling and no PCIe replay growth | M | T, D |
| FR-06 | The system shall expose GPU, link and error metrics in Prometheus format | M | T |
| FR-07 | The deployment shall include alert rules and a dashboard for GPU loss, link degradation, PCIe errors, thermals and chassis power | M | I, D |
| FR-08 | The deployment shall support *pooled* (one model across all GPUs) and *partitioned* (GPU-per-service) profiles with UUID-pinned device isolation | M | D |
| FR-09 | The system should expose the GPUs to Ray with typed resources (architecture, VRAM class) for elastic task scheduling | S | T |
| FR-10 | A boot-time self-test shall run the link gate and prevent the inference service from starting when it fails | M | D |
| FR-11 | Planning and compatibility checks shall work for hypothetical hardware given only model names (e.g. "RTX 3060, GT 1030"), so capacity questions can be answered before purchase | S | T |
| FR-12 | Planning should read GGUF model files directly to size each block exactly | S | T |
| FR-13 | The system shall determine, for any mix of NVIDIA GPUs (installed or hypothetical), the driver branch window that supports all of them, the engines each GPU can run, the CUDA architectures engine builds must include, and NVENC availability, and fail the link gate when the installed driver cannot support every GPU | M | T |
| FR-14 | The planner shall choose automatically which GPUs to use for a model, favouring estimated decode speed with a memory margin, and shall place fewer layers on slower GPUs | M | T |
| FR-16 | GPUs in other machines shall join the pool: each machine runs an agent that advertises its GPUs (UDP discovery or static peers) and serves them to the cluster | M | T |
| FR-17 | The planner shall place a model across local and remote GPUs, costing each machine boundary, and generate one llama.cpp launch that drives all of them behind a single OpenAI-compatible endpoint | M | T |
| FR-18 | The head node shall work without a local GPU (orchestration/API only) | S | T |
| FR-19 | A web console shall show the pool topology (machines, GPUs, links), live GPU stats, the active plan's layer placement and pipeline order, health findings, and let the user plan a model and chat with the running engine | S | T, D |
| FR-20 | One command shall detect every NVIDIA GPU (count, model, memory, links, GPU-to-GPU topology via NVIDIA tools) locally and on the network, write the as-built configuration, choose the best model that fits, download what is missing, launch, benchmark and validate, with no operator input | M | T, D |
| FR-15 | The system shall compute the chassis power budget from the installed GPUs, recommend a PSU size, and fail the link gate when the configured PSU is too small | M | T |

## 4. Non-functional requirements

| ID | Category | Requirement | Pri | Verify |
|---|---|---|---|---|
| NFR-01 | Signal integrity | 0 uncorrectable AER errors and 0 PCIe replays at acceptance, and none during a 24 h load soak | M | T (L07, L08, L09, R05) |
| NFR-02 | Thermal | GPU core < 83 °C sustained at full load at ≤ 30 °C ambient; no hardware/thermal slowdown events | M | T (L10, R04), D |
| NFR-03 | Power | Chassis PSU powers on and off with the host (skew ≤ 1 s); for the installed GPUs, sustained draw ≤ 70 % of the PSU rating and estimated peak ≤ the rating | M | T (L12), D (TC-HW-03, 08) |
| NFR-04 | Electrical safety | Each GPU powered only from the chassis PSU (slot and aux); common protective earth; link never connected while powered | M | I (TC-HW-11) |
| NFR-05 | Interconnect | Uplink trains Gen3 x4; measured host→device bandwidth ≥ 3.0 GB/s per GPU | M | T (L05), D (TC-HW-06) |
| NFR-06 | Portability | Control plane runs on Ubuntu 22.04/24.04 and Windows 11 with Python ≥ 3.11 | M | T (CI matrix) |
| NFR-07 | Maintainability | No third-party runtime dependencies in the core; ≥ 85 % line+branch coverage; lint and strict typing gate CI | M | T (CI) |
| NFR-08 | Observability | `/metrics` responds < 1 s; nvidia-smi sampled at most every 2 s regardless of scrape rate; 30 d retention | S | T, I |
| NFR-09 | Security | Services run as non-root with systemd/container hardening; endpoints bind to localhost by default; no secrets in the repository | M | I |
| NFR-10 | Reliability | Inference service restarts automatically on failure; cold boot enumerates all GPUs 10/10 times | M | D (TC-HW-10, TC-RT-05) |
| NFR-12 | Fabric security | Fabric traffic (agent, discovery, rpc-server) is confined to a trusted LAN/VLAN by firewall; the agent API is read-only; no fabric port is exposed to the internet | M | I |
| NFR-13 | Console security | The console binds to localhost by default, proxies chat only to the configured engine, caps request bodies, and loads no external assets | M | T, I |
| NFR-11 | Cost | Chassis hardware BOM ≤ €420 excluding GPUs, one-off tooling (BOM-11) and PSU upsizing beyond 650 W required by the chosen GPUs | S | A (bom.csv) |

## 5. Constraints

* The host exposes a single M.2 Key-M slot with PCIe x4 (no bifurcation needed).
* The GPUs are GeForce: no NVLink and no peer-to-peer over PCIe (driver-disabled).
  Pre-Ampere cards have no bfloat16.
* All GPUs share one driver. Maxwell/Pascal/Volta cap it at the R580 branch; Kepler
  is unsupported.
* The external link is not hot-pluggable.

## 6. Acceptance

The product is accepted at Gate G5 when all M-priority requirements are verified by
the methods above. Evidence is recorded in the test reports referenced from
`docs/05-testing/traceability-matrix.md`.

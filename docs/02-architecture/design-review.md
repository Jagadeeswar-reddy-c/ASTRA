# ASTRA — Architecture Design Review of the Concept Document

| | |
|---|---|
| Document ID | ASTRA-ARC-002 |
| Version | 1.0 (2026-09-23) |
| Reviewed artefact | "Project ASTRA" concept document (hardware diagram, pooling logic, BOM, roadmap) |
| Reviewers | Enterprise Architect, Solution Architect, Hardware Engineer, QA |

The concept is sound: native PCIe over OCuLink, a packet switch instead of
bifurcation, a dedicated chassis PSU with relay sync, and runtime pooling instead of
SLI. The review found **15 issues** that would have caused failed acceptance or
unsafe operation. Each is dispositioned below and traced to a fix.

Severity: **High** blocks acceptance or risks hardware; **Medium** degrades
function or operability; **Low** is an improvement.

| ID | Sev | Finding | Evidence / reasoning | Disposition |
|---|---|---|---|---|
| DR-01 | High | **Phase 3 uses `--tensor-parallel-size 2`, which does not produce a 43 %:57 % split.** Tensor parallelism shards every layer *equally*, so each GPU gets half. The 6 GB card caps the pool at 2 × ~5.4 GB, and each layer needs an all-reduce over the x4 uplink with no P2P. | Tensor-parallel partitioning is symmetric by construction | **Changed.** Pipeline parallelism with an explicit uneven layer partition: llama.cpp `--split-mode layer --tensor-split a,b`; vLLM `--pipeline-parallel-size 2` + `VLLM_PP_LAYER_PARTITION`. ADR-0003. Tested in `test_vllm_launch_uses_pipeline_not_tensor_parallel`. |
| DR-02 | High | **Llama-3.1-8B-Instruct does not fit in 14 GB unquantized.** The BF16 weights are ≈ 16 GB before KV cache, and the RTX 2060 (Turing) cannot run bfloat16. | 8.03 B params × 2 bytes; Turing has no BF16 tensor cores | **Changed.** Use AWQ-INT4 (vLLM) or GGUF Q4_K_M/Q8_0 (llama.cpp), `--dtype float16`. The planner refuses plans that don't fit (`test_llama8b_fp16_does_not_fit_14gb_pool`). |
| DR-03 | Medium | **The PEX8747 is a PCIe Gen3 switch.** The "PCIe 4.0 x4" cable will train Gen3 x4: ≈ 3.9 GB/s theoretical, ~3.2–3.5 GB/s measured. | PEX8747 is a 48-lane Gen3 part | **Accepted** for v1 (enough for pipeline traffic and model loading). The link gate expects Gen3 x4. Upgrade path: a Gen4 switch (PEX88xxx / Switchtec) — future ADR. |
| DR-04 | High | **BOM gap: no chassis-side OCuLink receiver** (SFF-8611 → x16 slot/upstream connector) and no power feed for the backplane or receiver. | The diagram shows a cable ending at the switch, but most PEX8747 boards take their upstream through an x16 edge or slot | **Changed.** Added to `hardware/bom.csv`, with backplane aux power from the chassis PSU. |
| DR-05 | High | **Host sleep/hibernate powers off the chassis.** The relay is driven by a host power rail, so S3/S4 drops the GPUs from the bus (Xid 79) and they don't return on resume. | Relay sense follows host standby state | **Changed.** `setup-host.sh` masks sleep targets; TC-HW-09; runbook RB-01. |
| DR-06 | Medium | **OCuLink is not hot-plug** in this topology; connecting while powered can latch up or damage the ReDriver. | No hot-plug controller or surprise-removal support | **Changed.** Assembly SOP and ops rule: connect or disconnect only with both systems off. |
| DR-07 | Medium | **`--gpus device=1` / `device=2` is fragile.** Indices are 0-based, include any host GPU, and follow CUDA's "fastest first" order unless `CUDA_DEVICE_ORDER=PCI_BUS_ID` is set. | CUDA and NVML index semantics | **Changed.** Address GPUs by UUID with PCI-bus order everywhere (ADR-0004); Compose uses `NVIDIA_VISIBLE_DEVICES=<uuid>`. |
| DR-08 | Medium | **GeForce peer-to-peer is disabled.** Inter-GPU tensors bounce through host RAM and cross the x4 uplink twice. | NVIDIA disables P2P on consumer GPUs | **Accepted.** Pipeline parallelism sends only one hidden state per token per boundary (≈ 10 KB for 14B at FP16). This informs ADR-0003. |
| DR-09 | Medium | **Link generation reads Gen1 at idle.** ASPM/power management downshifts idle links, so a naive check on `pcie.link.gen.current` fails randomly. | Observed on the dev machine: RTX 3060 Ti reports Gen1/Gen4 at idle | **Changed.** Gen downshift with max-capable ≥ expected is a WARN; lane width is the hard check (L05, L06). |
| DR-10 | Low | **"Common ground via the cable shield"** — the shield is not a return path and should not be relied on for equipotential bonding. | Shield bonding is for EMI; signal ground uses the dedicated ground conductors | **Changed.** Common protective earth (same outlet strip) plus the OCuLink ground conductors; no ad-hoc DC ground straps. See hardware design §3. |
| DR-11 | Medium | **Host OS unspecified.** vLLM is Linux-only; Windows (WDDM) also hides sysfs topology and AER. | vLLM platform support; WDDM driver model | **Decided.** Ubuntu 24.04 LTS for production (ADR-0006). Tooling stays cross-platform. |
| DR-12 | Medium | **MMIO/BAR space.** A switch plus two GPUs behind an M.2 port can exhaust 32-bit MMIO ("BAR n: no space"). | Common failure with external switches | **Changed.** BIOS checklist: Above 4G Decoding on; `pci=realloc` option; L09 detects it. |
| DR-13 | Low | **Pooling is not always better.** Llama-3.1-8B Q4 (≈ 4.9 GB) fits on the RTX 3050 alone, and splitting only adds a pipeline hop. | Planner output | **Changed.** The planner warns when one GPU suffices. Guidance: pool only when needed (e.g. 14B Q4, 8B Q8). |
| DR-14 | Low | **1.0 m cable length.** Fine at Gen3; marginal for a future Gen4 switch. | Channel loss grows with length and rate | **Accepted** for Gen3. The Gen4 upgrade needs ≤ 0.5 m or ReDrivers on both ends. |
| DR-15 | Low | **M.2 slot selection.** A chipset-attached M.2 shares DMI bandwidth, and some boards disable SATA ports when the M.2 is populated. | Motherboard lane-sharing tables | **Changed.** Hardware design §2: prefer a CPU-attached M.2 and check the board manual's lane-sharing table. |

## Addendum — CR-001 (any NVIDIA GPU mix)

| ID | Sev | Finding | Disposition |
|---|---|---|---|
| DR-16 | High | **Driver lock-in by old cards.** Maxwell/Pascal/Volta are unsupported after R580 and Kepler after R470. On the R610 dev host a GT 1030 or GTX 10-series card would not enumerate. | ADR-0009; check L11 fails on a mismatch; `astra compat` prints the driver window; runbook RB-07 |
| DR-17 | High | **The PSU budget depends on the cards.** 650 W suits the reference pair, but two RTX 3090s need ≥ 1,200 W. | Check L12 and `astra compat` compute the budget and recommend a PSU; alert `AstraChassisUndersized` |
| DR-18 | Medium | **Slow cards slow the pipeline.** A GT 1030 (48 GB/s) holding a VRAM-proportional share makes the pool slower than the best card alone. | ADR-0010 speed objective and automatic set selection |
| DR-19 | Medium | **Engine builds must target every architecture.** CUDA 13 dropped sm < 75, and vLLM needs cc ≥ 7.0. | `astra compat` prints `CMAKE_CUDA_ARCHITECTURES`; the planner excludes GPUs an engine can't use |
| DR-20 | Low | **Not every card has NVENC** (GT 1030) and some cards are x4/x8 or slot-powered. | NVENC column and finding; L06 compares against each card's own width; slot-power total reported for the backplane feed |

## Items confirmed as correct

* Native PCIe (OCuLink) over USB4/Thunderbolt for latency and CPU overhead — ADR-0001.
* A packet switch instead of bifurcation (works on any M.2 x4 slot; one endpoint per
  GPU) — ADR-0002.
* A dedicated chassis PSU with relay sync, and the single-rail/single-source rule.
* The VRAM-proportional split as the first-order rule (6/14 : 8/14). The planner
  refines it with fixed costs and reproduces it when those are negligible
  (`test_split_tracks_vram_ratio_when_fixed_costs_are_small`).
* The Phase 3 success metric of ~5.4 GB / ~7.2 GB holds for vLLM at
  `--gpu-memory-utilization 0.90`, because vLLM pre-allocates per GPU. It is
  implemented as runtime check R03 (`test_runtime_vllm_expects_capacity_ratio`).

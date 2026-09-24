# ASTRA Stack — Feasibility Study: stackable GPU modules ("like DGX Spark")

| | |
|---|---|
| Document ID | ASTRA-ARC-004 |
| Version | 0.1 (2026-09-24), for decision (CR-006) |
| Owner | Enterprise Architect · Reviewers: Solution Architect, Hardware Eng., Software Dev |
| Question | Can ASTRA become a set of identical modules that you stack, like DGX Spark units, and that together behave like **one big GPU**? |

## 1. Summary

| Part of the idea | Verdict |
|---|---|
| Identical, self-powered **modules ("bricks") you add one by one** | **Yes.** Best done as 1-GPU bricks on OCuLink, wired as a **star** from a hub in the host (§4) |
| The stack behaves as **one pool**: one model, one endpoint, one console, capacity adds up | **Yes, already true in software** (planner + Fabric). Needs module awareness: per-brick power, grouping, checks (§6) |
| The stack is **one CUDA device** to any program | **No**, and DGX Spark does not do this either (§2). It is LIM-01; vMerge layer 1 is the route for PyTorch apps |
| A stack **beats a DGX Spark** | On **speed per token and cost**, yes for dense models, when built from GeForce cards with fast memory. On memory size per watt, size, noise and FP4, no (§3) |

**Recommendation:** approve CR-006 as **"ASTRA Stack": hub + 1-GPU bricks over PCIe (star)**
for one host, with ASTRA Fabric (network) to join several stacks. Start with the
software (module awareness, sizing tool); build hardware bricks after the chassis
qualification (BL-03), because a brick is the same electrical design with one GPU.

## 2. What DGX Spark actually does

Published specifications ([NVIDIA hardware overview](https://docs.nvidia.com/dgx/dgx-spark/hardware.html)):
one GB10 superchip with **128 GB unified LPDDR5X at 273 GB/s** and a ConnectX-7 NIC
(two QSFP ports, up to 200 Gb/s). Two units cabled directly give 256 GB for models up
to ~405B parameters at FP4; three can run in a ring, four need a switch.

The important point: **stacked Sparks are not one GPU either.** Each unit is its own
computer with its own GPU. The software (vLLM, TensorRT-LLM, NCCL) splits the model
across units with pipeline/tensor parallelism over the 200 Gb/s link. The memory is
unified only *inside* one unit. That is the same architecture as ASTRA Fabric, with a
faster network. So the question is not "how to fuse GPUs" (impossible, see the
[vMerge study](vmerge-feasibility.md)), but "how to make the modules and the pool as
seamless as a Spark stack".

## 3. The numbers: a stack of GeForce bricks vs DGX Spark

Decode is memory-bandwidth-bound. The planner's model (validated within 2–14 % on real
hardware, [Stage-0 report](../../05-testing/reports/stage0-field-test.md)):
`tok/s ≈ 0.75 × bandwidth / bytes of weights read per token`. With a pipeline, capacity
adds up while speed stays at the (weighted) speed of one card.

**Example A: 70B dense model, Q4_K_M (~42.8 GB weights, 8k context)**

| Build | Pooled memory | Bandwidth per stage | Est. decode | Rated power |
|---|---|---|---|---|
| DGX Spark (1 unit) | 128 GB unified | 273 GB/s | ~4.8 tok/s | ~240 W |
| 5 × RTX 3060 12 GB bricks | 60 GB | 360 GB/s | ~6.3 tok/s | 850 W |
| 4 × RTX 5060 Ti 16 GB bricks | 64 GB | 448 GB/s | ~7.8 tok/s | 720 W |
| 3 × RTX 3090 24 GB bricks | 72 GB | 936 GB/s | ~16 tok/s | 1050 W |

(4 × 12 GB and 2 × 24 GB are just too small once KV cache and the per-GPU CUDA reserve
are counted. The planner will confirm these numbers once the 70B models are added,
BL-02.)

**Example B: the same bricks over PCIe vs over the network** (planner output, today)

| Pool | PCIe stack (one host) | Network stack (one GPU per PC) | Loss |
|---|---|---|---|
| 2 × RTX 3060 Ti, Qwen2.5-14B Q4_K_M | ~35 tok/s | ~31 tok/s | −11 % |
| 3 × RTX 3060 Ti, Qwen2.5-14B Q8_0 | ~21 tok/s | ~18 tok/s | −14 % |

A PCIe hop costs ~0.15 ms per token boundary; a network hop costs ~2 ms (measured
0.9 ms on loopback, mostly software, so a faster NIC helps loading but hardly helps
per-token latency). **Inside one stack, PCIe wins; between stacks, the network is fine.**

**Where Spark wins:** 128 GB in one 240 W box (large MoE models such as 120B-class fit
in one unit), FP4 tensor cores, silent, no assembly, one supported software stack.
**Where the ASTRA Stack wins:** 1.6–3.4× the decode speed on dense models, reuse of
GPUs people already own, grow one brick at a time, replace a brick when a better card
comes out.

## 4. Hardware options for a module

| | M1-cascade: bricks daisy-chained | **M1-star: hub + bricks** (recommended) | M2: network bricks (Spark-like) |
|---|---|---|---|
| Brick contents | GPU(s) + PSU + PCIe switch with an "up" and a "down" port | 1 GPU + PSU + passive OCuLink→x16 receiver | Mini-ITX PC + GPU + PSU |
| Connection | Host → brick 1 → brick 2 → … | Host hub card → one cable per brick | Ethernet switch (or direct cables) |
| Host sees | All GPUs as local PCIe | All GPUs as local PCIe | Separate machines (ASTRA Fabric) |
| Per-token hop | ~0.15 ms per switch | ~0.15 ms | ~2 ms per machine |
| A brick fails | **Every brick behind it disappears** | Only that brick | Only that brick |
| Add a brick | Power off, reboot | Power off, reboot (LIM-05) | Live: `astra auto` re-plans |
| Cost per brick | High (a switch per brick) | Lowest (receiver board + PSU) | Highest (a whole PC) |
| Limit | BIOS bus numbers / MMIO | Hub ports: 4 bricks at x4 on an x16 slot | Network, head relaying |

**Why star:** each brick is simple and identical. A failure is isolated. There is one
switch level (lowest latency). The hub is one of:
* **Switch hub**: PCIe x16 card with a PLX/Broadcom switch and 4 OCuLink ports. It works
  on any board, including the planned mini PC, and does not need BIOS bifurcation. This
  is the current chassis design's PEX8747 moved into the host.
* **Bifurcation hub**: x16 → 4 × OCuLink x4, passive and cheaper, but needs x4x4x4x4
  bifurcation in the BIOS (CR-002, open).

x4 per brick is enough for inference: a token boundary moves a few KB of activations.
Only model loading slows down (40 GB over Gen3 x4 ≈ 12 s).

**Brick electrical rules** (inherited from hardware-design.md): every GPU is powered
only from its own brick's PSU (slot power included, through the receiver board); the
brick PSU starts from the hub's sync signal (Add2PSU-style relay); all bricks are bonded
to one earth and plugged into one power strip.

**Main hardware risk:** many GPUs on consumer boards exhaust 64-bit MMIO / BAR space
(Above-4G decoding and Resizable BAR must be enabled; some BIOSes cap at 2–4 GPUs).
New risk R-15, to be tested with 4 bricks before committing to a hub design.

## 5. Making the stack feel like one GPU

What "one GPU" can honestly mean, and where ASTRA stands:

| Level | Meaning | Status |
|---|---|---|
| One **model endpoint** | One OpenAI URL; the model spans all bricks | Done (v0.3–0.5) |
| One **appliance** | `astra auto` finds bricks, sizes, downloads, launches, verifies; the console shows "ASTRA Stack: 4 bricks, 64 GB" | Mostly done; module grouping missing (§6) |
| One **device for PyTorch apps** | ComfyUI / diffusers / transformers split across bricks without code changes | vMerge layer 1 (BL-01) |
| One **CUDA device** | Any binary sees one 64 GB GPU | Not possible on GeForce (LIM-01). Spark does not do it either |

## 6. Software work (BL-16)

| # | Change | Where |
|---|---|---|
| S1 | **Module identity:** group GPUs into bricks from the PCIe topology (sysfs chain / `nvidia-smi topo`), with an optional `[[module]]` entry (id, PSU watts) in the config; written by `probe --emit-config` | `asbuilt.py`, `config.py` |
| S2 | **Power per brick:** L12 and the exporter per module instead of one chassis PSU | `compat.py`, `checks.py`, `exporter.py` |
| S3 | **Stack link check (L13):** each brick at its expected Gen/width, one switch level (star), Above-4G / BAR1 size sufficient | `checks.py`, `nvsmi.py` |
| S4 | **Sizing tool:** `astra size --model llama-3.3-70b --min-tps 8` → which bricks to buy (reverse planner over the GPU catalog) | new `planner/sizing.py`, CLI |
| S5 | **Console:** GPUs drawn grouped by brick; brick power and health | `ui/` |
| S6 | 70B / 32B models in the catalog (BL-02), required for meaningful stack plans | `planner/models.py` |

S4 and S6 are useful before any hardware exists: they answer "which bricks do I buy to
run model X at speed Y", the same question a Spark buyer asks.

## 7. Decisions requested (CR-006)

| # | Decision | Recommendation |
|---|---|---|
| D1 | Topology inside one host | Star (hub + bricks), not cascade |
| D2 | GPUs per brick | 1 (a 2-GPU brick = the current chassis) |
| D3 | Hub type | Switch hub unless the mini PC's BIOS supports x4x4x4x4 (ties to CR-002) |
| D4 | Target | 70B Q4 at ≥ 8 tok/s, i.e. 4 × 16 GB bricks with ≥ 448 GB/s |
| D5 | Order | Software S6 → S4 → S1–S3, S5; hardware brick after BL-03 |

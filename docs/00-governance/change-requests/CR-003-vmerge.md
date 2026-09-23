# CR-003 — vMerge: unmodified applications use several GPUs as one

| | |
|---|---|
| Raised | 2026-09-23 by the Sponsor ("the main goal") |
| Status | **Approved (2026-09-23)**: the sponsor delegated the design ("do it like the Mac clusters, the optimum way"). Increment 1 = ASTRA Fabric (ADR-0011), shipped in v0.3.0; increment 2 = vMerge layer 1 (PyTorch interposer) |
| Affects | Vision (charter §1), PRD (new FR-16…), architecture (new ADR-0011), new components `vmerge` (Python) and `libvmerge` (C) |

## Request
Build a user-space CUDA proxy driver that reports one virtual GPU with the combined
VRAM (e.g. "14 GB"). It splits allocations across physical GPUs behind a unified
address table and splits kernel execution, so PyTorch, ComfyUI and DaVinci Resolve
use all GPUs without changes.

## Study outcome
* **Goal is feasible; mechanism is not.** Transparent address-space merging needs
  GPU peer-to-peer, which GeForce drivers disable over PCIe. Even with P2P, pooled
  bytes are 50–100× slower than local VRAM for bandwidth-bound workloads.
  Automatic kernel-grid splitting is not correct for arbitrary compiled kernels.
* **Recommended scope:** vMerge in three layers. Layer 1: a framework interposer
  (unmodified PyTorch apps get models split across GPUs by the ASTRA planner).
  Layer 2: a CUDA driver shim (virtual device, VRAM overflow to host RAM, policy).
  Layer 3: a time-boxed GEMM-splitting spike with a measured go/no-go.

## Options for the sponsor

| Option | Scope | Outcome |
|---|---|---|
| **A (recommended)** | Layers 1 + 2, then the layer-3 spike | Unmodified PyTorch apps (ComfyUI, Forge, transformers) use all GPUs at near-single-GPU speed; every CUDA app gets overflow instead of OOM |
| B | As proposed (driver-level address/kernel merge) | Research project with a high risk of no usable result on GeForce hardware; the study recommends against it |
| C | Layer 1 only | Fastest route to the main use case (AI apps) |

## Decision
The sponsor pointed at the Mac-cluster approach (exo / MLX distributed) and delegated
the design. That approach is the study's core rule applied across machines: partition
the model, move only activations. Decided:

1. **Increment 1: ASTRA Fabric** (ADR-0011): every machine's NVIDIA GPUs join one pool
   (agent + discovery + planner + llama.cpp RPC), with one OpenAI-compatible endpoint.
   Delivered in v0.3.0.
2. **Increment 2: vMerge layer 1**: unmodified PyTorch apps (ComfyUI, Forge,
   transformers) are split across the same pool by the same planner.
3. **Later:** layer 2 (CUDA shim: overflow and policy) and the layer-3 GEMM spike, each
   gated by the measurements in ASTRA-ARC-003 §7.
4. The driver-level "one fake GPU" (option B) is **rejected** for the reasons in
   ASTRA-ARC-003 §2–4.

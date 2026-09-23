# vMerge — Feasibility Study: presenting several GPUs as one to unmodified software

| | |
|---|---|
| Document ID | ASTRA-ARC-003 |
| Version | 0.1 (2026-09-23), for decision (CR-003) |
| Owner | Enterprise Architect · Reviewers: Solution Architect, Software Dev |
| Question | Can a user-space CUDA shim make e.g. an RTX 2060 (6 GB) + RTX 3050 (8 GB) appear as one "14 GB GPU" to arbitrary software (PyTorch, ComfyUI, DaVinci Resolve), splitting memory and kernels transparently? |

## 1. Summary

| Part of the proposal | Feasible? | Useful on ASTRA hardware? |
|---|---|---|
| Intercepting CUDA (LD_PRELOAD on `libcuda.so`, proxy `nvcuda.dll` on Windows) and reporting one virtual device | **Yes.** Proven by existing projects (§5) | Yes, as a control layer |
| One `cudaMalloc(12 GB)` backed half by GPU A and half by GPU B, behind one address range | **Only if the GPUs can do peer-to-peer (P2P).** GeForce cards have P2P disabled over PCIe | No on stock drivers. Even with P2P, the remote half runs **35–100× slower** than local VRAM (§3) |
| Automatically splitting an arbitrary kernel's grid across both GPUs | **No, not in general.** Kernels are compiled binaries with data-dependent memory access, atomics and inter-block communication (§4) | No |
| Goal: **unmodified apps use both GPUs together** | **Yes, at a higher layer.** Split the *model*, not the *kernel* (§6) | **Yes.** This is what ASTRA's planner already does for LLMs |

**Recommendation:** keep the goal and change the mechanism. Build vMerge as three layers:
1. **Framework interposer (first).** Unmodified PyTorch apps (ComfyUI, Forge/A1111,
   transformers scripts, Whisper, …) get their models split across GPUs
   automatically.
2. **CUDA driver shim (second).** One stable virtual device, VRAM overflow into
   system RAM instead of out-of-memory crashes, per-app GPU policy.
3. **A time-boxed research spike** on splitting *semantically known* library calls
   (cuBLAS GEMM) across GPUs, with a measured go/no-go.

## 2. How a kernel sees memory (why "one fake GPU" is hard)

A CUDA kernel receives raw device pointers and computes addresses at run time, e.g.
`out[idx[i]] = w[j*ld + k]`. Every load and store is issued by the GPU executing
the kernel, against memory that GPU can address. For one kernel on GPU A to read a
byte that lives on GPU B, one of these must be true:

1. **P2P mapping.** A's memory accesses travel over PCIe into B's VRAM. The CUDA
   virtual-memory API (`cuMemCreate` / `cuMemMap` / `cuMemSetAccess`) can already
   map physical memory from two GPUs into **one contiguous virtual range**. That is
   exactly the "UVA translation table" in the proposal, and the driver provides it
   when peer access is allowed. **GeForce drivers report `canAccessPeer = 0` over
   PCIe**, so `cuMemSetAccess` refuses. (A community patch to NVIDIA's open kernel
   modules enables P2P on some cards with large Resizable BAR. Turing cards such as
   the RTX 2060 have no Resizable BAR support, so it does not apply to them.)
2. **Unified Memory page migration.** With `cudaMallocManaged`, a page is moved on
   first touch. Without P2P it moves GPU B → host RAM → GPU A, crossing the
   uplink twice. That is correct but slow, and pages ping-pong when both GPUs touch
   them.

There is no third way. A shim cannot run code "on the memory". It can only choose
where data lives and which GPU runs a kernel.

## 3. The bandwidth arithmetic

| Path | Bandwidth | vs RTX 2060 VRAM (336 GB/s) |
|---|---|---|
| RTX 2060 local VRAM | 336 GB/s | 1× |
| RTX 3050 local VRAM | 224 GB/s | 1.5× slower |
| P2P between the two GPUs through the PEX8747 switch (if enabled; RTX 3050 is x8 Gen3) | ≈ 6–7 GB/s | **≈ 50× slower** |
| Via host RAM over the shared Gen3 x4 uplink (no P2P) | ≈ 3.2 GB/s each way, crossed twice | **≈ 100× slower** |

Inference workloads (LLM decode, diffusion U-Nets/DiTs) are **memory-bandwidth
bound**: every step reads essentially all weights. Worked example: an 8B model at
Q8 (≈ 8.5 GB) on a "virtual 14 GB GPU" with ≈ 4.5 GB physically on the far card.
Each token reads those 4.5 GB over the link: 4.5 / 3.2 ≈ **1.4 s per token**. Split
by layers instead (ASTRA's pipeline plan), each GPU reads its own weights locally
and only a ~10 KB activation crosses the link: ≈ **30–40 ms per token**. That is
about **40× faster** for the same two cards.

> **Rule:** move the computation to the data (split the model), never the data to
> the computation (split the address space). A transparent address-space merge
> makes every pooled byte pay the PCIe price on every step.

## 4. Why kernels cannot be split automatically

* **Unknown access patterns.** Which addresses a thread block touches depends on
  runtime values (indices, strides, pointers inside structs). Splitting blocks
  between GPUs needs every block's inputs present on its GPU, which cannot be known
  before the kernel runs.
* **Cross-block semantics.** Global atomics, reductions, grid-wide sync
  (cooperative groups), persistent kernels, and ordering between kernels and memcpys
  on streams all assume one coherent memory. Splitting breaks them silently: wrong
  results, not just slow ones.
* **Closed binaries.** cuBLAS, cuDNN, cuFFT, TensorRT and apps like Resolve ship
  pre-compiled kernels (SASS). Research compilers that partition kernels across GPUs
  need source or IR with affine index expressions. They cannot handle arbitrary
  binaries.
* **Mixed architectures.** One virtual device must report one compute capability.
  With a 2060 (7.5) and a 3050 (8.6) it must report 7.5. Libraries then choose
  Turing code paths, and the Ampere card loses its features.
* **Surface area.** The CUDA driver API has several hundred entry points (contexts,
  streams, events, graphs, IPC, textures, VMM, interop with OpenGL/Vulkan/DirectX,
  NVENC/NVDEC sessions). Keeping one faithful virtual device coherent across two
  physical devices touches all of them.

Semantic splitting is possible only where the *meaning* of an operation is known:
a GEMM, a convolution, an elementwise op. That is library level (§6, layer 3), not
driver level.

## 5. Prior art

| Project | What it does | Relevance |
|---|---|---|
| HAMi-core (Project HAMi) | `libcuda` hook that limits/accounts GPU memory and compute per container | Proves driver-API interception works for PyTorch/TensorFlow; model for layer 2 |
| rCUDA, GVirtuS | Forward CUDA API calls to a remote GPU | Interception + virtual devices; shows the latency cost of moving work away from data |
| ZLUDA | Drop-in CUDA driver on non-NVIDIA GPUs | A full driver-API reimplementation is possible but a multi-year effort |
| CUDA Unified Memory | Oversubscription with page migration to host RAM | Basis of layer 2's "no OOM" mode |
| Hugging Face Accelerate (`device_map="auto"`) | Splits a model's layers across GPUs + CPU | Proves layer-level splitting works; layer 1 applies it to *unmodified* apps |
| ComfyUI multi-GPU node packs | Put model components (text encoder, UNet, VAE) on different GPUs | Component-level pooling in the app most users run |
| DaVinci Resolve **Studio** | Native multi-GPU processing | Already solved by the vendor. The free version stays single-GPU, and a shim cannot change a closed renderer's kernels |

## 6. Proposed architecture: vMerge in three layers

```mermaid
flowchart TB
    app[Unmodified app<br/>ComfyUI · Forge · transformers · Whisper]
    subgraph L1[Layer 1 - framework interposer: vmerge run ...]
        hook[Hook torch.nn.Module.to / .cuda<br/>and model-loader entry points]
        planner[ASTRA planner<br/>speed-aware layer placement]
        disp[Dispatch submodules across GPUs;<br/>forward hooks move activations]
    end
    subgraph L2[Layer 2 - CUDA driver shim: libvmerge.so / proxy nvcuda.dll]
        virt[Stable virtual device(s), policy per app]
        oom[Overflow: cuMemAlloc to managed memory<br/>preferred on GPU, spill to host RAM]
        tele[Per-app VRAM/compute accounting]
        vmm[P2P-capable hardware only:<br/>VMM mapping across GPUs]
    end
    subgraph L3[Layer 3 - research spike]
        blas[Intercept cuBLAS GEMM; split large<br/>compute-bound GEMMs by columns]
    end
    app --> L1 --> L2 --> gpu[(GPU 0 ... GPU n)]
    L1 -. optional .-> L3
```

### Layer 1: framework interposer. Delivers the goal for PyTorch apps
* **Launch:** `vmerge run python main.py`, or a desktop shortcut. It injects a
  `sitecustomize` hook through `PYTHONPATH`, so the app is not modified.
* **Hook points:** `torch.nn.Module.to()` / `.cuda()` for modules above a size
  threshold, plus known loader entry points (diffusers pipelines, transformers
  `from_pretrained`, ComfyUI model management).
* When a model does not fit on the target GPU, the ASTRA planner (ADR-0010) computes
  a layer placement across all GPUs. vMerge moves submodules there and registers
  forward pre-hooks that move activations to the next GPU. Only activations cross
  PCIe.
* The app still sees `cuda:0`. `torch.cuda.mem_get_info()` can report pooled free
  memory, so the app's own "low VRAM" logic doesn't fight the pooling.
* **Limits:** models that share tensors across distant layers or use custom
  device-pinned buffers need per-model adapters (a registry, like ComfyUI node
  packs). Non-PyTorch apps are out of scope for this layer.

### Layer 2: CUDA driver shim. Makes every CUDA app safer; merging only where hardware allows
* **Interception:** `LD_PRELOAD=libvmerge.so` (Linux). On Windows, a proxy
  `nvcuda.dll` placed in the application directory (DLL search order) rather than
  replacing the system driver.
* **Features on stock GeForce:** one stable device view and GPU choice per app;
  overflow mode (`cuMemAlloc` → managed memory with preferred location = GPU, so
  allocations beyond VRAM spill to host RAM instead of failing); VRAM accounting and
  limits.
* **Feature on P2P-capable hardware only:** span allocations across GPUs with the VMM
  API. It is correct, but slow as §3 shows, so it is only for cold data.
* **Not a goal:** splitting kernel grids.

### Layer 3: research spike. Semantic splitting of heavy library calls
Intercept cuBLAS GEMM calls. When a GEMM is large and **compute-bound** (prefill,
training, big batches), split output columns across both GPUs, replicate the
smaller operand, and gather the result. Go/no-go measurement: the end-to-end speed-up
on a real prompt-processing workload must be ≥ 1.3× with bit-identical (or within
fp16 tolerance) results. Otherwise, drop it.

## 7. Go/no-go measurements (on the assembled ASTRA node)

| ID | Measurement | Tool | Decides |
|---|---|---|---|
| M1 | `cudaDeviceCanAccessPeer` for every GPU pair | `astra` probe extension | Whether layer 2 VMM spanning is possible at all |
| M2 | P2P and host-staged copy bandwidth between the GPUs | `nvbandwidth` / copy benchmark | Confirms the §3 numbers on real hardware |
| M3 | Unified-memory overflow slowdown when a model is 1.5× the GPU's VRAM | Layer-2 prototype + ComfyUI SDXL | Whether overflow mode is usable or only a crash-avoider |
| M4 | Layer-1 speed and memory: SDXL/Flux in ComfyUI and an 8B/14B transformers model, pooled vs single GPU | Layer-1 prototype | The product's value |
| M5 | Layer-3 GEMM split speed-up on prefill | Spike | Keep or drop layer 3 |

## 8. Risks

| Risk | Mitigation |
|---|---|
| Frameworks change internals (PyTorch, diffusers, ComfyUI) | Hook only public APIs; version-pinned adapter registry; CI against pinned versions |
| Proxy DLL blocked by apps that load `nvcuda.dll` by absolute path or check signatures | Layer 2 on Windows is best-effort; Linux first |
| Users expect "one 14 GB GPU = one fast GPU" | Report the estimated speed in the UI (the planner already estimates it) and never silently choose a slow path |
| NVIDIA license terms | Interpose only public, documented APIs; ship no NVIDIA binaries; no reverse engineering |

# Study + Test Report — Overcoming the "more GPUs, same speed" limit

| | |
|---|---|
| Report ID | ASTRA-QA-FT-003 / ASTRA-ARC-005 |
| Date | 2026-09-24 |
| Question | LIM-02 said: pooling GPUs adds memory but not single-user speed. Which technologies overcome that on GeForce cards, and which can ASTRA use now? |
| Method | Every option checked **three ways**: (1) published results, (2) a calculation from the ASTRA speed model, (3) a measurement on the dev PC where one GPU allows it |
| Dev PC | Windows 11, RTX 3060 Ti 8 GB, PCIe Gen4 x16, driver 610.88, llama.cpp b11149 |
| Verdict | **LIM-02 is not a hard limit.** Tensor parallelism inside one machine makes one answer faster: +40–110 % for large models over plain PCIe by the model (published: +16–35 % for a 27B), more with P2P or NVLink. It is now in ASTRA (ADR-0016). The "one CUDA device" limit (LIM-01) stays |

## 1. The options

| # | Technology | What it changes | Verdict |
|---|---|---|---|
| T1 | **Tensor parallelism** (llama.cpp `--split-mode tensor`, vLLM `--tensor-parallel-size`) | Every GPU reads 1/k of every layer at the same time; 2 all-reduces per layer | **Use.** Implemented: `astra plan --split tensor/auto`, measured A/B in `astra auto` |
| T2 | **NVLink bridge** (RTX 3090 / 3090 Ti, 2080 / 2080 Ti, TITAN RTX; not RTX 40/50) | Makes T1's all-reduces ~4× cheaper for a pair | **Use where available.** Detected as `NV#` by `nvidia-smi topo -m` |
| T3 | **PCIe P2P** via the open-kernel-module P2P patch (tinygrad and forks; RTX 30/40/50, Linux) | GPUs read each other's memory directly instead of through host RAM | **Optional, Linux only, unofficial.** Needs Resizable BAR with BAR1 ≥ VRAM and IOMMU passthrough. llama.cpp warns P2P "may cause crashes or corrupted outputs on some motherboards". Detected with `nvidia-smi topo -p2p r` = OK |
| T4 | **PCIe switch with P2P routing** (Gen4 Switchtec PM40100 / Broadcom PEX88096 boards) | Keeps T3 traffic inside the switch, not through the CPU root complex | **Designed into the ASTRA backplane PCB** (CR-007) |
| T5 | Speculative decoding (small draft model) | More tokens per pass of the big model | Done in v0.6.0: +21–38 % on slow pools only (FT-SPEC-01) |
| T6 | Faster network (25/100 GbE, RDMA) for the Fabric | Faster loading; per-token cost is mostly software | **Not for T1**: an all-reduce over the network measured ~0.9 ms (§3), so tensor split stays inside one machine |
| T7 | One virtual CUDA device ("vMerge" driver) | Would let any program see one big GPU | **Still no** (vMerge study): kernels cannot be split, remote memory is 35–100× slower |

## 2. Published evidence (method 1)

| Source | Setup | Result |
|---|---|---|
| [HyperQwen issue #40](https://github.com/syv-ai/HyperQwen/issues/40) | 2 × RTX 3090, **no NVLink**, PCIe 4.0 x8 each (topology PHB), vLLM 0.27.1, 27B 4-bit | Single-stream decode TP=2 vs one GPU: **+16 % to +27 %**; 8 streams **+10 % to +35 %** |
| [Himesh P., vLLM on 4 × RTX 3090](http://himeshp.blogspot.com/2025/03/vllm-performance-benchmarks-4x-rtx-3090.html) | Qwen2.5-7B, TP=2/4, PCIe Gen4 x8, with/without NVLink (batched throughput) | TP=2: 715 vs 483 output tok/s with/without NVLink (**+48 %**); TP=4 without NVLink 490 ≈ TP=2 |
| [llama.cpp multi-GPU docs](https://github.com/ggml-org/llama.cpp/blob/master/docs/multi-gpu.md) | — | `tensor` mode is **experimental**; needs flash attention and an f16/bf16/f32 KV cache; no MoE / Mamba; NCCL recommended; P2P optional but risky; "bottlenecked by GPU interconnect speed" |
| [tinygrad P2P kernel modules (forks)](https://github.com/aikitoria/open-gpu-kernel-modules) | RTX 30/40/50, Linux | P2P over BAR1 needs Resizable BAR with BAR1 ≥ VRAM and IOMMU passthrough |
| [c-payne PCIe gen4 switch](https://c-payne.com/products/pcie-gen4-switch-5x-x16-microchip-switchtec-pm40100) | Microchip Switchtec PM40100, 5 × x16 out, 2 × SlimSAS 8i in | An off-the-shelf reference for a P2P-capable GPU backplane |

## 3. Measurements on the dev PC (method 3)

| Test | Result | What it shows |
|---|---|---|
| 16 KB GPU → pinned host → GPU (an all-reduce's building block), median of 2000 | **14.2 µs** (4 KB 13.0, 64 KB 17.7; pageable memory 30–45 µs) | An all-reduce through host memory costs a few tens of µs: 2 per layer per token is affordable for large models, heavy for small ones |
| Bulk copy, pinned, 1 GiB | 25.1 GB/s H2D, 26.3 GB/s D2H | The slot trains at Gen4 x16 as reported |
| llama.cpp `--split-mode tensor` across CUDA0 + RPC0 (loopback to the same GPU), Llama-3.2-3B Q8_0 | Correct output ("Paris. The capital of Italy is Rome…"); **17.6 tok/s** vs ~98 tok/s on one GPU | The tensor mode works with ASTRA's flags; each all-reduce through the network path costs ~0.9 ms (56 per token), so tensor split must not cross machines |
| `nvidia-smi topo -p2p r`, `-q` | P2P matrix readable; **BAR1 = 256 MiB** on the 3060 Ti | Resizable BAR is **off** on this PC: T3 (P2P driver) would not work until it is enabled in the BIOS |

## 4. The ASTRA speed model (method 2)

Tensor split: `t = max_i(share_i × bytes / (bw_i × 0.75)) + 2 × layers × t_allreduce(link, k)`,
with `t_allreduce` = 45 µs (host, calibrated), 25 µs (P2P, estimate), 12 µs (NVLink,
estimate), × (1 + 0.5 (k − 2)) for more GPUs. Shares follow bandwidth, capped by memory.

| Check against the evidence | Model | Evidence |
|---|---|---|
| 32B Q4, 2 × RTX 3090, host all-reduce vs one GPU | +43 % (34.9 → 49.8 tok/s) | +16–35 % (27B 4-bit, with MTP speculation, which raises the base speed and so shrinks the gain) |
| 7B Q4, 4 × vs 2 × RTX 3090 over PCIe | 4 GPUs no faster (150 vs 174 tok/s) | TP=4 490 ≈ TP=2 483 |
| 3B split across the network (0.9 ms per all-reduce) | ~17.9 tok/s | 17.6 tok/s measured |

Predictions for 70B Q4 (these become hardware tests):

| Pool | Layer split | Tensor, host | Tensor, P2P | Tensor, NVLink |
|---|---|---|---|---|
| 2 × RTX 5090 | ~31 tok/s | ~43 | ~49 | (no NVLink on RTX 50) |
| 4 × RTX 3090 | ~16 | ~33 | ~42 | ~52 if all links were NVLink (bridges join pairs only) |

## 5. What changes in ASTRA (v0.8.0)

* `astra plan --split layer|tensor|auto --interconnect auto|host|p2p|nvlink`.
* `astra auto --split auto`: on 2+ local GPUs it benchmarks the tensor split next to the
  pipeline and keeps the faster one (≥ 5 % better), like speculative decoding.
* `astra probe` shows the P2P matrix, BAR1 size and the all-reduce path; `astra compat`
  flags Resizable BAR being off on multi-GPU hosts.
* The backplane PCB (CR-007) is specified to make T3/T4 possible: a P2P-capable Gen4
  switch, Resizable-BAR-friendly layout, per-slot power telemetry.

## 6. Limits that remain

* llama.cpp tensor mode is experimental: no quantized KV cache, no MoE. ASTRA measures it first.
* P2P on GeForce needs the unofficial Linux driver patch and Resizable BAR. Resizable BAR
  makes each GPU claim VRAM-sized address space, which **raises** the risk R-15 on
  consumer boards: prefer workstation boards or the PCB's switch for 3+ GPUs.
* vLLM tensor parallelism needs identical GPUs (the smallest sets every share) and a head
  count divisible by the GPU count.

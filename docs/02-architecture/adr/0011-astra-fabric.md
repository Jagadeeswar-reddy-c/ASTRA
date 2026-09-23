# ADR-0011: ASTRA Fabric: exo-style GPU pooling across machines on NVIDIA

* Status: Accepted (CR-003, sponsor delegated the design: "do it the optimum way")
* Deciders: Enterprise Architect, Solution Architect
* Related: ASTRA-ARC-003 (vMerge feasibility), ADR-0003, ADR-0004, ADR-0010

## Context
The sponsor's goal: make several GPUs behave as one, "the way people did it for Macs"
(exo / MLX-distributed clusters of Apple-silicon machines) and the way NVIDIA does
multi-GPU. The feasibility study showed that a driver-level "one fake GPU" either
cannot work on GeForce (no P2P) or runs 40–100× slower. The Mac clusters succeed for
a different reason: they **partition the model, not the memory**. Every machine holds
a contiguous slice of layers, only activations travel between machines, and one
endpoint serves the whole ring. NVIDIA's own pattern (NCCL + NVLink + tensor
parallelism) depends on NVLink/P2P, which consumer cards lack, so pipeline
partitioning is the optimum for this hardware.

## Decision
1. **Same pattern as the Mac clusters, built from NVIDIA-native parts:**

   | Concern | Mac cluster (exo / MLX) | ASTRA Fabric |
   |---|---|---|
   | Node process | exo node | `astra agent` (one per machine) |
   | Discovery | automatic on the LAN | UDP broadcast `ASTRA-DISCOVER astra-fabric/1` (port 9836) or a static peer list |
   | Capabilities | device memory/FLOPS | `/v1/node` descriptor + GPU catalog (VRAM, bandwidth, cc, power) |
   | Partitioning | ring, memory-weighted | ADR-0010 speed-aware planner with per-link hop costs (PCIe 0.15 ms; network 2 ms default) and subset selection |
   | Execution | MLX / tinygrad kernels | **llama.cpp CUDA kernels**: local GPUs directly, remote GPUs via llama.cpp's RPC backend (`rpc-server` per GPU, `--rpc` + `--device` + `--tensor-split`) |
   | API | ChatGPT-compatible | OpenAI-compatible `llama-server` on the head node |
2. **Why llama.cpp RPC instead of a new runtime:** mature CUDA kernels for every
   generation ASTRA supports (Maxwell → Blackwell, incl. the GT 1030), GGUF
   quantizations, uneven layer splits and a remote-device protocol already exist.
   ASTRA supplies what they lack: discovery, capability and compatibility analysis,
   placement, supervision, validation and monitoring.
3. **One `rpc-server` per GPU**, pinned by UUID (ADR-0004), supervised and restarted
   by the agent, with `--cache` so weights reload from the node's disk instead of
   the network.
4. **Topology preference.** Local GPUs come first, remote GPUs are grouped per node,
   and each machine boundary costs a network hop per token. The planner therefore
   uses remote GPUs only when a model needs them or when they make generation
   faster.
5. **Head node may be GPU-less.** The new mini PC can host the chassis GPUs *and*
   drive GPUs in other PCs, or act purely as the API/orchestration node.
6. **vLLM stays local-only** (multi-node vLLM needs Ray + NCCL networking; out of
   scope for v1).
7. **Transparent use by unmodified apps** (the vMerge goal) continues as layer 1 of
   ASTRA-ARC-003: a PyTorch interposer that uses the same planner. That is the next
   increment.

## Consequences
* \+ Any mix of NVIDIA GPUs across any number of machines behaves as one model
  server: the chassis GPUs and the desktop's RTX 3060 Ti can serve one 14B model
  together.
* \+ Same planner, validation and telemetry for local and remote GPUs.
* − **Security:** llama.cpp's rpc-server is unauthenticated. The fabric must run on
  a trusted, isolated LAN/VLAN. The deployment guide gives the firewall rules; the
  agent API is read-only.
* − The network adds latency per token per machine boundary (~1–3 ms on gigabit),
  and first-time model loads send weights over the LAN (a 5 GB slice takes ~45 s on
  1 GbE, then is cached). 2.5 GbE or faster is recommended.
* − The runtime gate (R01–R05) validates the GPUs of the node it runs on. Each node
  runs its own `astra validate`.

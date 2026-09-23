# ADR-0003: Pipeline-parallel VRAM pooling; llama.cpp primary engine

* Status: Accepted (G2)
* Deciders: Enterprise Architect, Solution Architect
* Resolves: DR-01, DR-02, DR-08, DR-13

## Context
The GPUs differ in VRAM (6 GB vs 8 GB) and architecture (Turing vs Ampere). They
have no NVLink and no peer-to-peer. The concept document proposes splitting by the
VRAM ratio, but its Phase 3 command uses tensor parallelism, which splits evenly
and all-reduces every layer.

## Decision
1. **Pool memory with pipeline (layer) parallelism.** Each GPU holds a *contiguous*
   run of transformer blocks plus their KV cache. Only one hidden state per token
   crosses each stage boundary.
2. **Split with a min-max optimisation, not a fixed ratio.** For GPU *i* with usable
   memory `U_i = capacity_i − reserve_i`, choose contiguous block counts that
   minimise `max_i (fixed_i + Σ blocks_i (weights + KV)) / U_i`. `fixed_i` covers the
   embeddings on the first stage (vLLM) and the output head on the last stage. A
   dynamic programme over (GPU, first block) solves it exactly in O(k·n²). When
   fixed costs are negligible this reduces to the concept rule VRAM_i / VRAM_total.
3. **llama.cpp (`llama-server --split-mode layer --tensor-split …`) is the primary
   engine.** It supports Turing, uneven splits, GGUF quantizations and Windows.
   **vLLM is secondary**: `--pipeline-parallel-size k --tensor-parallel-size 1` with
   `VLLM_PP_LAYER_PARTITION`, `--dtype float16`, AWQ/GPTQ models, and a version
   pinned after TC-RT-02 passes.
4. The planner **warns when one GPU is enough**, because pooling then only adds
   latency.

## Consequences
* \+ Uses all 14 GB, e.g. Qwen2.5-14B Q4_K_M fits pooled and fits neither GPU alone.
* \+ Inter-GPU traffic is negligible, so the x4 uplink without P2P is fine.
* − Single-request decode runs through the stages in sequence (no speed-up from the
  second GPU, only capacity). Concurrency recovers throughput.
* − Planner estimates for catalog models are approximate. GGUF planning is exact
  for weights; compute buffers are covered by the reserve and verified at runtime
  (R03).

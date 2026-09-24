# ADR-0016: Tensor-parallel split inside one machine

* Status: Accepted (v0.8.0)
* Deciders: Solution Architect, Enterprise Architect, QA Lead
* Supersedes in part: ADR-0003 (pipeline split as the only mode), LIM-02 ("hard limit")

## Context
ADR-0003 chose pipeline (layer) splitting because GeForce cards lack P2P and NVLink
(except a few), so tensor parallelism looked unaffordable. The Sponsor asked how to
overcome the resulting limit: more GPUs added memory but not speed. The study
[ASTRA-QA-FT-003](../../05-testing/reports/tensor-parallel-study.md) checked the options
three ways. Published runs show vLLM TP=2 on two RTX 3090s over plain PCIe at +16–35 %
single-stream. The dev PC measured a 14 µs GPU→host→GPU round trip. llama.cpp now ships
an experimental `--split-mode tensor`, verified working with ASTRA's flags.

## Decision
1. The planner offers two split modes: `layer` (pipeline, default, works across
   machines) and `tensor` (all GPUs on every layer, **one machine only**).
2. Tensor speed model: the slowest GPU's share of the weights read per token, plus
   2 all-reduces per layer: 45 µs via host memory (calibrated), 25 µs P2P, 12 µs NVLink
   (estimates until TC-HW-14), growing with the GPU count. Shares follow memory bandwidth
   and are capped by each GPU's free memory; vLLM splits evenly.
3. The link class is detected on any OS from `nvidia-smi topo -m` (NV# = NVLink) and
   `nvidia-smi topo -p2p r` (OK = P2P); BAR1 size shows whether Resizable BAR is on.
4. `astra plan --split auto` picks the faster estimate. `astra auto` **measures** the
   tensor split next to the pipeline on 2+ local GPUs and keeps it only if it is ≥ 5 %
   faster (ADR-0015 rule), because llama.cpp's tensor mode is experimental.
5. Constraints enforced: f16 KV cache and flash attention for llama.cpp tensor mode; no
   remote GPUs; vLLM head counts divisible by the GPU count.

## Consequences
* LIM-02 becomes a soft limit: large models on 2–4 GPUs in one box gain roughly
  1.4–2× by the model (2 × RTX 3090, 32B: +43 %), less for small models.
* Hardware choices matter more: NVLink pairs (RTX 3090) and P2P-capable PCIe switches
  (CR-007 backplane) raise the gain; Resizable BAR must be on for P2P.
* The Compose stack keeps the layer split; tensor split runs through `astra auto` /
  `astra launch` for now.

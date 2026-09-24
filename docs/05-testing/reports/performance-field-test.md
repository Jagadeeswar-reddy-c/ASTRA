# Test Report — Performance levers on real hardware (FT-SPEC-01)

| | |
|---|---|
| Report ID | ASTRA-QA-FT-002 |
| Date | 2026-09-24 |
| System | Windows 11, RTX 3060 Ti 8 GB (448 GB/s, display GPU), driver 610.88 |
| Engine | llama.cpp b11149 (CUDA 12.4), `llama-server` |
| Models | Qwen2.5-7B-Instruct Q4_K_M (target), Qwen2.5-0.5B-Instruct Q8_0 (draft), Llama-3.1-8B-Instruct Q4_K_M |
| Question | Which engine settings make ASTRA answer faster (more tokens per second, fewer seconds to the first token), and when? |
| Verdict | Speculative decoding helps **only when the target model is slow**; ASTRA now measures it and keeps it only if it wins. A q8_0 KV cache saves memory at ~1 % speed. |

## Method

Each configuration started a fresh `llama-server` (context 8192, all layers on the GPU
unless stated). Decode speed: mean of 3 different prompts (explanation, code, list), 256
tokens each, default sampling and greedy (temperature 0). Prefill: one ~1,700-token
prompt. VRAM: `nvidia-smi` used memory after loading minus before. Script:
`specbench.py` (kept with the raw logs in the lab folder).

"Slow target" = the same 7B model with only 16 of 28 layers on the GPU (`-ngl 16`),
which runs at ~20 tok/s, standing in for a large model spread over several GPUs.

## Results

| Configuration | VRAM | Decode tok/s (default / greedy) | Draft acceptance | Prefill tok/s |
|---|---|---|---|---|
| 7B, f16 KV (baseline) | +4888 MiB | **73.4 / 74.5** | — | 2828 |
| 7B, q8_0 KV + flash attention | +4691 MiB (−197) | 73.1 / 72.8 | — | 2680 |
| 7B + 0.5B draft, builds' default settings (n-max 3, p-min 0.00) | +5583 MiB | 55.6 / 64.9 | 0.45 / 0.55 | 2530 |
| 7B + draft, n-max 8, p-min 0.00 | +5541 MiB | 55.6 / 60.1 | 0.28 / 0.30 | 2553 |
| 7B + draft, n-max 16, p-min 0.00 | +5541 MiB | 44.8 / 48.1 | 0.15 / 0.17 | 2560 |
| 7B + draft, **n-max 4, p-min 0.75** | +5536 MiB | 62.5 / 67.5 | **0.88 / 0.86** | 2543 |
| 7B + draft, n-max 8, p-min 0.75 | +5541 MiB | 71.0 / 61.1 | 0.83 / 0.84 | 2544 |
| 7B + n-gram lookup (no draft model) | +4914 MiB | 73.8 / 73.7 | — | 2829 |
| **Slow** 7B (16/28 layers on GPU), baseline | +3014 MiB | **21.8 / 19.9** | — | 1224 |
| Slow 7B + draft, n-max 3, p-min 0.00 | +3659 MiB | 24.4 / 24.5 | 0.48 | 1181 |
| **Slow 7B + draft, n-max 4, p-min 0.75** | +3659 MiB | **26.3 / 27.4 (+21 % / +38 %)** | 0.89 / 0.88 | 1190 |
| Slow 7B + draft, n-max 8, p-min 0.75 | +3659 MiB | 23.5 / 27.8 | 0.84 / 0.87 | 1189 |
| Slow 7B + draft, n-max 16, p-min 0.75 | +3659 MiB | 24.4 / 26.2 | 0.86 / 0.83 | 1194 |

`astra auto` end to end (v0.6.0, same PC):

| Run | Result |
|---|---|
| `astra auto --draft on --no-ui` | Chose Llama-3.1-8B Q4_K_M (its 1B draft did not fit with margin, so none); **74.2 tok/s vs 65 estimated (+14 %)**, time to first token for a 2k prompt ~0.78 s, gate 5/5 PASS |
| `astra auto --model qwen2.5-7b --quant q4_k_m --draft on --no-ui` | Baseline 75.9 tok/s; with draft 78.1 tok/s at 95 % acceptance (+3 %, below the 5 % bar), so **ASTRA went back to the baseline**; 75.9 vs 73 estimated (+4 %); gate 5/5 PASS |

## Findings

| ID | Finding | Action |
|---|---|---|
| PF-01 | On a fast single GPU (~75 tok/s) a 0.5B draft costs about as much per token as it saves (~4–5 ms per drafted token, dominated by kernel launches), so speculative decoding is −15 % to +3 % | ASTRA suggests it only below 30 tok/s (`SPEC_MAX_TPS`) and `astra auto` keeps it only if it measures ≥ 5 % faster |
| PF-02 | When the target is slow (~20 tok/s), the draft gives **+21 % to +38 %** | Default in `astra auto` for slow pools (big models across several GPUs, where verifying 5 tokens costs the same memory reads as 1) |
| PF-03 | This llama.cpp build's defaults (`--spec-draft-p-min 0.00`) waste drafts: acceptance 0.15–0.45. With p-min 0.75 it is 0.83–0.89 | ASTRA always passes n-max 4 and p-min 0.75 |
| PF-04 | Newer llama.cpp builds only speculate with `--spec-type draft-simple`; with `-md` alone the draft loads but is never used | ASTRA reads `llama-server --help` and emits the right flags (current or legacy) |
| PF-05 | The 0.5B Q8_0 draft at 8k context takes 645 MiB | Planner formula (blocks + head + KV + 64 MiB) gives ~660 MiB; the draft's memory is placed on the first local GPU and checked by R03 |
| PF-06 | q8_0 KV cache: −197 MiB at 8k context for a 7B (−1.25 GiB for a 70B), −1 % speed | `astra auto` uses it only when the model does not fit otherwise |
| PF-07 | `astra plan --kv-type q8_0` planned the smaller cache but never passed it to the engine (defect since v0.1), nor did the Compose stack | Fixed: `--cache-type-k/v` and `--flash-attn on` in the launch command, `ASTRA_KV_TYPE` in Compose |
| PF-08 | Time to first token is short on one GPU (0.7 s for a 2k prompt) and is now measured and reported by `astra auto` | Estimate for multi-GPU pools remains open (LIM-18) |

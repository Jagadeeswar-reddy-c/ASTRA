# ADR-0015: Speed features are measured, not assumed

* Status: Accepted (v0.6.0)
* Deciders: Solution Architect, QA Lead

## Context
llama.cpp offers several ways to trade memory or compute for speed: speculative decoding
with a small draft model, n-gram lookup, a quantized KV cache. Their effect depends on
the pool. The field test [FT-SPEC-01](../../05-testing/reports/performance-field-test.md)
measured speculative decoding at **−15 % to +3 %** on a fast single GPU and **+21 % to
+38 %** on a slow target, with the engine's default draft settings wasting most drafts.

## Decision
1. **Speculative decoding** (llama.cpp only). The catalog pairs each model with a draft
   that shares its tokenizer (Qwen2.5 → 0.5B, Llama 3.x → 3.2 1B). The planner places
   the draft on the first local GPU (never over the network) and counts its memory
   (blocks + head + KV + 64 MiB; measured within 3 %) but not its reads in the speed
   estimate. It is suggested only below 30 tok/s. Fixed settings: at most 4 drafted
   tokens, stop drafting below p = 0.75.
2. `astra auto --draft auto` (default) tries the draft on slow pools and **keeps it only
   if it measures ≥ 5 % faster** than the baseline in the same run; `on` always tries
   it, `off` never.
3. **KV cache**: f16 by default; q8_0 (with flash attention) only when the model does
   not fit otherwise. The launch command and the Compose stack carry the planned type.
4. ASTRA reads `llama-server --help` to use current or legacy flag names.
5. `astra auto` reports prefill speed and time to first token for a 2k-token prompt.

## Consequences
* A plan never promises a speed-up it has not measured; `results.json` records both runs.
* The first `astra auto` on a slow pool loads the model twice (baseline + draft).
* The Compose stack does not run the draft yet (noted in the generated `.env`).

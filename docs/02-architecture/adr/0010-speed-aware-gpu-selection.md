# ADR-0010: Speed-aware placement and automatic GPU-set selection

* Status: Accepted (CR-001). **Amends ADR-0003 §2 and §4.**
* Deciders: Solution Architect, Enterprise Architect

## Context
ADR-0003 splits layers so that memory pressure is even (min-max utilisation). That
suits two similar cards, but not an arbitrary mix. Single-stream decoding is
memory-bandwidth bound: every token reads each stage's weights, so a stage on a
48 GB/s GT 1030 costs about 7× as much per byte as one on a 360 GB/s RTX 3060. An
even memory split can therefore make a pool slower than its best card alone.

## Decision
1. **Two objectives** (`planner.objective`, `--objective`):
   * `balanced`: the ADR-0003 min-max split (the concept rule when fixed costs are
     negligible).
   * `speed` (default): compute the Pareto front of (Σ bytes_i / bandwidth_i, worst
     utilisation) over contiguous splits that keep every GPU ≤ **92 %** full, then
     take the most even split within **3 %** of the fastest time. Similar cards share
     the load; a slow card takes only what must spill onto it.
2. **Automatic set selection** (`--gpus auto`, the default): evaluate every subset of
   the eligible GPUs (engine-capable, not excluded, ≤ 8 GPUs) and rank by: fits →
   fits with margin (≤ 92 %) → estimated decode speed. Speeds within **5 %** count
   as equal, and then the set with fewer GPUs wins (fewer hops and failure points).
   If nothing fits, the plan covers all eligible GPUs so the shortfall is visible.
   The top alternatives are printed and saved in the plan JSON.
3. **Speed estimate** (reported, not promised): `1 / Σ_i (weights_i + fixed_i + KV_i/2)
   / (0.75 × bandwidth_i)` plus 150 µs per stage boundary. This is a single-stream
   ceiling; batching raises throughput.
4. `--gpus all` and explicit `--gpus 2,0` bypass selection. An explicit list's order
   is the pipeline order.

## Consequences
* \+ Worked example (GT 1030 + GTX 1660 SUPER + RTX 3060 12 GB):
  * Llama-3.1-8B Q4 → **RTX 3060 alone** (~52 tok/s est.). Adding the GT 1030 would
    drop it to ~43 tok/s.
  * Qwen2.5-14B Q4 → **GTX 1660 SUPER + RTX 3060**. The 3060 alone would be 95 %
    full; the pair runs at ~65 % and ~28 tok/s est.
  * The GT 1030 joins only when a model cannot fit without it.
* \+ The runtime gate (R03) is unchanged: it compares against the plan's own
  `planned_share`.
* − Subset search is exponential. It is capped at 8 GPUs (255 subsets), which is
  fine for a 4-slot chassis.
* − Estimates depend on catalog bandwidths. GPUs without one fall back to the
  `balanced` split with no speed estimate.

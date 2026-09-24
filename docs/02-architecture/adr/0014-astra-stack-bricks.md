# ADR-0014: ASTRA Stack — 1-GPU bricks on a PCIe star

* Status: Accepted (CR-006, Sponsor 2026-09-24: "yes, continue with the plan")
* Deciders: Enterprise Architect, Solution Architect, Hardware Eng.

## Context
The Sponsor asked for DGX Spark-style modules that stack into "one GPU". The study
[ASTRA-ARC-004](../studies/astra-stack-modules.md) found that neither ASTRA nor DGX Spark can
present several GPUs as one CUDA device, but that identical, self-powered modules can
form one pool with one endpoint, and that a PCIe stack is 10–15 % faster per token
than the same GPUs spread across machines.

## Decision
| # | Decision |
|---|---|
| D1 | Inside one host, bricks connect in a **star**: one OCuLink cable per brick from a hub card in the host. No daisy chains (a failed brick must not take others down) |
| D2 | A brick holds **one GPU**, its own PSU (started by the hub's sync signal), a passive OCuLink→x16 receiver and a fan. The existing 2-GPU chassis counts as a double brick |
| D3 | The hub is a **PCIe switch card** (works on any board, no BIOS dependency) unless the head PC's BIOS supports x4x4x4x4 bifurcation (CR-002 decides the cheaper passive hub) |
| D4 | Sizing target: **70B Q4 at ≥ 8 tok/s**. `astra size` says which bricks reach it (e.g. 3 × 24 GB at ~16 tok/s, 4 × RTX 5060 Ti 16 GB at ~7.8 tok/s plus ~20–40 % from speculative decoding) |
| D5 | Order: 32B/70B catalog → `astra size` → module awareness (identity, per-brick power, stack link check, console grouping) → brick hardware after the chassis qualification (BL-03) |
| — | Several stacks join through ASTRA Fabric (ADR-0011), as DGX Spark units pair over their NIC |

## Consequences
* One stack is one PCIe tree: no network hop inside it, and the planner's existing local
  pipeline model applies unchanged.
* Many GPUs on one consumer board can exhaust MMIO/BAR space (risk R-15): qualify four
  bricks on the chosen head PC before committing to a hub.
* No hot-plug (LIM-05): add a brick with the power off; `astra auto` re-plans.
* Power is budgeted per brick, not per chassis (FR-21).

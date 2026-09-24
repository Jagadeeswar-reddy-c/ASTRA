# CR-006 — ASTRA Stack: stackable GPU modules

| | |
|---|---|
| Raised | 2026-09-24 by the Sponsor ("like DGX Spark, maybe we can make modules to stack them into a single GPU") |
| Status | **Approved** 2026-09-24 (Sponsor: "yes, continue with the plan"; D1–D5 as recommended, ADR-0014). Delivered in v0.6.0: S6 (32B/70B models), S4 (`astra size`); next: S1–S3, S5 |
| Study | [ASTRA-ARC-004](../../02-architecture/studies/astra-stack-modules.md) |
| Affects | PRD (new FR-21 module awareness, FR-22 sizing), hardware design (brick + hub), config `[[module]]`, checks L12/L13, exporter, console, planner catalog |

## Proposal
Build ASTRA as **identical 1-GPU bricks** (GPU + own PSU + OCuLink receiver) wired as a
**star** to a hub card in the host. The stack is one pool: one model, one endpoint,
one console. Several stacks join over ASTRA Fabric, the way DGX Spark units pair
over ConnectX-7.

Not in scope: presenting the stack as a single CUDA device. The study (§2, §5) shows
neither this design nor DGX Spark can do that; vMerge layer 1 (BL-01) covers PyTorch apps.

## Impact analysis

| Area | Change |
|---|---|
| Requirements | FR-21: detect bricks and budget power per brick. FR-22: size a stack for a target model and speed |
| Architecture | New ADR once D1–D3 are decided (star topology, brick definition, hub type) |
| Hardware | Brick design (receiver board, SFX PSU, sync relay, fan, earth bond); hub card qualification; test with 4 bricks for MMIO/BAR limits (R-15) |
| Software | BL-16 items S1–S6 (study §6); 32B/70B catalog models (BL-02) |
| Tests | New TC-SW for grouping, per-brick L12, L13, sizing; TC-HW for a 4-brick stack |
| Schedule | Software S6 and S4 first (no hardware needed); hardware after the chassis qualification (BL-03) |
| Risk | R-15 (MMIO / BAR exhaustion with many GPUs) |

## Decisions requested
See study §7: D1 star vs cascade · D2 GPUs per brick · D3 switch vs bifurcation hub ·
D4 target model and speed · D5 order of work.

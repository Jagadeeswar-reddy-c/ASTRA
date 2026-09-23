# CR-001 — Support any NVIDIA GPU mix (not only RTX 2060 + RTX 3050)

| | |
|---|---|
| Raised | 2026-09-23 by the Sponsor |
| Status | **Approved** by the Enterprise Architect, 2026-09-23. Implemented in v0.2.0 |
| Affects | PRD (FR-11, FR-13–15, NFR-03, NFR-11), architecture (ADR-0009, ADR-0010), hardware design §1/§3, BOM-05, test cases, runbooks, deployment guide |

## Request

"The RTX 2060 and RTX 3050 were an example. I don't yet know which GPUs I'll connect.
They will be NVIDIA, anything from a GT 1030 to a GTX or RTX card."

## Impact analysis

| Area | Impact | Change |
|---|---|---|
| Requirements | The v1.0 PRD treated the 2060/3050 pair as fixed. The GPU set is now an input, not a constant | FR-11 extended (simulate by model name); new FR-13 (compatibility gate), FR-14 (automatic, speed-aware GPU selection), FR-15 (power budget from the installed GPUs); NFR-03/NFR-11 re-expressed per GPU set |
| Architecture | A mix of generations brings hard constraints: one driver for all cards (Kepler ≤ R470; Maxwell/Pascal/Volta ≤ R580), CUDA 12 builds for sm < 75, vLLM needs cc ≥ 7.0, very different memory bandwidth (48 → 1,792 GB/s) | ADR-0009 (GPU capability catalog + compatibility analysis), ADR-0010 (speed-aware placement and set selection, amending ADR-0003 §2/§4) |
| Hardware | PSU sizing depends on the cards: 2 × GTX 1650 ≈ 175 W, while 2 × RTX 3090 needs ≥ 1,200 W. x4/x8 cards and slot-powered cards | Hardware design §3.1 now gives a formula plus examples; BOM-05 sized by `astra compat` |
| Software | New `hardware/gpu_specs.py` and `hardware/compat.py`; planner objective and subset search; `astra compat`; `probe --emit-config`; link-gate checks L11 (compatibility) and L12 (PSU budget); exporter power metrics | v0.2.0 |
| Tests | New TC-SW-16…19 and TC-HW-12 (GPU change procedure) | 143 automated tests |
| Operations | New alert `AstraChassisUndersized`; power alert now relative to the configured PSU; runbook RB-07 (GPU missing after a swap / driver window) | |
| Cost | The BOM is unchanged except the PSU line, whose price now depends on the GPUs | NFR-11 re-worded |
| Schedule | Hardware phases are unaffected. Before buying GPUs, run `astra compat --simulate "<cards>"` | |

## Key finding for the sponsor

The development workstation runs **driver R610**. Under NVIDIA's lifecycle, R580 is
the last branch for Maxwell, Pascal and Volta, so **a GT 1030 or any GTX 9xx/10xx card
will not work on this driver.** If one goes into the chassis, the whole host must stay
on the **R580 branch**. Turing (GTX 16xx, RTX 20xx) and newer have no such ceiling.
`astra compat` reports this automatically (check L11).

## Decision

Approved. The reference build (2060 + 3050) remains the documented example and the
`--simulate reference` preset.

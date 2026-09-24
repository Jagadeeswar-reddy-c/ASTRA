# CR-007 — ASTRA Backplane PCB

| | |
|---|---|
| Raised | 2026-09-24 by the Sponsor ("we will design a powerful PCB for this to mount it easily") |
| Status | **Proposed**: requirements drafted; decisions B1–B4 open |
| Spec | [ASTRA-HW-004 backplane-pcb-spec.md](../../03-solution-design/backplane-pcb-spec.md) |
| Affects | Hardware design (replaces the no-name PEX8747 board, risk R-09), ADR-0014 hub, BOM, TC-HW-14/15, exporter (per-slot power), R-15, R-16 |

## Proposal
One in-house PCB that carries the GPUs and makes them fast together:
* a **Gen4 PCIe switch** with P2P routing between its GPU ports (tensor split, ADR-0016);
* **four x16 slots** spaced for 2–3-slot GPUs, or OCuLink/SlimSAS ports for bricks;
* **per-slot power switching and measurement**, PSU sync, fan control, status LEDs;
* a small **management MCU** on USB that reports to `astra` (measured watts per GPU,
  closing LIM-25).

## Impact
| Area | Change |
|---|---|
| Requirements | HW-BP-01…25 (spec); new FR-25 (per-slot power telemetry) |
| Architecture | ADR to follow after B1–B3; hub of ADR-0014 becomes this board |
| Software | `astra` reads the board over USB serial (telemetry, slot power, fan); P2P-enable check |
| Tests | TC-HW-14 P2P bandwidth/latency, TC-HW-15 power telemetry accuracy, TC-HW-16 signal integrity (Gen4 x16, zero AER under load) |
| Cost / risk | Gen4 switch silicon and a 12–14-layer controlled-impedance board: new risk R-16 (first-spin failure). Mitigation: validate the architecture on an off-the-shelf PM40100 board first (Phase A) |

## Decisions requested
| # | Decision | Recommendation |
|---|---|---|
| B1 | Build order | **Phase A**: buy an off-the-shelf Gen4 switch board (e.g. PM40100-based) and qualify P2P + tensor split on real GPUs. **Phase B**: design our PCB with what Phase A proves |
| B2 | Switch chip | Microchip Switchtec PM40084/PM40100 (documented P2P, per-port config) over Broadcom PEX88096 (NDA-heavy) |
| B3 | Form factor | Backplane with 4 × x16 slots (2-slot spacing + riser option) and 2 × OCuLink 8i for external bricks |
| B4 | Host link | 2 × SlimSAS 8i (x16 Gen4) from a host adapter card; x8 fallback for mini PCs |

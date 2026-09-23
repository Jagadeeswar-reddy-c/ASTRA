# ASTRA — Hardware Solution Design

| | |
|---|---|
| Document ID | ASTRA-SDD-HW-001 |
| Version | 1.0 (2026-09-23) |
| Status | Draft — BOM frozen at Gate G3 |
| Owner | Hardware Engineer · Approver: Enterprise Architect |
| Traces to | NFR-01…05, NFR-11; ADR-0001, 0002; DR-03…06, DR-10, DR-12, DR-14, DR-15 |

## 1. Bill of materials

The source of truth is [`hardware/bom.csv`](../../hardware/bom.csv). Its total is
**€297–438**, or €285–420 excluding the one-off tooling (NFR-11). Compared with the
concept document, the BOM adds a chassis-side OCuLink receiver (BOM-03), a
specified backplane power-feed rating (BOM-04), a shared power strip (BOM-09),
strain relief (BOM-10), and cold-check tooling (BOM-11).

## 2. Interconnect

| Item | Design value | Rationale |
|---|---|---|
| Host slot | M.2 Key-M, PCIe x4, **CPU-attached** preferred | A chipset-attached slot shares DMI bandwidth; check the board manual's lane-sharing table, because populating some M.2 slots disables SATA ports (DR-15) |
| Adapter | M.2 → OCuLink with ReDriver | Restores eye margin over the cable |
| Cable | SFF-8611, shielded; **0.5 m preferred**, 1.0 m max | Gen3 has comfortable margin at 1 m; a future Gen4 switch needs ≤ 0.5 m (DR-14) |
| Uplink | Trains **Gen3 x4** (PEX8747 is Gen3) ≈ 3.94 GB/s theoretical, ≥ 3.0 GB/s measured | DR-03, NFR-05 |
| Downstream | RTX 2060 Gen3 x16; RTX 3050 Gen3 **x8** (the card is x8 electrically) | L06 compares against each card's own maximum |
| Hot-plug | **Not supported** — connect/disconnect only with both systems off | DR-06 |

## 3. Electrical design

### 3.1 Power budget: depends on the GPUs installed (CR-001)

```
sustained = Σ board_power(GPU) + overhead (25 W: switch, receiver, fans)   ≤ 0.70 × PSU
peak      = Σ board_power(GPU) × 1.3 + overhead                            ≤ 1.00 × PSU
```

`astra compat` (or `astra compat --simulate "…"` before buying) computes both from
the GPU catalog and recommends the smallest standard PSU. Check L12 enforces it at
every boot. Examples:

| GPUs in the chassis | Σ board power | Sustained / peak | Minimum PSU |
|---|---|---|---|
| GT 1030 + GTX 1650 (both slot-powered) | 105 W | 130 / 162 W | 450 W |
| RTX 2060 + RTX 3050 8 GB (reference) | 290 W | 315 / 402 W | 450 W (650 W fitted) |
| GT 1030 + GTX 1660 SUPER + RTX 3060 | 325 W | 350 / 448 W | 550 W |
| 2 × RTX 3090 | 700 W | 725 / 935 W | **1,200 W** |

**Backplane feed:** every GPU draws up to 75 W of slot power through the backplane's
own 12 V input, and slot-only cards (GT 1030, GTX 1050/1650) draw *all* their power
there. Use every power input the backplane provides. A single 6-pin feed (75 W) is
**not sufficient** for two slots. `astra compat` reports the total slot power.

**Card form factors:** some cards are electrically x4 (GT 1030) or x8 (RTX 3050,
RTX 4060/5060 families). They work in the x16 slots, and L06 compares each card
against its own maximum width.

### 3.2 Single-source rule (NFR-04)

Each GPU draws power from exactly one PSU, the chassis PSU: slot power through the
backplane and auxiliary power through its 8-pin connector. The host PSU connects to
the chassis only through the relay's sense lead. If a GPU is fed from two PSUs, the
PSUs' 12 V rails are paralleled through the card and circulating currents flow.

### 3.3 Power sequencing

1. The host powers on. Its SATA/Molex rail energises the relay sense input (check
   the relay module's datasheet for which rail it senses).
2. The relay shorts PS_ON (24-pin, pin 16, green) to COM on the chassis PSU.
3. The chassis reaches PWR_OK in ~100–500 ms, well before the host BIOS enumerates
   PCIe (seconds). TC-HW-03 verifies the skew is ≤ 1 s, and TC-HW-10 verifies that
   10 cold boots all enumerate.
4. When the host shuts down, the relay opens and the chassis powers off with it.
5. **Host sleep and hibernate are disabled** (DR-05), because S3/S4 drops the sense
   rail and powers off the GPUs.

### 3.4 Grounding and earthing (DR-10)

* Plug **both PSUs into the same earthed power strip (BOM-09)**. This gives the host
  and the chassis the same protective-earth reference, so no ground current flows
  through the PCIe signal grounds.
* Signal-ground continuity comes from the OCuLink cable's dedicated ground
  conductors. The cable shield is for EMI only; do not rely on it as a return path,
  and do not add ad-hoc DC ground straps between the two chassis.
* Bond the aluminium frame to the chassis PSU's earth (through its mounting screws).
  TC-HW-11 checks continuity (< 0.1 Ω) between the frame and the PSU earth pin.

## 4. Mechanical and thermal

* Leave at least one empty slot pitch between the GPUs. Place the two 120 mm fans
  to push room air across both GPU intakes and the switch heatsink.
* Route the OCuLink cable with strain relief at both ends and a bend radius of at
  least 10× the cable diameter. It must not be under tension when the host is moved.
* Design ambient ≤ 30 °C. Acceptance: < 83 °C sustained at full load and no
  thermal slowdown (NFR-02, TC-HW-07).

## 5. Host firmware (BIOS/UEFI) checklist

| Setting | Value | Why |
|---|---|---|
| Above 4G Decoding | **Enabled** | A switch plus two GPUs need 64-bit MMIO (DR-12) |
| Resizable BAR | Enabled; disable if GPUs fail to enumerate | Some switch boards mishandle large BARs |
| CSM / legacy boot | Disabled (UEFI only) | Required for 64-bit MMIO allocation on many boards |
| M.2 slot link speed | Auto; force **Gen3** if AER errors appear | The switch is Gen3 anyway; a lower rate gives more margin |
| Native ASPM | Disabled if the link gate shows corrected errors at idle | Link power-state transitions are a common error source on external links |
| Fast Boot | Disabled | Gives the chassis time to come up before enumeration |
| ErP / deep sleep, S3 | Disabled | See §3.3 step 5 |

## 6. Assembly standard operating procedure (Phase 1)

Wear an ESD wrist strap (BOM-11). Keep both systems unplugged from mains until step 9.

1. Mount the chassis PSU, the PEX8747 backplane, the OCuLink receiver and the fans
   on the frame.
2. **Cold check the PSU:** jump PS_ON with the PSU tester and confirm the 12 V, 5 V
   and 3.3 V rails and PWR_OK (TC-HW-02). Disconnect the tester.
3. Seat the RTX 2060 in downstream slot 1 and the RTX 3050 in slot 2, with at least
   one empty slot between them. Secure the brackets.
4. Connect the chassis PSU to **every** backplane power input, to the receiver
   power, to the fans, and one dedicated 8-pin to each GPU (no daisy-chain pigtail
   for the 2060).
5. Install the M.2 → OCuLink adapter in the host (a CPU-attached M.2 slot).
6. Connect the OCuLink cable between the host adapter and the chassis receiver.
   Fit the strain relief.
7. Wire the sync relay: sense lead to a host SATA/Molex connector, switched output
   to the chassis PSU's 24-pin harness.
8. Plug both PSUs into the same power strip. Measure frame-to-earth continuity
   (TC-HW-11).
9. Apply mains and press the host power button. The chassis fans and both GPU fans
   should spin up within 1 s of the host (TC-HW-03).
10. Continue with the deployment guide §3 (OS) and the link gate (Phase 2).

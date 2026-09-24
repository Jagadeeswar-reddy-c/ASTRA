# ASTRA Backplane PCB — Requirements Specification

| | |
|---|---|
| Document ID | ASTRA-HW-004 |
| Version | 0.1 (2026-09-24), draft for CR-007 |
| Owner | Hardware Eng. · Reviewers: Solution Architect, QA Lead |
| Purpose | Define a PCB that mounts 4 GPUs (or feeds external bricks), connects them through a **P2P-capable PCIe Gen4 switch** for tensor-parallel speed (ADR-0016), powers them safely, and reports per-slot power and health to `astra` |

## 1. Why a custom board

| Problem today | What the board fixes |
|---|---|
| GPUs talk through the CPU's root complex; all-reduces go via host memory (45 µs) | GPU↔GPU traffic stays inside the switch (P2P, ~25 µs estimated): faster tensor split |
| No-name PEX8747 Gen3 boards, no datasheet (R-09); Gen3 x8 per GPU | Our own Gen4 x16 design with known parts |
| Loose risers and cables; ad-hoc mounting | Slots on one rigid board with standard ATX mounting |
| Power is estimated from board ratings (LIM-25) | Measured watts per slot and PSU telemetry over USB |
| No PCIe hot-plug on consumer boards (LIM-05) | Switch ports have hot-plug controllers (stretch goal) |

## 2. Block diagram

```text
 Head PC                                   ASTRA Backplane (12-14 layer, low-loss)
 ┌──────────────┐  2 x SlimSAS 8i (x16)    ┌───────────────────────────────────────────────┐
 │ x16 host     │═════════════════════════▶│ redrivers ─▶ PCIe Gen4 switch (PM40084/100)   │
 │ adapter card │  sideband: PERST#,       │              │  P2P between downstream ports │
 └──────────────┘  PS_ON, 100 MHz clock    │   ┌──────────┼──────────┬──────────┐        │
                                           │   x16       x16        x16        x16  x8/x8 │
                                           │  SLOT0     SLOT1      SLOT2      SLOT3  OCuLink│
                                           │   │eFuse     │eFuse     │eFuse     │eFuse  8i │
                                           │  INA 12V/3V3 per slot ─┐                ▶ bricks│
                                           │ ┌──────────────────────┴───────────────┐       │
                                           │ │ MCU (USB): sensors, eFuses, PS_ON,   │──USB──▶ head PC (astra)
                                           │ │ fans, LEDs, PSU PMBus, temp, EEPROM  │       │
                                           │ └──────────────────────────────────────┘       │
                                           │ 12 V in: ATX 24-pin + 2 x EPS 8-pin; 3.3 V buck│
                                           └───────────────────────────────────────────────┘
 GPU 8-pin / 12V-2x6 cables come straight from the SAME PSU (single-source rule).
```

## 3. Requirements

Verification: **T** test, **A** analysis, **I** inspection, **D** demonstration.

### PCIe fabric
| ID | Requirement | Ver. |
|---|---|---|
| HW-BP-01 | PCIe Gen4 switch with ≥ 80 lanes: x16 upstream, 4 × x16 downstream slots, ≥ 1 × x8 (2 × OCuLink 4i or 1 × 8i) downstream for external bricks | I |
| HW-BP-02 | **Peer-to-peer between downstream ports**: ACS redirect off by default in switch configuration, so GPU↔GPU traffic does not go to the root complex | T (TC-HW-14) |
| HW-BP-03 | Every GPU slot trains **Gen4 x16** with a GPU fitted and holds it under full load for 30 min with **zero** AER errors | T (TC-HW-16) |
| HW-BP-04 | Host link: 2 × SlimSAS 8i (SFF-8654) forming x16 Gen4, x8 mode for hosts with one cable; linear redrivers on the host side | T |
| HW-BP-05 | 100 MHz reference clock: local clock generator + fan-out buffer; the switch uses separate-refclk mode when the host clock is not carried by the cable | A, T |
| HW-BP-06 | PERST# from the host fanned out to every slot, held until slot power is good for ≥ 100 ms (CEM timing) | T (scope) |
| HW-BP-07 | Switch configuration in an on-board EEPROM, updatable from the MCU; hot-plug controllers wired to the slots (for a later hot-plug feature) | I, D |
| HW-BP-08 | Differential pairs 85 Ω ±10 %, length-matched, back-drilled vias; channel loss within the Gen4 budget, confirmed by simulation before fabrication | A |

### Power
| ID | Requirement | Ver. |
|---|---|---|
| HW-BP-09 | Each slot supplies CEM slot power: 12 V at 5.5 A and 3.3 V at 3 A (75 W), 3.3 Vaux 375 mA | T |
| HW-BP-10 | Slot 12 V from 2 × EPS 8-pin (4 slots × 66 W = 264 W needs ~22 A; the ATX 24-pin 12 V pins alone are not enough); 3.3 V from an on-board 12 V → 3.3 V buck ≥ 15 A | A |
| HW-BP-11 | Per-slot eFuse on 12 V and 3.3 V (current limit, fast trip, reverse protection); the MCU can switch each slot off | T |
| HW-BP-12 | Single-source rule: board and all GPU power leads from **one** PSU; the board input is keyed so a second PSU cannot feed slot power | I |
| HW-BP-13 | PS_ON follows the host (sideband or the MCU), power-good gated; the chassis never powers GPUs while the host is off | T (TC-HW-03) |
| HW-BP-14 | Earth: mounting holes bonded to ground, < 0.1 Ω to the PSU earth pin | T (TC-HW-11) |

### Telemetry and control
| ID | Requirement | Ver. |
|---|---|---|
| HW-BP-15 | Current/voltage/power sensor per slot rail (12 V, 3.3 V), ±1 % at > 10 W, sampled ≥ 10 Hz | T (TC-HW-15) |
| HW-BP-16 | Reads the PSU over PMBus when the PSU offers it (wall-side watts) | D |
| HW-BP-17 | MCU with USB (CDC serial, documented line protocol) exposing: per-slot power, eFuse state, temperatures, fan RPM, PSU data, link LEDs; `astra` reads it into the exporter | D |
| HW-BP-18 | Two 4-pin PWM fan headers with tach; fan curve from board and switch temperatures | T |
| HW-BP-19 | Per-slot LEDs: power, link up, fault; a board status LED | I |
| HW-BP-20 | Watchdog: if the MCU hangs, slot power stays in its last safe state; a missed host heartbeat is reported, not acted on | T |

### Mechanical and thermal
| ID | Requirement | Ver. |
|---|---|---|
| HW-BP-21 | Mounts on standard ATX/E-ATX standoffs; fits open-frame mining/test benches | I |
| HW-BP-22 | Slot pitch 3 slots (60.96 mm) for 2.5–3-slot GPUs; riser option for 4 thick cards | I |
| HW-BP-23 | Switch heatsink (+ fan) keeps the switch below its rated case temperature at 35 °C ambient | T |
| HW-BP-24 | Bracket rail for GPU I/O brackets and cable strain relief | I |
| HW-BP-25 | Board ID + serial in EEPROM; silkscreen slot numbers match `astra` GPU order | I |

## 4. Parts to evaluate (Phase A informs the choice)

| Function | Candidates | Note |
|---|---|---|
| Gen4 switch | Microchip Switchtec PM40084 / PM40100; Broadcom PEX88080 / PEX88096 | Switchtec has documented per-port and P2P configuration; an off-the-shelf PM40100 board exists (c-payne) for Phase A |
| Host redrivers | TI DS160PR410 (Gen4, 4 lanes) class | Only if cable + trace loss needs it (simulation) |
| Slot eFuse | TI TPS2598x (12 V), TPS25947 (3.3 V) class | Current limit set per CEM |
| Power sensor | TI INA228 / INA238 | I2C, 20/16-bit |
| MCU | RP2040 or STM32G0/G4 with USB | Open toolchain, cheap, USB CDC |
| Clock | 100 MHz generator + PCIe fan-out buffer (Renesas / Skyworks class) | Low jitter, HCSL |

## 5. Plan and verification

| Phase | Content | Exit criteria |
|---|---|---|
| A (buy) | Off-the-shelf Gen4 switch board + 2 same-generation GPUs; Linux + open kernel modules (+ P2P patch), Resizable BAR on | TC-HW-14: `nvidia-smi topo -p2p r` = OK, P2P bandwidth ≥ 20 GB/s, latency measured; tensor split measured vs pipeline; ASTRA's 25 µs P2P constant updated |
| B1 (design) | Schematic, SI/PI simulation, layout, design review against this spec | Review sign-off; simulated channel margin at Gen4 |
| B2 (prototype) | 3–5 boards, bring-up | TC-HW-14/15/16, TC-HW-03/10/11 PASS |
| B3 (release) | Rev B with fixes, BOM freeze, assembly SOP | G3 BOM freeze for the backplane |

## 6. Risks

| ID | Risk | Mitigation |
|---|---|---|
| R-16 | First Gen4 board spin fails signal integrity | Phase A first; simulation; test coupons; board house with Gen4 experience |
| R-15 | Resizable BAR (needed for P2P) makes every GPU claim VRAM-sized address space; consumer boards may not map 4 × 24 GB | Workstation/server boards for 3+ GPUs; test in Phase A; the board allows running with ReBAR off (host all-reduce) |
| R-17 | P2P on GeForce needs an unofficial driver patch (Linux only) that can crash on some platforms | Keep host all-reduce as the default; P2P is opt-in and measured (`astra auto`) |

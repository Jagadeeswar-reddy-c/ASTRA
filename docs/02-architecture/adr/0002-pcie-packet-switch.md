# ADR-0002: PCIe packet switch instead of host bifurcation

* Status: Accepted (G2)
* Deciders: Enterprise Architect, Hardware Engineer

## Context
One x4 host link must serve two or more GPUs. Bifurcation (x4 → 2 × x2) needs BIOS
support that M.2 slots almost never offer, and it halves the lanes per GPU
permanently.

## Decision
Use an active **PCIe packet switch backplane (Broadcom/PLX PEX8747, 48-lane Gen3)**
in the chassis. It has one x4 upstream and 2–4 downstream x16 slots.

## Consequences
* \+ Works on any host with an x4 M.2 slot, and each GPU is a separate endpoint to
  the kernel.
* \+ A GPU that is idle leaves the whole uplink to the busy one (dynamic sharing,
  unlike a static bifurcation split).
* − The PEX8747 is Gen3, so the uplink runs at Gen3 x4 even on a Gen4 host
  (DR-03). A Gen4 switch is a future upgrade that will need its own ADR.
* − Extra MMIO/BAR demand (DR-12) and ~10–25 W of switch power from the chassis PSU.
* − Backplane boards are often undocumented imports (risk R-09); qualify them
  before the BOM freeze.
* The link gate identifies the switch by PCI vendor `0x10b5`, which is configurable.

# ADR-0001: Native PCIe over OCuLink instead of USB4/Thunderbolt

* Status: Accepted (G2)
* Deciders: Enterprise Architect, Hardware Engineer

## Context
The GPUs sit in a separate chassis and need a cable link to the host. The options
are USB4/Thunderbolt eGPU enclosures, OCuLink (SFF-8611/8612), and SlimSAS (SFF-8654).

## Decision
Use **OCuLink 4i (SFF-8611)** carrying native PCIe x4 from an M.2 Key-M adapter with
a ReDriver. SlimSAS 4i/8i is the approved alternative if cabling availability
requires it.

## Consequences
* \+ No protocol translation: the GPU is a plain PCIe endpoint, with no tunnelling
  overhead or controller CPU load, and switch latency stays in the sub-microsecond
  range.
* \+ Cheap (≈ €40 for the adapter and cable).
* − Not hot-pluggable. Connect only when both systems are powered off (DR-06).
* − Bandwidth is capped by the x4 uplink (Gen3 x4 ≈ 3.9 GB/s with our switch,
  DR-03).
* − Signal integrity depends on cable quality and length. It is monitored on every
  boot (L05–L09).

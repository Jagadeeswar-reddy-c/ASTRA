# ADR-0006: Ubuntu 24.04 LTS as the production host OS

* Status: Accepted (G2)
* Resolves: DR-11

## Context
The concept allows "Linux (or Windows)". vLLM, the NVIDIA Container Toolkit, sysfs
topology, AER counters and systemd hardening are Linux-only. Windows WDDM exposes
none of the PCIe path data the link gate needs.

## Decision
Production ASTRA hosts run **Ubuntu 24.04 LTS** (22.04 is also supported) with the
Ubuntu-packaged NVIDIA driver (≥ 535) and Docker Engine plus the NVIDIA Container
Toolkit. The `astra` tooling stays cross-platform so engineers can plan and inspect
from Windows workstations.

## Consequences
* \+ Full validation coverage (L04, L05, L08 and L09 run instead of reporting SKIP).
* \+ Both engines and all Compose profiles are available.
* − A Windows host is limited to llama.cpp and partial validation. This is
  documented as unsupported for production.

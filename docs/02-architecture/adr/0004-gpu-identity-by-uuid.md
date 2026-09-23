# ADR-0004: Address GPUs by UUID in PCI-bus order

* Status: Accepted (G2)
* Resolves: DR-07

## Context
The concept pins containers with `--gpus device=1` / `device=2`. GPU indices depend
on whether the host has its own GPU and on CUDA's default "fastest first" ordering,
which differs from nvidia-smi's PCI order. A wrong index silently puts a workload on
the wrong card.

## Decision
* Every generated launch sets `CUDA_DEVICE_ORDER=PCI_BUS_ID` and selects devices by
  **UUID** (`CUDA_VISIBLE_DEVICES` / `NVIDIA_VISIBLE_DEVICES`), never by index.
* Pipeline stage order is PCI bus order. The planner sorts devices by bus ID, and
  inside a container CUDA enumerates the visible UUIDs in bus order.
* Operators select GPUs in the CLI by index, UUID or bus ID (`--gpus`). The
  generated artefacts always contain UUIDs.

## Consequences
* \+ Deterministic placement across reboots, driver updates and host GPU changes.
* − If a card is replaced, its UUID changes and `.env` files must be regenerated
  with `astra plan --env-file`. The runtime gate (R01) catches a stale UUID.

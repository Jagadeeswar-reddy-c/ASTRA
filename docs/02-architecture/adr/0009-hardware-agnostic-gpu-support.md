# ADR-0009: Hardware-agnostic GPU support through a capability catalog and a compatibility gate

* Status: Accepted (CR-001)
* Deciders: Enterprise Architect, Solution Architect, Hardware Engineer

## Context
The chassis may hold any NVIDIA GPUs, from a 2 GB GT 1030 (Pascal, 48 GB/s, no NVENC,
x4, slot-powered) to RTX 50-series cards. The driver does not report several facts
that planning and safety depend on: memory bandwidth, reference board power, NVENC
presence, and which driver/CUDA generations still support the architecture. Mixing
generations adds hard constraints:

| Constraint | Source |
|---|---|
| One driver branch must support every installed GPU. Kepler ends at R470; Maxwell/Pascal/Volta end at R580; Blackwell needs ≥ R570 | NVIDIA driver lifecycle notices |
| CUDA 13 no longer compiles for sm < 75, so mixes containing Maxwell/Pascal/Volta need CUDA 12.x engine builds with every GPU's `sm` in `CMAKE_CUDA_ARCHITECTURES` | CUDA 13 release notes |
| vLLM needs compute capability ≥ 7.0; llama.cpp runs from Maxwell up (ASTRA qualifies ≥ 6.1) | Engine documentation |
| PSU load scales with the installed cards | Board power figures |

## Decision
1. **A data-only capability catalog** (`hardware/gpu_specs.py`) holds reference-board
   figures for GeForce GT/GTX/RTX models from Maxwell to Blackwell, plus
   per-architecture facts (compute capability, driver window, CUDA ceiling, bf16).
   Lookup matches the normalised product name (longest model name wins) and picks
   the variant with the nearest VRAM, e.g. RTX 3060 8 GB (240 GB/s) vs 12 GB
   (360 GB/s). Unknown cards fall back to architecture facts and the driver's power
   limit, and are flagged.
2. **A compatibility analysis** (`hardware/compat.py`) turns the installed (or
   simulated) GPUs and the driver version into: the permissible driver window,
   engine eligibility per GPU, CUDA build architectures, NVENC availability, slot
   count, and a PSU budget (`Σ board power + overhead` sustained ≤ 70 % of the PSU;
   `Σ board power × 1.3 + overhead` peak ≤ the PSU). It powers `astra compat`,
   link-gate checks **L11/L12**, the exporter, and the planner.
3. **Chassis membership** comes from topology (GPUs behind the switch). Without
   topology it is the GPUs without an active display, so the host's own display GPU
   is not counted against the chassis PSU.
4. **Before buying:** `astra compat --simulate "RTX 3060, GT 1030" --driver 580.95`
   answers "will this mix work, on which driver, and with what PSU?"

## Consequences
* \+ Any mix is supported and its constraints are surfaced before power-on and on
  every boot.
* \+ The same numbers drive planning (bandwidth), safety (power) and operations
  (alerts).
* − The catalog must be maintained as NVIDIA ships cards. An unknown card still
  works but has no speed estimate. Adding a row is a one-line PR.
* − Reference figures differ from factory-overclocked boards. The transient factor
  and the 70 % sustained ceiling absorb this, and the live `astra_chassis_power_watts`
  metric measures the real draw.

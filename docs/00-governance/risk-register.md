# ASTRA — Risk Register

| | |
|---|---|
| Document ID | ASTRA-GOV-002 |
| Version | 1.0 (2026-09-23) |
| Review | Weekly; the PM owns this register |

Scoring: Likelihood (L) and Impact (I) are each rated 1–5. Score = L × I.
≥ 12 is **high**, 6–11 is medium, ≤ 5 is low.

| ID | Risk | L | I | Score | Mitigation | Owner | Status |
|---|---|---|---|---|---|---|---|
| R-01 | Marginal signal integrity on the 1 m external link (AER errors, replays, Xid 79) | 3 | 5 | **15** | Shielded cable; ReDriver adapter; link gate L05–L09 at every boot; fall back to a 0.5 m cable; `pcie_aspm=off` option | Hardware Eng. | Open |
| R-02 | BAR/MMIO allocation fails for switch + 2 GPUs behind an M.2 slot | 3 | 4 | **12** | BIOS "Above 4G Decoding" on; `pci=realloc` option; L09 detects "BAR n: no space" | Hardware Eng. | Open |
| R-03 | Host sleep/hibernate removes relay power; GPUs vanish mid-workload | 4 | 4 | **16** | Mask sleep targets (setup-host.sh step 4); TC-HW-09 | DevOps | Mitigated (design) |
| R-04 | Pre-Ampere (Turing) support narrows in newer vLLM releases | 3 | 3 | 9 | llama.cpp is the primary engine (ADR-0003); vLLM pinned to a version verified by TC-RT-02 | Solution Architect | Open |
| R-05 | Power sequencing wrong: chassis powered after host PCIe enumeration | 2 | 4 | 8 | Relay starts chassis PSU with host PS_ON; TC-HW-03 measures skew; TC-HW-10 cold-boot × 10 | Hardware Eng. | Open |
| R-06 | Ground loop or current sharing between PSUs damages a GPU | 1 | 5 | 5 | Single-source power rule; both PSUs on one outlet strip; TC-HW-11 | Hardware Eng. | Mitigated (design) |
| R-07 | Thermal throttling in the open frame at high ambient temperature | 2 | 3 | 6 | 2 × 120 mm fans; L10/R04 checks; alert `AstraGpuHot` | Hardware Eng. | Open |
| R-08 | Planner estimate differs from real engine allocation (compute buffers) | 3 | 2 | 6 | 768 MiB per-GPU reserve; exact GGUF sizing; R03 runtime check with ±15 % tolerance | Software Dev | Open |
| R-09 | PLX backplane is a no-name import with no datasheet or support | 3 | 3 | 9 | Buy from a vendor with a return policy; qualify with TC-HW-04/05 before BOM freeze; keep a spare budget line | PM | Open |
| R-10 | Pooling adds latency with no benefit for small models | 4 | 2 | 8 | Planner warns when one GPU suffices; ops guidance in the deployment guide | Solution Architect | Mitigated |
| R-11 | Driver/nvidia-smi XML schema changes break the parser | 2 | 3 | 6 | Schema-tolerant parser, tested against current and legacy fixtures; hardware integration test in CI on the node | Software Dev | Mitigated |
| R-13 | A pre-Turing card (GT 1030, GTX 9xx/10xx) locks the host to the R580 driver branch; later GPUs needing a newer branch cannot be mixed in | 3 | 3 | 9 | Check L11; `astra compat` before purchase; runbook RB-07; prefer Turing or newer when buying | Solution Architect | Open |
| R-14 | A GPU swap exceeds the chassis PSU | 3 | 5 | **15** | Check L12 at every boot; alert `AstraChassisUndersized`; deployment guide §8a | Hardware Eng. | Mitigated (design) |
| R-12 | ESD or incorrect cabling during assembly | 2 | 4 | 8 | Assembly SOP with an ESD strap; cold checks with a PSU tester before first power-on | Hardware Eng. | Open |

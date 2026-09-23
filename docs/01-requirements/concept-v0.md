# ASTRA — Concept Document v0 (as received, input baseline)

> This is the originating concept, kept unchanged in substance as the input to
> requirements and architecture. Formatting has been normalised. Issues found in it
> are recorded in [design-review.md](../02-architecture/design-review.md) and are
> **not** corrected here.

Project ASTRA (Asynchronous Scalable Topology for Resource Aggregation) establishes an
external, modular GPU compute node (JBOG — "Just a Bunch of GPUs"). It decouples
graphics/compute hardware from the host system using high-speed PCIe-over-cable
signalling and automated distributed runtimes. The plan covers the end-to-end
hardware architecture, electrical isolation, automated workload distribution
(pooling heterogeneous VRAM), bill of materials and deployment milestones.

```
HOST PC
  - Motherboard M.2 (Key-M) / PCIe x4 slot
  - Host adapter: M.2 to OCuLink 4i (SFF-8611) ReDriver
        |
  [ 1.0 m shielded OCuLink cable (PCIe 4.0 x4) ]
        |
ASTRA EXPANSION CHASSIS
  PCIe packet switch / backplane (PLX PEX8747)
    Upstream: 1x PCIe x4 (from host)
    Downstream: 2 to 4 independent physical PCIe x16 slots
  [Slot 1] NVIDIA RTX 2060 (6 GB GDDR6)     [Slot 2] NVIDIA RTX 3050 (8 GB GDDR6)
  Auxiliary power subsystem
    - Dedicated 650 W modular ATX PSU
    - Dual-PSU ATX power relay (synchronised via host sense wire)
    - Unified chassis ground plane
```

* **Physical interconnect:** native PCIe over cable via OCuLink 4i (SFF-8611) or
  SlimSAS (SFF-8654). USB4/Thunderbolt adds protocol-translation latency and high
  CPU overhead; native PCIe-over-cable provides direct memory mapping with
  sub-microsecond interconnect overhead.
* **PCIe packet switch:** a Broadcom/PLX PEX-based active backplane multiplexes the
  upstream link into independent downstream links. No BIOS bifurcation support is
  needed, and each GPU is a separate endpoint to the OS kernel.
* **Interlocked power timing:** the secondary PSU is triggered by the host's boot
  state through a dual-PSU solid-state relay board connected to a spare SATA or
  Molex line from the host. When the host turns on, 12 V trips the relay and
  closes PS_ON on the secondary PSU's 24-pin connector.
* **Single-rail power rule:** each GPU's slot power (75 W) and 8-pin auxiliary power
  (150 W) are fed strictly from the chassis PSU, to prevent circulating currents.
* **Common ground reference:** the chassis frame, the secondary PSU earth ground and
  the host ground line (via the cable shield) keep an equipotential plane to
  preserve PCIe signal integrity.

Because the cards have asymmetric architectures (Turing vs Ampere) and mismatched
VRAM (6 GB vs 8 GB), hardware SLI is bypassed in favour of dynamic runtime
orchestration. For LLMs, vision-language models and multimodal pipelines, memory is
pooled via sequential pipeline parallelism:

* Engine: vLLM or llama.cpp.
* Memory split: `VRAM_GPU0 / VRAM_total : VRAM_GPU1 / VRAM_total = 6/14 : 8/14 ≈ 43 % : 57 %`.
  The runtime maps early transformer blocks to the RTX 2060 and the rest to the RTX
  3050, shuttling activations across the PCIe bus.
* Ray orchestrator: a local Ray cluster exposes both GPUs as an elastic pool
  (`@ray.remote(num_gpus=1)`).
* Heterogeneous containerisation: NVIDIA Container Runtime exposes discrete GPU IDs —
  Container A (`--gpus device=1`) inference server; Container B (`--gpus device=2`)
  video transcoding.

## Bill of materials (concept)

| Category | Component | Est. cost | Purpose |
|---|---|---|---|
| Host interface | M.2 Key-M to OCuLink 4i adapter (ReDriver) | €25–35 | Break out host M.2 lanes |
| Interconnect | Shielded OCuLink SFF-8611 cable, 1.0 m, PCIe 4.0 | €15–20 | Transmission line |
| Chassis switch | Dual/Quad PCIe x16 backplane with PLX PEX switch | €90–140 | Multiplex lanes |
| Power supply | 650 W 80+ Gold fully modular ATX | €65–85 | Backplane and GPU rails |
| Power sync | Add2PSU solid-state relay | €8–12 | Auto-power with host |
| Enclosure | Open-frame aluminium test bench | €30–45 | Mounting, ventilation |
| Active cooling | 2 × 120 mm PWM fans 1800 rpm | €15–20 | Forced air |
| **Total** | | **€248–357** | Excluding GPUs |

## Roadmap (concept)

1. **Physical assembly and cold checks (days 1–2):** mount PSU, backplane and fans;
   seat the GPUs; wire 8-pin power; install the M.2 adapter; wire the sync relay.
   Verify that the chassis PSU and GPUs power up with the host.
2. **Bus link negotiation (day 3):** boot Linux (or Windows) with the cable seated;
   `lspci -tv | grep -i nvidia`; `nvidia-smi --query-gpu=index,name,pci.bus_id,pcie.link.gen.current,pcie.link.width.current --format=csv`.
   Success: both GPUs negotiate links with no bus-reset errors in `dmesg`.
3. **Runtime abstraction and split validation (days 4–5):**
   `python3 -m vllm.entrypoints.openai.api_server --model meta-llama/Llama-3.1-8B-Instruct --tensor-parallel-size 2 --gpu-memory-utilization 0.90`;
   `watch -n 0.5 nvidia-smi`. Success: proportional allocation (~5.4 GB on GPU 0,
   ~7.2 GB on GPU 1) under continuous load, without thermal throttling or kernel
   panics.

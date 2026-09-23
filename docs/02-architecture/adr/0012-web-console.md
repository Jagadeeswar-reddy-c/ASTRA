# ADR-0012: Web console (`astra ui`): exo-style topology, planning, health and chat

* Status: Accepted (CR-004)
* Deciders: Solution Architect, Enterprise Architect

## Context
The CLI shows the pool as tables. The sponsor asked for a frontend like the exo
dashboard, because seeing which machines and GPUs exist and where each layer runs
makes the system understandable at a glance. The console must run on a GPU-less
mini PC, work offline, and add no supply-chain risk.

## Decision
* **`astra ui`**: a stdlib HTTP server (same pattern as the exporter and agent) with
  a small JSON API: `/api/state`, `/api/models`, `/api/plan`, `/api/chat`. The page
  is a single self-contained `index.html` (inline CSS/JS, inline SVG, **no CDN, no
  build toolchain, no npm dependencies**) shipped as package data.
* **Views:**
  * **Topology:** machines on a ring (head at the top), GPUs inside each machine,
    network links between machines. The active plan is drawn as numbered pipeline
    stages with colour-coded layer ranges and arrows, and remote→remote hops are
    routed through the head node, because llama.cpp RPC relays them there.
  * **GPUs:** live VRAM, temperature, utilisation, power and bandwidth.
  * **Plan:** model/engine/quantization/context/objective → fit, estimated speed,
    maximum context, layer strip, stages, alternatives and the launch command.
  * **Health:** driver window, CUDA archs, chassis power meter, compatibility findings.
  * **Chat:** streams from the engine through a proxy on the console.
* **Sources:** the same `astra.pool.collect()` the CLI uses (local, fabric agents,
  or `--simulate`), so the console and CLI can never disagree.
* **Security:** binds to `127.0.0.1` by default. The chat proxy targets only the
  engine URL given at start-up (no open proxy / SSRF). Bodies are capped at 1 MiB.
  Remote access goes through an authenticating reverse proxy or an SSH tunnel.

## Consequences
* \+ One page explains the whole system: what is in the pool, what the planner
  chose and why, and whether it is healthy.
* \+ Simulation mode doubles as a sales and planning tool before hardware is bought.
* − Hand-written JS/SVG must be maintained without a framework; it is kept to one
  file with API-level tests (`tests/unit/test_ui.py`) and screenshot checks in
  light/dark themes and at narrow widths.
* − Polling (2.5 s) re-queries fabric agents. Acceptable for a handful of nodes;
  larger clusters would need push updates.

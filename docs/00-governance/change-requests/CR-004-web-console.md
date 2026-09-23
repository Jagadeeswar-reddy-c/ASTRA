# CR-004 — Web console like the exo dashboard

| | |
|---|---|
| Raised | 2026-09-23 by the Sponsor ("build a frontend like exo labs; it will help understand better") |
| Status | **Approved and implemented** in v0.4.0 (ADR-0012) |
| Affects | PRD (FR-19, NFR-13), architecture (ADR-0012), new package `astra.ui`, `astra-ui.service`, deployment guide §11 |

## Impact analysis

| Area | Change |
|---|---|
| Requirements | FR-19 (web console: topology, GPUs, plan, health, chat); NFR-13 (console security: localhost bind, no open proxy, no external assets) |
| Software | `astra/pool.py` (shared pool collection for CLI and console), `astra/ui/server.py`, `astra/ui/static/index.html`; fabric descriptors now carry live temperature, utilisation and power |
| Planner | Fixed during review: the head↔first/last remote stage and remote→remote relays (via the head) now count as network hops in the speed estimate |
| Tests | `tests/unit/test_ui.py` (API contract, chat streaming through a fake engine, validation); visual checks in light/dark and at 400 px |
| Operations | `astra-ui.service` (localhost only); access via SSH tunnel or reverse proxy |

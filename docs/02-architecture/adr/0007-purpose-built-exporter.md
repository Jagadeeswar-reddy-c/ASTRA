# ADR-0007: Purpose-built Prometheus exporter instead of DCGM

* Status: Accepted (G2)

## Context
NVIDIA DCGM-exporter targets data-centre GPUs. GeForce support is partial, and it
knows nothing about the ASTRA-specific failure modes: the uplink behind a switch,
AER counters on every hop, and missing GPUs relative to the as-built configuration.

## Decision
Ship `astra exporter`, a stdlib HTTP server exposing `/metrics` (Prometheus text
format 0.0.4) and `/healthz`. It reuses the same probe as the link gate, so a
metric and a validation result never disagree. Metric names use the `astra_`
prefix and are catalogued in `docs/07-operations/monitoring.md`.

## Consequences
* \+ One source of truth for health; the alert rules map directly to runbooks.
* \+ Scrapes never overload the driver: samples are cached and rate-limited, and a
  failed sample sets `astra_up 0` instead of crashing the exporter.
* − We maintain the metric set ourselves. Engine-level metrics (tokens/s, queue
  depth) come from llama-server `--metrics` and vLLM `/metrics`, scraped directly.

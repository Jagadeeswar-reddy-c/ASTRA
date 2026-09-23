# ADR-0008: Docker Compose profiles for workloads, systemd for node services

* Status: Accepted (G2)

## Context
The node runs two kinds of software: node-level services that must work before any
workload (self-test, exporter), and workloads that change often (inference engines,
transcoding). The concept document calls for container isolation per GPU.

## Decision
* **Workloads run as Docker Compose profiles** (`pooled`, `vllm`, `partitioned`) in
  `deploy/compose/compose.yaml`. GPU assignment comes from the `.env` file written
  by `astra plan --env-file` (UUIDs, layer split, context).
* **Node services run under systemd**: `astra-selftest` (oneshot, boot) and
  `astra-exporter`. `astra-inference` is a bare-metal alternative to Compose and is
  hard-gated on `astra-selftest` (`Requires=`/`After=`).
* Monitoring (Prometheus, Grafana) runs in the same Compose project, with ports
  bound to localhost.

## Consequences
* \+ Workloads upgrade by changing an image tag; the node services keep working even
  when Docker is broken, which is exactly when they are needed.
* \+ The Compose file is validated in CI (`docker compose config`).
* − Two deployment mechanisms to learn. The deployment guide gives one procedure
  for each.

# ASTRA — Monitoring

## 1. Service level objectives

| SLO | Target | Measured by |
|---|---|---|
| Node availability (all expected GPUs visible, exporter up) | 99.5 % monthly | `astra_up`, `astra_gpu_count >= astra_gpu_expected_count` |
| Link health | 0 uncorrectable AER, 0 replay increase per week | `astra_path_aer_errors_total`, `astra_gpu_pcie_replay_total` |
| Thermal headroom | < 1 h/month above 83 °C | `astra_gpu_temperature_celsius` |

## 2. Metric catalog (`astra exporter`, port 9835)

| Metric | Type | Labels | Meaning |
|---|---|---|---|
| `astra_up` | gauge | node | 1 if the last hardware sample succeeded |
| `astra_build_info` | gauge | node, version, driver, cuda | Build and driver info (value 1) |
| `astra_scrape_errors_total` | counter | node | Failed samples since start |
| `astra_gpu_count` / `astra_gpu_expected_count` | gauge | node | Visible GPUs / GPUs listed in the config |
| `astra_gpu_info` | gauge | node, gpu, uuid, name, bus_id, architecture, compute_capability, in_chassis, vbios | Identity (value 1) |
| `astra_chassis_psu_watts` / `_power_limit_watts` | gauge | node | Configured PSU rating / sustained ceiling (rating × max ratio) |
| `astra_chassis_power_watts` | gauge | node | Measured draw of the chassis GPUs + overhead |
| `astra_chassis_rated_power_watts` | gauge | node | Board power of the installed chassis GPUs + overhead |
| `astra_gpu_memory_total_bytes` / `_used_bytes` | gauge | node, gpu, uuid | Framebuffer |
| `astra_gpu_utilization_ratio` | gauge | node, gpu, uuid | 0–1 |
| `astra_gpu_temperature_celsius` | gauge | node, gpu, uuid | Core temperature |
| `astra_gpu_slowdown_temperature_celsius` | gauge | node, gpu, uuid | Driver slowdown threshold |
| `astra_gpu_power_watts` / `_power_limit_watts` | gauge | node, gpu, uuid | Board power |
| `astra_gpu_fan_ratio` | gauge | node, gpu, uuid | 0–1 |
| `astra_gpu_clock_event_active` | gauge | node, gpu, uuid, reason | 1 per active clock-event reason |
| `astra_gpu_pcie_link_gen` / `_gen_max` / `_link_width` | gauge | node, gpu, uuid | GPU↔switch link |
| `astra_gpu_pcie_replay_total` | counter | node, gpu, uuid | PCIe replays (signal-integrity indicator) |
| `astra_path_bottleneck_link_gen` / `_width` | gauge | node, gpu, uuid, hop | Slowest hop on the path (normally the OCuLink uplink) |
| `astra_path_bottleneck_bandwidth_bytes` | gauge | node, gpu, uuid, hop | One-way bandwidth of that hop |
| `astra_path_aer_errors_total` | counter | node, gpu, uuid, severity | AER errors summed over every hop |

Engine metrics come from llama-server (`--metrics`, :8080/metrics) and vLLM (:8000/metrics).

## 3. Alerts

The rules are in `deploy/prometheus/astra-alerts.yml` and each links to a runbook:

| Alert | Severity | Runbook |
|---|---|---|
| AstraExporterDown | critical | RB-01 |
| AstraGpuMissing | critical | RB-01 |
| AstraUplinkDegraded | warning | RB-02 |
| AstraPcieReplays | warning | RB-03 |
| AstraAerUncorrectable | critical | RB-03 |
| AstraGpuHot | warning | RB-04 |
| AstraGpuFaultSlowdown | critical | RB-04 |
| AstraChassisPowerHigh | warning | RB-05 |
| AstraChassisUndersized | warning | RB-05, RB-07 |

## 4. Dashboards

Grafana → *ASTRA → ASTRA node* shows: up, GPUs visible/expected, uplink
lanes/gen/bandwidth, chassis power gauge, VRAM, utilisation, temperature, power,
replay and AER increases, and active clock events.

# ADR-0005: Stdlib-only Python control plane using nvidia-smi XML and sysfs

* Status: Accepted (G2)

## Context
The control plane runs on bare hosts, inside minimal containers and in systemd
units at boot, possibly before networking or package mirrors are available. Data
sources: NVML (via pynvml), the nvidia-smi CLI, sysfs, and the kernel log.

## Decision
* Python ≥ 3.11, **no third-party runtime dependencies** (`tomllib`, `xml.etree`,
  `http.server` and `argparse` cover the needs). Ray is an optional extra.
* GPU data comes from **`nvidia-smi -q -x`**, which ships with every driver on Linux
  and Windows and includes the PCIe replay counter and clock-event reasons. The
  parser tolerates schema drift between driver branches (`clocks_throttle_reasons`
  vs `clocks_event_reasons`, `power_readings` vs `gpu_power_readings`).
* PCIe topology and AER counters come from **sysfs**. Fault signatures come from
  `journalctl -k`, falling back to `dmesg`.
* All I/O goes through injectable ports (`CommandRunner`, `SysfsReader`), so the
  logic is tested on any OS against recorded fixtures.

## Consequences
* \+ Installs anywhere with `pip install .`, and the container image is small.
* \+ Tested against the current (R610) and legacy (R535) XML schemas.
* − Spawning nvidia-smi costs ~100–300 ms. The exporter rate-limits samples
  (`min_sample_interval_s`, default 2 s).
* − Topology and AER data is Linux-only. On Windows those checks report SKIP rather
  than PASS.

"""Prometheus exporter (ADR-0007).

A dependency-free HTTP endpoint serving ``/metrics`` in the Prometheus text
format and ``/healthz``. DCGM-exporter targets data-centre GPUs; GeForce cards
and the PCIe-path metrics ASTRA cares about (uplink width, AER over every hop)
need this small purpose-built exporter instead.

nvidia-smi is not called more often than ``min_sample_interval_s`` no matter how
many scrapers hit the endpoint.
"""

from __future__ import annotations

import logging
import threading
import time
from collections.abc import Callable, Iterable
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from astra import __version__
from astra.config import AstraConfig
from astra.hardware.compat import analyse
from astra.hardware.models import Inventory

log = logging.getLogger("astra.exporter")

CONTENT_TYPE = "text/plain; version=0.0.4; charset=utf-8"


def _escape(value: str) -> str:
    return value.replace("\\", "\\\\").replace("\n", "\\n").replace('"', '\\"')


def _labels(**labels: str) -> str:
    inner = ",".join(f'{k}="{_escape(v)}"' for k, v in labels.items())
    return "{" + inner + "}" if inner else ""


class _Family:
    def __init__(self, name: str, help_text: str, kind: str = "gauge") -> None:
        self.name, self.help, self.kind = name, help_text, kind
        self.samples: list[tuple[str, float]] = []

    def add(self, value: float | int | None, **labels: str) -> None:
        if value is not None:
            self.samples.append((_labels(**labels), float(value)))

    def render(self) -> Iterable[str]:
        if not self.samples:
            return
        yield f"# HELP {self.name} {self.help}"
        yield f"# TYPE {self.name} {self.kind}"
        for labels, value in self.samples:
            # Integral values (bytes, counters) must not lose digits to exponent notation.
            text = str(int(value)) if value.is_integer() else repr(value)
            yield f"{self.name}{labels} {text}"


def render_metrics(
    inventory: Inventory | None,
    config: AstraConfig,
    scrape_errors: int = 0,
    last_error: str | None = None,
) -> str:
    node = config.node.name
    fam = {
        "up": _Family("astra_up", "1 if the last hardware sample succeeded"),
        "info": _Family("astra_build_info", "Exporter build and driver versions"),
        "count": _Family("astra_gpu_count", "GPUs visible to the NVIDIA driver"),
        "expected": _Family("astra_gpu_expected_count", "GPUs listed in the node configuration"),
        "errors": _Family("astra_scrape_errors_total", "Failed hardware samples", "counter"),
        "gpu_info": _Family("astra_gpu_info", "Static GPU identity"),
        "mem_total": _Family("astra_gpu_memory_total_bytes", "Framebuffer size"),
        "mem_used": _Family("astra_gpu_memory_used_bytes", "Framebuffer in use"),
        "util": _Family("astra_gpu_utilization_ratio", "GPU core utilisation (0-1)"),
        "temp": _Family("astra_gpu_temperature_celsius", "GPU core temperature"),
        "temp_slow": _Family(
            "astra_gpu_slowdown_temperature_celsius", "Driver thermal slowdown point"
        ),
        "power": _Family("astra_gpu_power_watts", "Board power draw"),
        "power_limit": _Family("astra_gpu_power_limit_watts", "Enforced power limit"),
        "fan": _Family("astra_gpu_fan_ratio", "Fan duty (0-1)"),
        "slowdown": _Family(
            "astra_gpu_clock_event_active", "1 while a clock-event reason is active"
        ),
        "gen": _Family("astra_gpu_pcie_link_gen", "GPU-side PCIe generation (current)"),
        "gen_max": _Family("astra_gpu_pcie_link_gen_max", "GPU-side PCIe generation (max)"),
        "width": _Family("astra_gpu_pcie_link_width", "GPU-side PCIe lanes (current)"),
        "replay": _Family("astra_gpu_pcie_replay_total", "PCIe replay counter", "counter"),
        "path_gen": _Family("astra_path_bottleneck_link_gen", "PCIe generation of the slowest hop"),
        "path_width": _Family("astra_path_bottleneck_link_width", "Lanes of the slowest hop"),
        "path_bw": _Family(
            "astra_path_bottleneck_bandwidth_bytes", "One-way bandwidth of the slowest hop"
        ),
        "aer": _Family(
            "astra_path_aer_errors_total", "AER errors summed along the GPU path", "counter"
        ),
        "psu": _Family("astra_chassis_psu_watts", "Configured chassis PSU rating"),
        "psu_limit": _Family(
            "astra_chassis_power_limit_watts", "Sustained power ceiling (PSU x max ratio)"
        ),
        "chassis_draw": _Family(
            "astra_chassis_power_watts", "Measured chassis GPU draw plus configured overhead"
        ),
        "chassis_rated": _Family(
            "astra_chassis_rated_power_watts",
            "Board power of the installed chassis GPUs + overhead",
        ),
        "module_psu": _Family("astra_module_psu_watts", "PSU rating of an ASTRA Stack brick"),
        "module_limit": _Family(
            "astra_module_power_limit_watts", "Brick sustained power ceiling (PSU x max ratio)"
        ),
        "module_draw": _Family(
            "astra_module_power_watts", "Measured draw of a brick's GPUs plus its overhead"
        ),
        "module_rated": _Family(
            "astra_module_rated_power_watts", "Board power of a brick's GPUs plus its overhead"
        ),
        "module_missing": _Family(
            "astra_module_missing_gpus", "Configured GPUs of a brick that are not enumerated"
        ),
    }

    fam["info"].add(
        1,
        node=node,
        version=__version__,
        driver=(inventory.driver_version or "") if inventory else "",
        cuda=(inventory.cuda_version or "") if inventory else "",
    )
    fam["up"].add(1 if inventory is not None and last_error is None else 0, node=node)
    fam["errors"].add(scrape_errors, node=node)
    fam["expected"].add(len(config.node.expected_gpus), node=node)

    fam["psu"].add(config.chassis.psu_watts, node=node)
    fam["psu_limit"].add(
        round(config.chassis.psu_watts * config.chassis.max_sustained_ratio, 3), node=node
    )

    if inventory is not None:
        fam["count"].add(len(inventory.gpus), node=node)
        report = analyse(
            inventory.gpus,
            inventory.driver_version,
            config.chassis,
            inventory.paths,
            config.interconnect.switch_vendor_ids,
            config.modules,
        )
        in_chassis = {c.gpu.uuid for c in report.chassis}
        for m in report.modules:
            ml = {"node": node, "module": m.name}
            fam["module_psu"].add(m.power.psu_watts, **ml)
            fam["module_limit"].add(round(m.power.sustained_limit_watts, 3), **ml)
            m_draw = sum(c.gpu.power_draw_w or 0.0 for c in m.members)
            fam["module_draw"].add(m_draw + m.power.overhead_watts, **ml)
            fam["module_rated"].add(m.power.sustained_watts, **ml)
            fam["module_missing"].add(len(m.missing), **ml)
        draw = sum(c.gpu.power_draw_w or 0.0 for c in report.chassis)
        fam["chassis_draw"].add(draw + config.chassis.overhead_watts, node=node)
        fam["chassis_rated"].add(report.power.sustained_watts, node=node)
        caps = {c.gpu.uuid: c for c in report.caps}
        for g in inventory.gpus:
            lbl = {"node": node, "gpu": str(g.index), "uuid": g.uuid}
            cc = caps[g.uuid].compute_capability
            fam["gpu_info"].add(
                1,
                **lbl,
                name=g.name,
                bus_id=g.bus_id,
                architecture=caps[g.uuid].architecture,
                compute_capability="" if cc is None else f"{cc:.1f}",
                in_chassis="true" if g.uuid in in_chassis else "false",
                module=report.module_of(g.uuid) or "",
                vbios=g.vbios or "",
            )
            fam["mem_total"].add(g.memory_total_bytes, **lbl)
            fam["mem_used"].add(g.memory_used_bytes, **lbl)
            fam["util"].add(None if g.utilization_pct is None else g.utilization_pct / 100, **lbl)
            fam["temp"].add(g.temperature_c, **lbl)
            fam["temp_slow"].add(g.temperature_slowdown_c, **lbl)
            fam["power"].add(g.power_draw_w, **lbl)
            fam["power_limit"].add(g.power_limit_w, **lbl)
            fam["fan"].add(None if g.fan_pct is None else g.fan_pct / 100, **lbl)
            for reason in g.active_clock_events:
                fam["slowdown"].add(1, **lbl, reason=reason)
            fam["gen"].add(g.link_gen_current, **lbl)
            fam["gen_max"].add(g.link_gen_max, **lbl)
            fam["width"].add(g.link_width_current, **lbl)
            fam["replay"].add(g.replay_counter, **lbl)

            path = inventory.paths.get(g.uuid)
            if path is not None:
                hop = path.bottleneck
                if hop is not None:
                    fam["path_gen"].add(hop.current_gen, **lbl, hop=hop.bdf)
                    fam["path_width"].add(hop.current_width, **lbl, hop=hop.bdf)
                    bw = hop.bandwidth_gbps
                    fam["path_bw"].add(None if bw is None else bw * 1e9, **lbl, hop=hop.bdf)
                fam["aer"].add(path.aer.correctable, **lbl, severity="correctable")
                fam["aer"].add(path.aer.nonfatal, **lbl, severity="nonfatal")
                fam["aer"].add(path.aer.fatal, **lbl, severity="fatal")

    lines: list[str] = []
    for family in fam.values():
        lines.extend(family.render())
    return "\n".join(lines) + "\n"


class Collector:
    """Rate-limited, thread-safe cache around a probe function."""

    def __init__(
        self,
        probe: Callable[[], Inventory],
        config: AstraConfig,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._probe, self._config, self._clock = probe, config, clock
        self._lock = threading.Lock()
        self._inventory: Inventory | None = None
        self._sampled_at = float("-inf")
        self.errors = 0
        self.last_error: str | None = None

    def sample(self) -> Inventory | None:
        with self._lock:
            now = self._clock()
            if now - self._sampled_at >= self._config.telemetry.min_sample_interval_s:
                self._sampled_at = now
                try:
                    self._inventory = self._probe()
                    self.last_error = None
                except Exception as exc:  # keep serving; surface via astra_up/astra_scrape_errors
                    self.errors += 1
                    self.last_error = str(exc)
                    self._inventory = None
                    log.warning("hardware sample failed: %s", exc)
            return self._inventory

    def metrics(self) -> str:
        inventory = self.sample()
        return render_metrics(inventory, self._config, self.errors, self.last_error)


def make_handler(collector: Collector) -> type[BaseHTTPRequestHandler]:
    class Handler(BaseHTTPRequestHandler):
        server_version = f"astra-exporter/{__version__}"

        def do_GET(self) -> None:
            if self.path.split("?")[0] == "/metrics":
                body, status, ctype = collector.metrics().encode(), 200, CONTENT_TYPE
            elif self.path == "/healthz":
                ok = collector.sample() is not None
                body = b"ok\n" if ok else f"degraded: {collector.last_error}\n".encode()
                status, ctype = (200 if ok else 503), "text/plain; charset=utf-8"
            else:
                body, status, ctype = b"see /metrics\n", 404, "text/plain; charset=utf-8"
            self.send_response(status)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, fmt: str, *args: object) -> None:
            log.debug("%s " + fmt, self.address_string(), *args)

    return Handler


def serve(collector: Collector, host: str, port: int) -> None:
    server = ThreadingHTTPServer((host, port), make_handler(collector))
    log.info("astra exporter listening on http://%s:%d/metrics", host, port)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()

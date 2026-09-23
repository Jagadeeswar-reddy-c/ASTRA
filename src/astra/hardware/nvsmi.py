"""NVIDIA driver inventory via ``nvidia-smi -q -x``.

The XML report is used instead of NVML bindings (ADR-0005): it ships with every
driver on Linux and Windows and carries fields the CSV query interface lacks
(PCIe replay counter, clock-event reasons). The schema drifts between driver
branches, so every lookup below tolerates the older and newer tag names.
"""

from __future__ import annotations

import xml.etree.ElementTree as ET
from dataclasses import dataclass

from astra.errors import ParseError
from astra.hardware.models import GpuInfo
from astra.hardware.runner import CommandRunner
from astra.units import is_missing, parse_bytes, parse_int, parse_link_width, parse_number

NVIDIA_SMI = "nvidia-smi"


@dataclass(frozen=True)
class DriverReport:
    driver_version: str | None
    cuda_version: str | None
    gpus: tuple[GpuInfo, ...]


def _text(node: ET.Element | None, *paths: str) -> str | None:
    """First non-missing text among several candidate paths (schema-drift tolerant)."""
    if node is None:
        return None
    for path in paths:
        found = node.find(path)
        if found is not None and found.text is not None and not is_missing(found.text):
            return found.text.strip()
    return None


def _clock_events(gpu: ET.Element) -> tuple[str, ...]:
    block = gpu.find("clocks_event_reasons")
    prefix = "clocks_event_reason_"
    if block is None:
        block = gpu.find("clocks_throttle_reasons")  # drivers < 535
        prefix = "clocks_throttle_reason_"
    if block is None:
        return ()
    active = []
    for child in block:
        if (child.text or "").strip() == "Active" and child.tag.startswith(prefix):
            active.append(child.tag[len(prefix) :])
    return tuple(sorted(active))


def _power(gpu: ET.Element) -> tuple[float | None, float | None, float | None]:
    block = gpu.find("gpu_power_readings")
    if block is None:
        block = gpu.find("power_readings")
    draw = parse_number(_text(block, "instant_power_draw", "power_draw", "average_power_draw"))
    limit = parse_number(_text(block, "current_power_limit", "power_limit", "enforced_power_limit"))
    max_limit = parse_number(_text(block, "max_power_limit"))
    return draw, limit, max_limit


def _display_active(gpu: ET.Element) -> bool | None:
    text = _text(gpu, "display_active")
    return None if text is None else text.lower() == "enabled"


def parse_xml(xml_text: str) -> DriverReport:
    try:
        root = ET.fromstring(xml_text)
    except ET.ParseError as exc:
        raise ParseError(f"nvidia-smi XML is not well formed: {exc}") from exc
    if root.tag != "nvidia_smi_log":
        raise ParseError(f"unexpected root element <{root.tag}>")

    gpus: list[GpuInfo] = []
    for index, gpu in enumerate(root.findall("gpu")):
        pci = gpu.find("pci")
        link = pci.find("pci_gpu_link_info") if pci is not None else None
        bus_id = _text(pci, "pci_bus_id") or gpu.get("id") or ""
        uuid = _text(gpu, "uuid")
        if not uuid:
            raise ParseError(f"GPU at {bus_id or index} has no UUID")
        draw, limit, max_limit = _power(gpu)
        gpus.append(
            GpuInfo(
                index=index,
                uuid=uuid,
                name=_text(gpu, "product_name") or "unknown",
                bus_id=bus_id,
                architecture=_text(gpu, "product_architecture"),
                vbios=_text(gpu, "vbios_version"),
                link_gen_current=parse_int(_text(link, "pcie_gen/current_link_gen")),
                link_gen_max=parse_int(_text(link, "pcie_gen/max_link_gen")),
                link_width_current=parse_link_width(_text(link, "link_widths/current_link_width")),
                link_width_max=parse_link_width(_text(link, "link_widths/max_link_width")),
                memory_total_bytes=parse_bytes(_text(gpu, "fb_memory_usage/total")),
                memory_used_bytes=parse_bytes(_text(gpu, "fb_memory_usage/used")),
                memory_free_bytes=parse_bytes(_text(gpu, "fb_memory_usage/free")),
                utilization_pct=parse_number(_text(gpu, "utilization/gpu_util")),
                temperature_c=parse_number(_text(gpu, "temperature/gpu_temp")),
                temperature_slowdown_c=parse_number(
                    _text(gpu, "temperature/gpu_temp_slow_threshold")
                ),
                power_draw_w=draw,
                power_limit_w=limit,
                fan_pct=parse_number(_text(gpu, "fan_speed")),
                performance_state=_text(gpu, "performance_state"),
                replay_counter=parse_int(_text(pci, "replay_counter")),
                active_clock_events=_clock_events(gpu),
                power_max_limit_w=max_limit,
                display_active=_display_active(gpu),
            )
        )

    return DriverReport(
        driver_version=_text(root, "driver_version", "kmd_version"),
        cuda_version=_text(root, "cuda_version", "cuda_umd_version"),
        gpus=tuple(gpus),
    )


def query(runner: CommandRunner) -> DriverReport:
    return parse_xml(runner.run([NVIDIA_SMI, "-q", "-x"]))

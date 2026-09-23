"""Scan the kernel log for PCIe / NVIDIA fault signatures.

These are the messages that show up when an external PCIe link is marginal
(bad cable, poor grounding, missing ReDriver) or mis-configured (BAR space).
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import StrEnum

from astra.errors import CommandError
from astra.hardware.runner import CommandRunner


class Severity(StrEnum):
    WARN = "warn"
    FAIL = "fail"


@dataclass(frozen=True)
class Signature:
    key: str
    pattern: re.Pattern[str]
    severity: Severity
    hint: str


SIGNATURES: tuple[Signature, ...] = (
    Signature(
        "gpu_fell_off_bus",
        re.compile(r"fallen off the bus", re.I),
        Severity.FAIL,
        "Link dropped: reseat OCuLink cable, check chassis PSU stayed on, check grounding.",
    ),
    Signature(
        "nvidia_xid",
        re.compile(r"NVRM: Xid", re.I),
        Severity.FAIL,
        "Driver-reported GPU fault; look up the Xid code (79 = fell off bus, 61/62 = firmware).",
    ),
    Signature(
        "aer_uncorrected",
        re.compile(r"AER:.*(Uncorrected|Fatal)|PCIe Bus Error: severity=Uncorrected", re.I),
        Severity.FAIL,
        "Uncorrectable PCIe error: signal integrity problem on the external link.",
    ),
    Signature(
        "aer_corrected",
        re.compile(r"AER:.*Corrected|PCIe Bus Error: severity=Corrected", re.I),
        Severity.WARN,
        "Corrected PCIe errors: link is marginal; try a shorter/better cable or force Gen3.",
    ),
    Signature(
        "bar_assign_failed",
        re.compile(r"BAR \d+: (no space|failed to assign)", re.I),
        Severity.FAIL,
        "BAR allocation failed: enable 'Above 4G Decoding' in BIOS; try kernel arg pci=realloc.",
    ),
    Signature(
        "link_down",
        re.compile(r"pcieport .*(Link Down|link is down)", re.I),
        Severity.FAIL,
        "A PCIe port reported link down.",
    ),
)


@dataclass(frozen=True)
class Finding:
    key: str
    severity: Severity
    line: str
    hint: str


def scan(text: str) -> list[Finding]:
    findings: list[Finding] = []
    for line in text.splitlines():
        for sig in SIGNATURES:
            if sig.pattern.search(line):
                findings.append(Finding(sig.key, sig.severity, line.strip(), sig.hint))
                break  # first (most severe) signature wins per line
    return findings


def read_kernel_log(runner: CommandRunner) -> str:
    """Current-boot kernel log; tries journalctl first (no root needed on most distros)."""
    errors: list[str] = []
    for argv in (["journalctl", "-k", "-b", "--no-pager", "-q"], ["dmesg"]):
        if not runner.available(argv[0]):
            continue
        try:
            return runner.run(argv, timeout=20)
        except CommandError as exc:
            errors.append(str(exc))
    raise CommandError("kernel log unavailable: " + ("; ".join(errors) or "no journalctl/dmesg"))

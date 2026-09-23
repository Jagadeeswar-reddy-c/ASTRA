"""Check results and report rendering (table, JSON, Markdown, JUnit XML)."""

from __future__ import annotations

import json
import time
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any


class Status(StrEnum):
    PASS = "PASS"
    WARN = "WARN"
    FAIL = "FAIL"
    SKIP = "SKIP"


@dataclass(frozen=True)
class CheckResult:
    id: str
    title: str
    status: Status
    detail: str
    requirements: tuple[str, ...] = ()


@dataclass
class Report:
    node: str
    phase: str
    results: list[CheckResult] = field(default_factory=list)
    started: float = field(default_factory=time.time)
    finished: float | None = None

    @property
    def status(self) -> Status:
        statuses = {r.status for r in self.results}
        if Status.FAIL in statuses:
            return Status.FAIL
        if Status.WARN in statuses:
            return Status.WARN
        return Status.PASS

    @property
    def exit_code(self) -> int:
        return 1 if self.status is Status.FAIL else 0

    def counts(self) -> dict[str, int]:
        return {s.value: sum(1 for r in self.results if r.status is s) for s in Status}

    def to_dict(self) -> dict[str, Any]:
        return {
            "node": self.node,
            "phase": self.phase,
            "status": self.status.value,
            "counts": self.counts(),
            "started": self.started,
            "finished": self.finished,
            "results": [
                {
                    "id": r.id,
                    "title": r.title,
                    "status": r.status.value,
                    "detail": r.detail,
                    "requirements": list(r.requirements),
                }
                for r in self.results
            ],
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2)

    def to_table(self) -> str:
        width = max((len(r.title) for r in self.results), default=10)
        lines = [f"ASTRA validation — node {self.node}, phase {self.phase}", ""]
        for r in self.results:
            lines.append(f"  [{r.status.value:<4}] {r.id:<5} {r.title:<{width}}  {r.detail}")
        c = self.counts()
        lines += [
            "",
            f"  Result: {self.status.value}  "
            f"(pass {c['PASS']}, warn {c['WARN']}, fail {c['FAIL']}, skip {c['SKIP']})",
        ]
        return "\n".join(lines)

    def to_markdown(self) -> str:
        c = self.counts()
        out = [
            f"# ASTRA validation report — `{self.node}` / {self.phase}",
            "",
            f"**Overall: {self.status.value}** — pass {c['PASS']}, warn {c['WARN']}, "
            f"fail {c['FAIL']}, skip {c['SKIP']}",
            "",
            "| ID | Check | Status | Detail | Requirements |",
            "|---|---|---|---|---|",
        ]
        for r in self.results:
            detail = r.detail.replace("|", "\\|")
            out.append(
                f"| {r.id} | {r.title} | {r.status.value} | {detail} "
                f"| {', '.join(r.requirements)} |"
            )
        return "\n".join(out) + "\n"

    def to_junit(self) -> str:
        suite = ET.Element(
            "testsuite",
            name=f"astra.{self.phase}",
            tests=str(len(self.results)),
            failures=str(self.counts()["FAIL"]),
            skipped=str(self.counts()["SKIP"]),
            time=f"{(self.finished or time.time()) - self.started:.3f}",
        )
        for r in self.results:
            case = ET.SubElement(
                suite, "testcase", classname=f"astra.{self.phase}", name=f"{r.id} {r.title}"
            )
            if r.status is Status.FAIL:
                ET.SubElement(case, "failure", message=r.detail).text = r.detail
            elif r.status is Status.SKIP:
                ET.SubElement(case, "skipped", message=r.detail)
            else:
                ET.SubElement(case, "system-out").text = f"{r.status.value}: {r.detail}"
        return ET.tostring(suite, encoding="unicode")

    def render(self, fmt: str) -> str:
        renderers = {
            "table": self.to_table,
            "json": self.to_json,
            "md": self.to_markdown,
            "junit": self.to_junit,
        }
        return renderers[fmt]()

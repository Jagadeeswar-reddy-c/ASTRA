"""Thin, injectable wrapper around subprocess so hardware access is testable."""

from __future__ import annotations

import shutil
import subprocess
from collections.abc import Sequence
from typing import Protocol

from astra.errors import CommandError


class CommandRunner(Protocol):
    def run(self, argv: Sequence[str], timeout: float = 15.0) -> str: ...

    def available(self, program: str) -> bool: ...


class SubprocessRunner:
    def run(self, argv: Sequence[str], timeout: float = 15.0) -> str:
        try:
            proc = subprocess.run(
                list(argv),
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=timeout,
                check=False,
            )
        except FileNotFoundError as exc:
            raise CommandError(f"{argv[0]} not found on PATH") from exc
        except subprocess.TimeoutExpired as exc:
            raise CommandError(f"{argv[0]} timed out after {timeout:.0f}s") from exc
        if proc.returncode != 0:
            detail = (proc.stderr or proc.stdout).strip().splitlines()
            msg = detail[-1] if detail else f"exit code {proc.returncode}"
            raise CommandError(f"{' '.join(argv)} failed: {msg}")
        return proc.stdout

    def available(self, program: str) -> bool:
        return shutil.which(program) is not None

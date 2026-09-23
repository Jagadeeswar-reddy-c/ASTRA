"""Runs against the real NVIDIA driver on the machine executing the tests.

Skipped automatically where nvidia-smi is absent (CI runners). On the ASTRA host
this is the smoke test that the parser understands the installed driver branch.
"""

import shutil

import pytest

from astra.config import AstraConfig
from astra.hardware.probe import probe
from astra.planner.models import from_catalog
from astra.planner.split import budgets_from_gpus, plan
from astra.telemetry.exporter import render_metrics
from astra.validation.checks import LinkContext, run_link_checks
from astra.validation.report import Status

pytestmark = [
    pytest.mark.hardware,
    pytest.mark.skipif(
        shutil.which("nvidia-smi") is None, reason="no NVIDIA driver on this machine"
    ),
]


def test_probe_real_driver() -> None:
    inv = probe()
    assert inv.gpus, "driver present but no GPUs parsed"
    for g in inv.gpus:
        assert g.uuid.startswith("GPU-")
        assert g.memory_total_bytes and g.memory_total_bytes > 1 << 30
        assert g.link_width_max


def test_link_checks_run_cleanly_on_real_driver() -> None:
    inv = probe()
    results = run_link_checks(LinkContext(AstraConfig(), inv, None, "not collected in test"))
    assert all("crashed" not in r.detail for r in results)
    assert next(r for r in results if r.id == "L01").status is Status.PASS


def test_exporter_renders_real_inventory() -> None:
    text = render_metrics(probe(), AstraConfig())
    assert "astra_gpu_memory_total_bytes" in text


def test_plan_against_real_gpus() -> None:
    budgets, _ = budgets_from_gpus(list(probe(with_topology=False).gpus), 0.9, 768)
    result = plan(from_catalog("llama-3.2-3b", "q4_k_m"), budgets, "llamacpp", 4096, 0.9)
    assert sum(result.layer_counts) == 28

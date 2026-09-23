import json
import threading
import urllib.request
from http.server import ThreadingHTTPServer
from pathlib import Path

import pytest

from astra import cli
from astra.config import AstraConfig
from astra.hardware.models import Inventory
from astra.orchestration.ray_pool import ray_resources
from astra.telemetry.exporter import Collector, make_handler, render_metrics
from conftest import UUID_2060, FakeRunner

# -------------------------------------------------------------------------- exporter


def test_metrics_contain_gpu_and_path_series(reference_inventory: Inventory) -> None:
    text = render_metrics(reference_inventory, AstraConfig())
    assert 'astra_up{node="astra-01"} 1' in text
    assert 'astra_gpu_count{node="astra-01"} 2' in text
    assert (
        f'astra_gpu_memory_total_bytes{{node="astra-01",gpu="0",uuid="{UUID_2060}"}} 6442450944'
        in text
    )
    assert 'reason="sw_power_cap"' in text
    assert 'astra_path_bottleneck_link_width{node="astra-01",gpu="0",' in text
    assert 'severity="fatal"' in text
    assert "# TYPE astra_gpu_pcie_replay_total counter" in text


def test_metrics_when_driver_down() -> None:
    text = render_metrics(None, AstraConfig(), scrape_errors=3, last_error="nvidia-smi not found")
    assert 'astra_up{node="astra-01"} 0' in text
    assert 'astra_scrape_errors_total{node="astra-01"} 3' in text
    assert "astra_gpu_memory" not in text


def test_label_escaping(reference_inventory: Inventory) -> None:
    from dataclasses import replace

    inv = replace(
        reference_inventory, gpus=(replace(reference_inventory.gpus[0], name='bad"name\\x'),)
    )
    assert 'name="bad\\"name\\\\x"' in render_metrics(inv, AstraConfig())


def test_collector_rate_limits_and_survives_errors(reference_inventory: Inventory) -> None:
    now = [0.0]
    calls = {"n": 0}

    def probe() -> Inventory:
        calls["n"] += 1
        if calls["n"] == 2:
            raise RuntimeError("driver hiccup")
        return reference_inventory

    c = Collector(probe, AstraConfig(), clock=lambda: now[0])
    assert c.sample() is reference_inventory
    assert c.sample() is reference_inventory and calls["n"] == 1  # cached within 2 s
    now[0] = 5.0
    assert c.sample() is None and c.errors == 1
    assert 'astra_up{node="astra-01"} 0' in c.metrics()  # still cached failure
    now[0] = 10.0
    assert c.sample() is reference_inventory and c.last_error is None


def test_http_endpoints(reference_inventory: Inventory) -> None:
    collector = Collector(lambda: reference_inventory, AstraConfig())
    server = ThreadingHTTPServer(("127.0.0.1", 0), make_handler(collector))
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    base = f"http://127.0.0.1:{server.server_address[1]}"
    try:
        with urllib.request.urlopen(f"{base}/metrics", timeout=5) as resp:
            assert resp.status == 200
            assert "text/plain" in resp.headers["Content-Type"]
            assert b"astra_gpu_temperature_celsius" in resp.read()
        with urllib.request.urlopen(f"{base}/healthz", timeout=5) as resp:
            assert resp.read() == b"ok\n"
    finally:
        server.shutdown()
        server.server_close()


# ------------------------------------------------------------------------------- ray


def test_ray_resources(reference_inventory: Inventory) -> None:
    res = ray_resources(reference_inventory.gpus)
    assert res["gpu_turing"] == 1 and res["gpu_ampere"] == 1
    assert res["gpu_vram_6g"] == 2 and res["gpu_vram_8g"] == 1
    assert res["vram_mib"] == 6144 + 8192


# ------------------------------------------------------------------------------- CLI


@pytest.fixture
def no_config(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("ASTRA_CONFIG", raising=False)
    return tmp_path


def test_cli_plan_reference_fits(no_config: Path, capsys: pytest.CaptureFixture[str]) -> None:
    code = cli.main(
        ["plan", "--simulate", "reference", "--model", "qwen2.5-14b", "--quant", "q4_k_m"]
    )
    out = capsys.readouterr().out
    assert code == 0
    assert "VRAM ratio: 43% : 57%" in out
    assert "--tensor-split" in out and "Fits: YES" in out


def test_cli_plan_not_fitting_exits_1(no_config: Path, capsys: pytest.CaptureFixture[str]) -> None:
    code = cli.main(
        [
            "plan",
            "--simulate",
            "reference",
            "--model",
            "llama-3.1-8b",
            "--quant",
            "f16",
            "--engine",
            "vllm",
        ]
    )
    assert code == 1
    assert "Fits: NO" in capsys.readouterr().out


def test_cli_plan_json_output_file(no_config: Path) -> None:
    out = no_config / "plan.json"
    code = cli.main(
        [
            "plan",
            "--simulate",
            "reference",
            "--model",
            "llama-3.1-8b",
            "--engine",
            "vllm",
            "--output",
            str(out),
        ]
    )
    data = json.loads(out.read_text())
    assert code == 0
    assert data["engine"] == "vllm" and data["model"]["quant"] == "awq-int4"
    assert data["launch"]["env"]["VLLM_PP_LAYER_PARTITION"] == ",".join(
        map(str, data["layer_counts"])
    )


def test_cli_rejects_engine_quant_mismatch(
    no_config: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    code = cli.main(
        ["plan", "--simulate", "reference", "--model", "llama-3.1-8b", "--quant", "awq-int4"]
    )
    assert code == 2
    assert "not supported by llamacpp" in capsys.readouterr().err


def test_cli_bad_simulate_and_gpu_selector(no_config: Path) -> None:
    assert cli.main(["plan", "--simulate", "oops", "--model", "llama-3.1-8b"]) == 2
    assert (
        cli.main(["plan", "--simulate", "reference", "--gpus", "7", "--model", "llama-3.1-8b"]) == 2
    )


def test_cli_plan_gpu_subset(no_config: Path, capsys: pytest.CaptureFixture[str]) -> None:
    code = cli.main(["plan", "--simulate", "reference", "--gpus", "1", "--model", "llama-3.1-8b"])
    assert code == 0
    assert "CUDA_VISIBLE_DEVICES=GPU-sim-0001 " in capsys.readouterr().out


def test_cli_models(no_config: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert cli.main(["models"]) == 0
    assert "qwen2.5-14b" in capsys.readouterr().out


def test_cli_validate_and_probe_with_fake_driver(
    no_config: Path,
    reference_xml: str,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    from astra.hardware import probe as probe_mod

    fake = FakeRunner({"nvidia-smi": reference_xml})
    monkeypatch.setattr(probe_mod, "SubprocessRunner", lambda: fake)
    monkeypatch.setattr(cli, "SubprocessRunner", lambda: fake)
    monkeypatch.setattr(cli.platform, "system", lambda: "Windows")
    monkeypatch.setattr(cli.sysfs.LocalSysfs, "available", lambda self: False)
    assert cli.main(["probe"]) == 0
    assert "RTX 3050" in capsys.readouterr().out
    report = no_config / "r.xml"
    assert cli.main(["validate", "--format", "junit", "--output", str(report)]) == 0
    assert report.read_text().startswith("<testsuite")


def test_cli_launch_dry_run_requires_real_hardware(no_config: Path) -> None:
    assert cli.main(["launch", "--simulate", "reference", "--model", "llama-3.1-8b"]) == 2


def test_cli_runtime_needs_plan(no_config: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert cli.main(["validate", "--phase", "runtime"]) == 2
    assert "--plan" in capsys.readouterr().err


def test_cli_plan_env_file_for_compose(no_config: Path) -> None:
    env = no_config / ".env"
    code = cli.main(
        ["plan", "--simulate", "reference", "--model", "qwen2.5-14b", "--env-file", str(env)]
    )
    values = dict(
        line.split("=", 1) for line in env.read_text().splitlines() if not line.startswith("#")
    )
    assert code == 0
    assert values["ASTRA_GPU_UUIDS"] == "GPU-sim-0000,GPU-sim-0001"
    assert sum(map(int, values["ASTRA_LAYER_SPLIT"].split(","))) == 48
    assert values["ASTRA_N_GPU_LAYERS"] == "49"

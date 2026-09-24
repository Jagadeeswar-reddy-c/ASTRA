"""ASTRA Stack bricks (CR-006 S1-S3, S5; ADR-0014): config, assignment, power, L12/L13."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import replace
from pathlib import Path

import pytest

from astra import asbuilt, cli, pool
from astra.config import AstraConfig, ModuleConfig, load_config, parse_config
from astra.errors import ConfigError
from astra.hardware.compat import analyse
from astra.hardware.models import GpuInfo, GpuPath, Inventory, PcieLink
from astra.hardware.modules import assign
from astra.telemetry.exporter import render_metrics
from astra.validation.checks import LinkContext, check_power_budget, check_stack
from astra.validation.report import Status

BRICKS = """
[[module]]
name = "brick-1"
gpus = ["RTX 2060"]
psu_watts = 450
link_width = 8

[[module]]
name = "brick-2"
gpus = ["RTX 3050"]
psu_watts = 450
overhead_watts = 20
"""


def bricks(text: str = BRICKS) -> AstraConfig:
    import tomllib

    return parse_config(tomllib.loads(text))


# --------------------------------------------------------------------------- config


def test_module_config_parsing() -> None:
    cfg = bricks()
    assert [m.name for m in cfg.modules] == ["brick-1", "brick-2"]
    assert cfg.modules[0].gpus == ("RTX 2060",) and cfg.modules[0].overhead_watts == 15
    assert cfg.modules[1].overhead_watts == 20 and cfg.modules[0].link_width == 8
    assert AstraConfig().modules == ()


@pytest.mark.parametrize(
    ("data", "message"),
    [
        ({"module": {"name": "x"}}, "array of tables"),
        ({"module": [{"name": "x", "gpus": ["a"]}]}, "needs 'name'"),
        ({"module": [{"name": "x", "psu_watts": 450, "gpus": []}]}, "non-empty list"),
        ({"module": [{"name": "x", "psu_watts": 450, "gpus": "RTX"}]}, "non-empty list"),
        ({"module": [{"name": "x", "psu_watts": 450, "gpus": ["a"], "fans": 2}]}, "unknown key"),
        ({"module": [{"name": "x", "psu_watts": 0, "gpus": ["a"]}]}, "psu_watts must be > 0"),
        (
            {"module": [{"name": "x", "psu_watts": 450, "gpus": ["a"]}] * 2},
            "names must be unique",
        ),
    ],
)
def test_module_config_errors(data: dict[str, object], message: str) -> None:
    with pytest.raises(ConfigError, match=message):
        parse_config(data)


@pytest.mark.parametrize(
    ("encoding", "prefix"), [("utf-8", b"\xef\xbb\xbf"), ("utf-16-le", b"\xff\xfe"), ("utf-8", b"")]
)
def test_config_written_by_windows_powershell_loads(
    tmp_path: Path, encoding: str, prefix: bytes
) -> None:
    # PowerShell 5.1: `>` writes UTF-16 LE with a BOM, Out-File -Encoding utf8 adds a BOM.
    path = tmp_path / "astra.toml"
    path.write_bytes(prefix + ('[node]\nname = "ps"\n' + BRICKS).encode(encoding))
    cfg = load_config(path)
    assert cfg.node.name == "ps" and len(cfg.modules) == 2
    path.write_bytes(b"\xff\xfe\x00\xd8")  # a lone surrogate: not decodable
    with pytest.raises(ConfigError):
        load_config(path)


# ----------------------------------------------------------------------- assignment


def test_assignment_prefers_exact_ids_then_names(make_gpu: Callable[..., GpuInfo]) -> None:
    gpus = [
        make_gpu(0, "NVIDIA GeForce RTX 3090", 24576),
        make_gpu(1, "NVIDIA GeForce RTX 3090", 24576),
    ]
    modules = (
        ModuleConfig("a", ("RTX 3090",), 550),  # name: takes whichever 3090 is left
        ModuleConfig("b", ("0000:04:00.0",), 550),  # bus id of GPU 0, short form
        ModuleConfig("c", ("RTX 4090", "GPU-TEST-1"), 850),  # uuid is case-insensitive
    )
    result, owner = assign(gpus, modules)
    by = {a.module.name: a for a in result}
    assert [g.index for g in by["b"].gpus] == [0]
    assert [g.index for g in by["c"].gpus] == [1] and by["c"].missing == ("RTX 4090",)
    assert by["a"].gpus == () and by["a"].missing == ("RTX 3090",)  # both 3090s were claimed
    assert owner == {"GPU-test-0": "b", "GPU-test-1": "c"}


# ------------------------------------------------------------------ power (S2), L12


def test_power_is_budgeted_per_brick(reference_inventory: Inventory) -> None:
    inv = reference_inventory
    cfg = bricks()
    report = analyse(inv.gpus, inv.driver_version, cfg.chassis, inv.paths, modules=cfg.modules)
    m1, m2 = report.modules
    assert [c.gpu.name for c in m1.members] == ["NVIDIA GeForce RTX 2060"]
    assert m1.power.psu_watts == 450 and m1.power.overhead_watts == 15 and m1.power.ok
    assert m2.power.overhead_watts == 20
    assert report.chassis_basis.startswith("GPUs in ASTRA Stack modules")
    assert report.module_of(inv.gpus[0].uuid) in ("brick-1", "brick-2")
    assert report.to_dict()["modules"][0]["name"] == "brick-1"
    assert check_power_budget(LinkContext(cfg, inv, "")).status is Status.PASS

    small = bricks(BRICKS.replace("psu_watts = 450\nlink_width", "psu_watts = 150\nlink_width"))
    report = analyse(inv.gpus, inv.driver_version, small.chassis, inv.paths, modules=small.modules)
    codes = {f.code for f in report.findings}
    assert "module_psu_undersized" in codes and "psu_undersized" not in codes
    result = check_power_budget(LinkContext(small, inv, ""))
    assert result.status is Status.FAIL and "brick-1 needs >= " in result.detail


def test_unknown_power_in_a_brick_warns(make_gpu: Callable[..., GpuInfo]) -> None:
    gpu = make_gpu(0, "Mystery Accelerator", 8192, arch="Ampere")
    inv = Inventory((gpu,), "580.1", "12.9", "Linux", {})
    cfg = replace(AstraConfig(), modules=(ModuleConfig("m", ("Mystery",), 450),))
    assert check_power_budget(LinkContext(cfg, inv, "")).status is Status.WARN


# -------------------------------------------------------------------- L13 (S3)


def test_l13_stack_check(reference_inventory: Inventory) -> None:
    inv = reference_inventory
    assert check_stack(LinkContext(AstraConfig(), inv, "")).status is Status.SKIP
    cfg = bricks(BRICKS.replace("link_width = 8", ""))
    ok = check_stack(LinkContext(cfg, inv, ""))
    assert ok.status is Status.PASS and "brick-1 (1 GPU)" in ok.detail

    missing = bricks(BRICKS.replace('gpus = ["RTX 3050"]', 'gpus = ["RTX 3050", "RTX 4090"]'))
    result = check_stack(LinkContext(missing, inv, ""))
    assert result.status is Status.FAIL and "RTX 4090 missing" in result.detail

    # The 2060 negotiates x4 behind a bad cable although its brick expects x8.
    narrow = replace(inv, gpus=tuple(replace(g, link_width_max=4) for g in inv.gpus))
    result = check_stack(LinkContext(bricks(), narrow, ""))
    assert result.status is Status.FAIL and "x4 < x8" in result.detail
    slow = bricks(BRICKS.replace("link_width = 8", "link_gen = 5"))
    assert "< Gen5" in check_stack(LinkContext(slow, inv, "")).detail


def test_l13_warns_on_daisy_chained_switches(reference_inventory: Inventory) -> None:
    inv = reference_inventory
    cfg = bricks(BRICKS.replace("link_width = 8", ""))
    uuid = next(g.uuid for g in inv.gpus if "2060" in g.name)
    path = inv.paths[uuid]
    plx = [h for h in path.chain if h.vendor_id == "0x10b5"]
    assert len(plx) == 2  # one switch = upstream + downstream port: still a star
    chained = GpuPath(
        path.gpu_bdf,
        (*path.chain[:-1], *[replace(h, bdf=h.bdf + "x") for h in plx], path.chain[-1]),
    )
    inv2 = replace(inv, paths={**inv.paths, uuid: chained})
    result = check_stack(LinkContext(cfg, inv2, ""))
    assert result.status is Status.WARN and "2 switch levels" in result.detail
    assert isinstance(chained.chain[0], PcieLink)


# ---------------------------------------------------------- exporter, as-built, CLI


def test_exporter_module_series(reference_inventory: Inventory) -> None:
    text = render_metrics(reference_inventory, bricks())
    assert 'astra_module_psu_watts{node="astra-01",module="brick-1"} 450' in text
    assert 'astra_module_missing_gpus{node="astra-01",module="brick-2"} 0' in text
    assert 'astra_module_power_limit_watts{node="astra-01",module="brick-1"} 315' in text
    assert 'module="brick-1"' in next(
        line for line in text.splitlines() if line.startswith("astra_gpu_info") and "2060" in line
    )
    plain = render_metrics(reference_inventory, AstraConfig())
    assert "astra_module_psu_watts{" not in plain


def test_emit_config_stack_round_trip(reference_inventory: Inventory) -> None:
    import tomllib

    text = asbuilt.render_config(reference_inventory, AstraConfig(), stack=True)
    cfg = parse_config(tomllib.loads(text))
    assert [m.name for m in cfg.modules] == ["brick-1", "brick-2"]
    assert all(m.gpus[0].count(":") == 2 and m.psu_watts >= 450 for m in cfg.modules)
    assert check_stack(LinkContext(cfg, reference_inventory, "")).status is Status.PASS
    # A config that already has bricks keeps them, even without --stack.
    again = asbuilt.render_config(reference_inventory, bricks())
    assert 'name = "brick-2"' in again and "overhead_watts = 20" in again
    assert "[[module]]" not in asbuilt.render_config(reference_inventory, AstraConfig())


def test_cli_compat_lists_bricks(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    path = tmp_path / "astra.toml"
    path.write_text(BRICKS.replace('"RTX 3050"', '"RTX 3050", "RTX 4090"'), encoding="utf-8")
    code = cli.main(["--config", str(path), "compat", "--simulate", "reference"])
    out = capsys.readouterr().out
    assert code == 1 and "ASTRA Stack modules" in out
    assert "brick-1" in out and "MISSING RTX 4090" in out


def test_console_pool_labels_bricks() -> None:
    cfg = bricks()
    data = pool.collect(cfg, 0.9, 768, simulate="reference").to_dict()
    labels = {g["name"]: g["module"] for n in data["nodes"] for g in n["gpus"]}
    assert labels == {"NVIDIA GeForce RTX 2060": "brick-1", "NVIDIA GeForce RTX 3050": "brick-2"}
    plain = pool.collect(AstraConfig(), 0.9, 768, simulate="reference").to_dict()
    assert all(g["module"] is None for n in plain["nodes"] for g in n["gpus"])

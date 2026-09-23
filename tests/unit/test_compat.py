"""Hardware-agnostic support (ADR-0009): spec lookup, driver/engine compatibility,
power budget, GPU auto-selection (ADR-0010) and the checks/CLI built on them."""

from collections.abc import Callable
from dataclasses import replace
from pathlib import Path

import pytest

from astra import cli
from astra.config import AstraConfig, ChassisConfig
from astra.errors import PlanningError
from astra.hardware import nvsmi, sysfs
from astra.hardware.compat import analyse, capabilities, driver_branch
from astra.hardware.gpu_specs import arch_info, lookup
from astra.hardware.models import GpuInfo, Inventory
from astra.orchestration.ray_pool import ray_resources
from astra.planner.models import from_catalog
from astra.planner.split import budgets_from_gpus, estimate_decode_tps, plan, select_gpus
from astra.telemetry.exporter import render_metrics
from astra.validation.checks import LinkContext, run_link_checks
from astra.validation.report import Status
from conftest import FakeRunner, build_reference_sysfs

GIB = 1 << 30


def gpu(i: int, name: str, gib: float, arch: str | None = None, **kw: object) -> GpuInfo:
    """A GPU as the driver would report it; arch defaults to the spec table's."""
    spec = lookup(name, int(gib * GIB))
    return GpuInfo(
        index=i,
        uuid=f"GPU-{i}",
        name=f"NVIDIA GeForce {name}",
        bus_id=f"00000000:{4 + i:02X}:00.0",
        architecture=arch or (spec.architecture if spec else None),
        memory_total_bytes=int(gib * GIB),
        memory_used_bytes=0,
        **kw,  # type: ignore[arg-type]
    )


BUDGET_MIX = [gpu(0, "GT 1030", 2), gpu(1, "GTX 1660 SUPER", 6), gpu(2, "RTX 3060", 12)]


# ----------------------------------------------------------------------- spec table


@pytest.mark.parametrize(
    ("name", "gib", "model", "bw"),
    [
        ("NVIDIA GeForce RTX 3060 Ti", 8, "RTX 3060 Ti", 448),
        ("NVIDIA GeForce RTX 3060", 12, "RTX 3060", 360),
        ("NVIDIA GeForce RTX 3060", 8, "RTX 3060", 240),  # 8 GB variant: narrower bus
        ("NVIDIA GeForce GTX 1650 SUPER", 4, "GTX 1650 SUPER", 192),
        ("NVIDIA GeForce GTX 1650", 4, "GTX 1650", 128),
        ("NVIDIA GeForce RTX 4070 Ti SUPER", 16, "RTX 4070 Ti SUPER", 672),
        ("NVIDIA GeForce GT 1030", 2, "GT 1030", 48),
        ("NVIDIA GeForce RTX 5090", 32, "RTX 5090", 1792),
    ],
)
def test_lookup(name: str, gib: int, model: str, bw: float) -> None:
    spec = lookup(name, gib * GIB)
    assert spec is not None and spec.model == model and spec.mem_bandwidth_gbps == bw


def test_lookup_facts_that_matter() -> None:
    gt1030 = lookup("GT 1030")
    assert gt1030 and not gt1030.nvenc and gt1030.pcie_lanes == 4 and not gt1030.aux_power
    assert lookup("Quadro P2000") is None
    assert lookup("RTX 30600") is None  # no partial-number matches


def test_arch_info() -> None:
    assert arch_info("Ada Lovelace") == arch_info("ada")
    pascal = arch_info("Pascal")
    assert pascal and pascal.compute_capability == 6.1 and pascal.last_driver_branch == 580
    assert arch_info("Turing") and arch_info("Turing").last_driver_branch is None  # type: ignore[union-attr]
    assert arch_info(None) is None and arch_info("Rubin") is None


def test_capabilities() -> None:
    c = capabilities(gpu(0, "GT 1030", 2))
    assert (c.compute_capability, c.sm, c.engines, c.nvenc) == (6.1, "61", ("llamacpp",), False)
    c = capabilities(gpu(1, "RTX 4060", 8))
    assert c.engines == ("llamacpp", "vllm") and c.architecture == "Ada Lovelace"
    unknown = capabilities(
        GpuInfo(0, "GPU-x", "Mystery Card", "0", memory_total_bytes=GIB, power_max_limit_w=90.0)
    )
    assert (
        unknown.spec is None and unknown.board_power_w == 90.0 and unknown.engines == ("llamacpp",)
    )


# ------------------------------------------------------------------ compatibility


def _codes(report) -> dict[str, str]:  # type: ignore[no-untyped-def]
    return {f.code: f.severity for f in report.findings}


def test_budget_mix_on_r580_needs_pinning_and_cuda12() -> None:
    report = analyse(BUDGET_MIX, "580.95.05", ChassisConfig())
    codes = _codes(report)
    assert report.worst == "warn"
    assert codes["driver_pinned"] == "warn" and codes["cuda12_build"] == "warn"
    assert report.cuda_architectures == "61;75;86"
    assert report.required_driver_window == (455, 580)
    assert codes["no_vllm"] == "info" and codes["small_vram"] == "info"


def test_pascal_with_new_driver_fails() -> None:
    report = analyse(BUDGET_MIX, "610.88", ChassisConfig())
    assert _codes(report)["driver_too_new"] == "fail"
    assert "driver_pinned" not in _codes(report)


def test_turing_and_newer_have_no_driver_ceiling() -> None:
    report = analyse([gpu(0, "RTX 2060", 6), gpu(1, "RTX 5070", 12)], "610.88", ChassisConfig())
    assert report.worst in ("ok", "warn")
    assert "driver_too_new" not in _codes(report) and report.required_driver_window == (570, None)
    assert _codes(report)["multi_arch_build"] == "info"


def test_blackwell_needs_new_enough_driver() -> None:
    report = analyse([gpu(0, "RTX 5060", 8)], "560.35", ChassisConfig())
    assert _codes(report)["driver_too_old"] == "fail"


def test_kepler_and_blackwell_cannot_share_a_driver() -> None:
    kepler = GpuInfo(
        0, "GPU-k", "NVIDIA GeForce GTX 780", "0", architecture="Kepler", memory_total_bytes=3 * GIB
    )
    report = analyse([kepler, gpu(1, "RTX 5070", 12)], None, ChassisConfig())
    codes = _codes(report)
    assert codes["kepler"] == "fail" and codes["driver_window_empty"] == "fail"


def test_no_nvenc_and_slot_count() -> None:
    assert _codes(analyse([gpu(0, "GT 1030", 2)], None, ChassisConfig()))["no_nvenc"] == "info"
    five = [gpu(i, "GTX 1650", 4) for i in range(5)]
    assert _codes(analyse(five, None, ChassisConfig(slots=4)))["slots"] == "fail"


def test_driver_branch() -> None:
    assert driver_branch("580.95.05") == 580 and driver_branch(None) is None
    assert driver_branch("abc") is None


# ------------------------------------------------------------------------ power


def test_power_budget_for_small_cards() -> None:
    pw = analyse([gpu(0, "GT 1030", 2), gpu(1, "GTX 1650", 4)], None, ChassisConfig()).power
    assert pw.gpu_watts == 105 and pw.ok and pw.recommended_psu_watts == 450
    assert pw.slot_powered_watts == 105  # both cards run from slot power only


def test_two_3090s_overload_a_650w_psu() -> None:
    report = analyse([gpu(0, "RTX 3090", 24), gpu(1, "RTX 3090", 24)], None, ChassisConfig())
    assert _codes(report)["psu_undersized"] == "fail"
    assert report.power.recommended_psu_watts == 1200  # (700+25)/0.7 = 1036 W -> 1200 W


def test_unknown_power_is_flagged() -> None:
    mystery = GpuInfo(0, "GPU-m", "Mystery", "0", memory_total_bytes=GIB)
    report = analyse([mystery], None, ChassisConfig())
    assert _codes(report)["power_unknown"] == "warn" and report.power.unknown_gpus == ("Mystery",)


def test_chassis_membership() -> None:
    host = gpu(0, "RTX 3060 Ti", 8, display_active=True)
    ext = gpu(1, "RTX 3050", 8, display_active=False)
    report = analyse([host, ext], None, ChassisConfig())
    assert [c.gpu.uuid for c in report.chassis] == ["GPU-1"]
    assert "display" in report.chassis_basis


def test_chassis_membership_from_topology(reference_xml: str) -> None:
    rep = nvsmi.parse_xml(reference_xml)
    fs = build_reference_sysfs()
    paths = {g.uuid: p for g in rep.gpus if (p := sysfs.gpu_path(g.short_bdf, fs))}
    report = analyse(rep.gpus, rep.driver_version, ChassisConfig(), paths)
    assert len(report.chassis) == 2 and "switch" in report.chassis_basis


# ------------------------------------------------------------------ link gate L11/L12


def test_l11_l12_fail_for_pascal_on_new_driver_and_small_psu(
    reference_inventory: Inventory,
) -> None:
    pascal = replace(
        reference_inventory.gpus[0], name="NVIDIA GeForce GTX 1080 Ti", architecture="Pascal"
    )
    inv = replace(
        reference_inventory, gpus=(pascal, reference_inventory.gpus[1]), driver_version="610.88"
    )
    cfg = replace(AstraConfig(), chassis=ChassisConfig(psu_watts=300))
    results = {r.id: r for r in run_link_checks(LinkContext(cfg, inv, ""))}
    assert results["L11"].status is Status.FAIL and "R580" in results["L11"].detail
    assert results["L12"].status is Status.FAIL and "at least" in results["L12"].detail


# --------------------------------------------------------------------- planning


def _select(
    gpus: list[GpuInfo],
    model: str,
    quant: str,
    engine: str = "llamacpp",
    objective: str = "speed",
    **kw: object,
):  # type: ignore[no-untyped-def]
    budgets, warnings = budgets_from_gpus(gpus, 0.9, 768)
    return select_gpus(
        from_catalog(model, quant), budgets, engine, 8192, 0.9, warnings, objective, **kw
    )  # type: ignore[arg-type]


def test_auto_leaves_slow_small_gpus_out_when_not_needed() -> None:
    result = _select(BUDGET_MIX, "llama-3.1-8b", "q4_k_m")
    assert [p.device.name for p in result.placements] == ["NVIDIA GeForce RTX 3060"]
    assert any("left out" in w and "GT 1030" in w for w in result.warnings)
    assert len(result.alternatives) > 1


def test_auto_pools_when_single_gpu_would_be_too_tight() -> None:
    result = _select(BUDGET_MIX, "qwen2.5-14b", "q4_k_m")
    names = [p.device.name for p in result.placements]
    assert names == ["NVIDIA GeForce GTX 1660 SUPER", "NVIDIA GeForce RTX 3060"]
    assert result.fits and max(p.n_layers for p in result.placements) < 48


def test_speed_gives_slow_gpu_only_the_spill() -> None:
    budgets, _ = budgets_from_gpus([gpu(0, "GT 1030", 2), gpu(1, "RTX 3060", 8)], 0.9, 768)
    model = from_catalog("llama-3.1-8b", "q4_k_m")
    fast = plan(model, budgets, "llamacpp", 8192, 0.9, objective="speed")
    even = plan(model, budgets, "llamacpp", 8192, 0.9, objective="balanced")
    assert fast.fits and even.fits
    assert fast.placements[0].n_layers < even.placements[0].n_layers  # fewer on the GT 1030
    assert (fast.est_decode_tokens_per_s or 0) > (even.est_decode_tokens_per_s or 0)
    # ...and when the fast card alone is too tight, the slow one takes just the spill.
    tight = _select([gpu(0, "GT 1030", 2), gpu(1, "RTX 3060", 8)], "llama-3.1-8b", "q6_k")
    assert tight.fits and tight.placements[0].n_layers < tight.placements[1].n_layers / 4


def test_vllm_auto_skips_pre_volta_and_explicit_use_is_rejected() -> None:
    result = _select(BUDGET_MIX, "llama-3.1-8b", "awq-int4", engine="vllm")
    assert all("GT 1030" not in p.device.name for p in result.placements)
    assert any("GT 1030" in w and "compute capability" in w for w in result.warnings)
    budgets, _ = budgets_from_gpus(BUDGET_MIX[:1], 0.9, 768)
    with pytest.raises(PlanningError, match="not supported by vllm"):
        plan(from_catalog("llama-3.2-3b", "awq-int4"), budgets, "vllm", 4096, 0.9)


def test_exclude_list() -> None:
    result = _select(BUDGET_MIX, "llama-3.1-8b", "q4_k_m", exclude=("RTX 3060",))
    assert all("3060" not in p.device.name for p in result.placements)


def test_nothing_fits_reports_whole_pool() -> None:
    result = _select([gpu(0, "GT 1030", 2), gpu(1, "GTX 1650", 4)], "qwen2.5-14b", "q4_k_m")
    assert not result.fits and len(result.placements) == 2
    assert any("no GPU combination fits" in w for w in result.warnings)


def test_speed_estimate() -> None:
    budgets, _ = budgets_from_gpus([gpu(0, "RTX 3060", 12)], 0.9, 768)
    single = plan(from_catalog("llama-3.1-8b", "q4_k_m"), budgets, "llamacpp", 2048, 0.9)
    tps = single.est_decode_tokens_per_s
    assert tps is not None and 40 < tps < 80  # 360 GB/s x 0.75 / ~4.6 GB ≈ 55
    unknown = replace(
        single.placements[0], device=replace(single.placements[0].device, bandwidth_gbps=None)
    )
    assert estimate_decode_tps([unknown]) is None


# ------------------------------------------------------------------------- misc


def test_ray_resources_typed_by_capability() -> None:
    res = ray_resources(BUDGET_MIX)
    assert res["gpu_pascal"] == 1 and res["gpu_turing"] == 1 and res["gpu_ampere"] == 1
    assert res["gpu_cc_61"] == 3 and res["gpu_cc_75"] == 2 and res["gpu_cc_86"] == 1
    assert res["nvenc"] == 2 and res["gpu_vram_12g"] == 1


def test_exporter_power_and_capability_series(reference_inventory: Inventory) -> None:
    text = render_metrics(reference_inventory, AstraConfig())
    assert 'astra_chassis_psu_watts{node="astra-01"} 650' in text
    assert 'astra_chassis_power_limit_watts{node="astra-01"} 455' in text
    assert 'astra_chassis_rated_power_watts{node="astra-01"} 315' in text  # 160+130+25
    assert 'compute_capability="7.5"' in text and 'in_chassis="true"' in text


def test_parser_reads_display_and_max_power(single_3060ti_xml: str, reference_xml: str) -> None:
    (g,) = nvsmi.parse_xml(single_3060ti_xml).gpus
    assert g.display_active is True and g.power_max_limit_w == 200.0
    assert nvsmi.parse_xml(reference_xml).gpus[0].display_active is None


# -------------------------------------------------------------------------- CLI


@pytest.fixture
def no_config(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("ASTRA_CONFIG", raising=False)
    return tmp_path


def test_cli_simulate_by_name(no_config: Path, capsys: pytest.CaptureFixture[str]) -> None:
    gpus = cli._simulated_gpus("2x GTX 1660 SUPER, GT 1030, RTX 3060:8192")
    assert [g.name for g in gpus] == ["NVIDIA GeForce GTX 1660 SUPER"] * 2 + [
        "NVIDIA GeForce GT 1030",
        "NVIDIA GeForce RTX 3060",
    ]
    assert gpus[3].memory_total_bytes == 8 * GIB and gpus[2].architecture == "Pascal"
    assert cli._simulated_gpus("RTX 3060")[0].memory_total_bytes == 12 * GIB  # largest variant
    assert cli.main(["plan", "--simulate", "Radeon 7900", "--model", "llama-3.1-8b"]) == 2
    assert "unknown GPU" in capsys.readouterr().err


def test_cli_compat(no_config: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert cli.main(["compat", "--simulate", "budget-mix", "--driver", "580.95"]) == 0
    out = capsys.readouterr().out
    assert "R455 to R580" in out and "61;75;86" in out and "GT 1030" in out
    assert cli.main(["compat", "--simulate", "budget-mix", "--driver", "610.88"]) == 1
    assert cli.main(["compat", "--simulate", "2x RTX 3090", "--json"]) == 1
    assert '"recommended_psu_watts": 1200' in capsys.readouterr().out


def test_cli_plan_objective_and_all(no_config: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert cli.main(["plan", "--simulate", "budget-mix", "--model", "llama-3.1-8b"]) == 0
    assert "GPU sets considered" in capsys.readouterr().out
    assert (
        cli.main(
            [
                "plan",
                "--simulate",
                "budget-mix",
                "--gpus",
                "all",
                "--model",
                "llama-3.1-8b",
                "--objective",
                "balanced",
            ]
        )
        == 0
    )
    out = capsys.readouterr().out
    assert "objective balanced" in out and "GT 1030" in out


def test_cli_emit_config(
    no_config: Path,
    reference_xml: str,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    from astra.config import parse_config
    from astra.hardware import probe as probe_mod

    fake = FakeRunner({"nvidia-smi": reference_xml})
    monkeypatch.setattr(probe_mod, "SubprocessRunner", lambda: fake)
    monkeypatch.setattr(cli.sysfs.LocalSysfs, "available", lambda self: False)
    assert cli.main(["probe", "--emit-config"]) == 0
    text = capsys.readouterr().out
    import tomllib

    cfg = parse_config(tomllib.loads(text))
    assert [(g.match, g.vram_mib) for g in cfg.node.expected_gpus] == [
        ("RTX 2060", 6144),
        ("RTX 3050", 8192),
    ]


GpuFactory = Callable[..., GpuInfo]

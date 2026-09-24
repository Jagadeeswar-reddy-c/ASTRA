"""Tensor parallelism (ADR-0016): link detection, planning, speed model, launch, auto A/B."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pytest

from astra import auto, cli
from astra.config import AstraConfig, ChassisConfig
from astra.errors import PlanningError
from astra.hardware import nvsmi, nvtopo
from astra.hardware.compat import analyse
from astra.planner import engines
from astra.planner.models import from_catalog, gguf_download
from astra.planner.split import DeviceBudget, budgets_from_gpus, plan
from astra.planner.tensor import (
    ALLREDUCE_S,
    _shares,
    allreduce_s,
    better,
    classify_links,
    plan_tensor,
)
from astra.pool import simulated
from conftest import FakeRunner, write_gguf


def sim(spec: str) -> list[DeviceBudget]:
    gpus, remote = simulated(spec)
    budgets, _ = budgets_from_gpus(gpus, 0.9, 768)
    return [
        replace(b, node=remote[b.uuid], rpc_endpoint=f"{remote[b.uuid]}:50052")
        if b.uuid in remote
        else b
        for b in budgets
    ]


# ------------------------------------------------------------------ links and detection


def test_allreduce_cost_by_link_and_size() -> None:
    assert allreduce_s("nvlink", 2) < allreduce_s("p2p", 2) < allreduce_s("host", 2)
    assert allreduce_s("host", 4) == pytest.approx(2 * ALLREDUCE_S["host"])  # ring: more steps
    with pytest.raises(PlanningError, match="unknown interconnect"):
        allreduce_s("infiniband", 2)


def test_classify_links() -> None:
    nv = {(0, 1): "NV4"}
    assert classify_links([0, 1], nv) == "nvlink"
    pcie = {(0, 1): "PIX", (0, 2): "PHB", (1, 2): "PHB"}
    assert classify_links([0, 1], pcie, {(0, 1): "OK"}) == "p2p"
    assert classify_links([0, 1, 2], pcie, {(0, 1): "OK", (0, 2): "GNS", (1, 2): "GNS"}) == "host"
    assert classify_links([1, 0], {(0, 1): "NV2"}) == "nvlink"  # order does not matter
    assert classify_links([0], nv) == "host"


def test_parse_p2p_matrix() -> None:
    text = (
        " \x1b[4mGPU0\tGPU1\tGPU2\x1b[0m\n GPU0\tX\tOK\tCNS\n GPU1\tOK\tX\tGNS\n"
        " GPU2\tCNS\tGNS\tX\n\nLegend:\n  X    = Self\n  OK   = Status Ok\n"
    )
    assert nvtopo.parse_p2p(text) == {(0, 1): "OK", (0, 2): "CNS", (1, 2): "GNS"}
    assert nvtopo.query_p2p(FakeRunner({})) == {}  # tool missing: optional


def test_bar1_and_resizable_bar(single_3060ti_xml: str) -> None:
    gpu = nvsmi.parse_xml(single_3060ti_xml).gpus[0]
    assert gpu.bar1_total_bytes == 256 << 20 and gpu.resizable_bar is False
    assert replace(gpu, bar1_total_bytes=8 << 30).resizable_bar is True
    assert replace(gpu, bar1_total_bytes=None).resizable_bar is None


def test_compat_flags_resizable_bar_off_only_with_several_gpus(single_3060ti_xml: str) -> None:
    gpu = nvsmi.parse_xml(single_3060ti_xml).gpus[0]
    one = analyse([gpu], "610.88", ChassisConfig())
    assert "rebar_off" not in {f.code for f in one.findings}
    two = analyse([gpu, replace(gpu, index=1, uuid="GPU-2")], "610.88", ChassisConfig())
    assert "rebar_off" in {f.code for f in two.findings}


# ---------------------------------------------------------------------------- planning


def test_water_filling_shares() -> None:
    shares, fits = _shares([1.0, 1.0], [936.0, 448.0])
    assert fits and shares[0] == pytest.approx(936 / 1384)  # faster card takes more
    capped, fits = _shares([0.4, 1.0], [936.0, 448.0])
    assert fits and capped == pytest.approx([0.4, 0.6])  # memory caps the fast card
    short, fits = _shares([0.3, 0.3], [1.0, 1.0])
    assert not fits and short == pytest.approx([0.5, 0.5])


def test_tensor_split_speeds_up_a_70b() -> None:
    model = from_catalog("llama-3.3-70b", "q4_k_m")
    budgets = sim("2x RTX 5090")  # 2 x 24 GB is too small for a 70B Q4 (see astra size)
    layer = plan(model, budgets, "llamacpp", 8192, 0.9, objective="speed")
    host = plan_tensor(model, budgets, "llamacpp", 8192, 0.9, "host")
    nvlink = plan_tensor(model, budgets, "llamacpp", 8192, 0.9, "nvlink")
    assert host.fits and host.split_mode == "tensor" and host.interconnect == "host"
    tl, th, tn = (p.est_decode_tokens_per_s or 0 for p in (layer, host, nvlink))
    assert 1.3 * tl < th < tn < 2 * tl  # faster than pipeline, never beyond 2 GPUs' worth
    assert host.tensor_split == pytest.approx((0.5, 0.5))
    assert all(p.n_layers == 80 and p.first_layer == 0 for p in host.placements)
    assert host.max_context > 8192
    assert host.to_dict()["split_mode"] == "tensor" and layer.to_dict()["split_mode"] == "layer"
    assert better(layer, host) is host and better(host, layer) is host


def test_model_matches_published_and_measured_behaviour() -> None:
    # Published single-stream: vLLM TP=2 on 2 x RTX 3090 over PCIe x8, no NVLink: +16-35 %
    # for a ~27B 4-bit model. The model must predict a gain of that order for a 32B Q4.
    model = from_catalog("qwen2.5-32b", "q4_k_m")
    one = plan(model, sim("RTX 3090"), "llamacpp", 4096, 0.9)
    two = plan_tensor(model, sim("2x RTX 3090"), "llamacpp", 4096, 0.9, "host")
    gain = (two.est_decode_tokens_per_s or 0) / (one.est_decode_tokens_per_s or 1) - 1
    assert 0.15 <= gain <= 0.6
    # Published: 4 x RTX 3090 over PCIe is no faster than 2 for a 7B (all-reduce bound).
    small = from_catalog("qwen2.5-7b", "q4_k_m")
    tp2 = plan_tensor(small, sim("2x RTX 3090"), "llamacpp", 4096, 0.9, "host")
    tp4 = plan_tensor(small, sim("4x RTX 3090"), "llamacpp", 4096, 0.9, "host")
    assert (tp4.est_decode_tokens_per_s or 0) < 1.1 * (tp2.est_decode_tokens_per_s or 0)
    # Measured (RPC loopback, dev PC): ~0.9 ms per all-reduce across machines, so a 3B in
    # tensor mode ran at 17.6 tok/s. Remote GPUs are therefore refused.
    with pytest.raises(PlanningError, match="one machine"):
        plan_tensor(small, sim("RTX 3090, @desktop RTX 3090"), "llamacpp", 4096, 0.9)


def test_tensor_plan_constraints() -> None:
    model = from_catalog("qwen2.5-7b", "q4_k_m")
    with pytest.raises(PlanningError, match="at least 2"):
        plan_tensor(model, sim("RTX 3090"), "llamacpp", 4096, 0.9)
    q8 = from_catalog("qwen2.5-7b", "q4_k_m", "q8_0")
    with pytest.raises(PlanningError, match="f16 KV"):
        plan_tensor(q8, sim("2x RTX 3090"), "llamacpp", 4096, 0.9)
    awq = from_catalog("qwen2.5-7b", "awq-int4")
    with pytest.raises(PlanningError, match="cannot be split 3 ways"):  # 28 heads, 4 KV heads
        plan_tensor(awq, sim("3x RTX 3090"), "vllm", 4096, 0.9)
    mixed = plan_tensor(awq, sim("RTX 3090, RTX 3060:12288"), "vllm", 4096, 0.9)
    assert mixed.tensor_split == pytest.approx((0.5, 0.5))
    assert any("smallest GPU" in w for w in mixed.warnings)
    big = from_catalog("llama-3.3-70b", "q8_0")
    tight = plan_tensor(big, sim("2x RTX 3060:12288"), "llamacpp", 4096, 0.9)
    assert not tight.fits and any("exceed" in w or "does not fit" in w for w in tight.warnings)


def test_tensor_with_draft_and_mixed_bandwidth() -> None:
    model = from_catalog("qwen2.5-14b", "q4_k_m")
    draft = from_catalog("qwen2.5-0.5b", "q8_0")
    p = plan_tensor(model, sim("RTX 3090, RTX 3060:12288"), "llamacpp", 8192, 0.9, draft=draft)
    assert p.draft is not None and p.draft_stage == 0 and p.placements[0].draft_bytes > 0
    share_fast, share_slow = p.tensor_split
    assert share_fast > share_slow  # 936 vs 360 GB/s: the 3090 takes the bigger share


def test_launch_flags_for_tensor_split() -> None:
    model = from_catalog("llama-3.3-70b", "q4_k_m")
    p = plan_tensor(model, sim("2x RTX 5090"), "llamacpp", 8192, 0.9, "p2p")
    joined = " ".join(engines.llamacpp(p, "m.gguf").argv)
    assert "--split-mode tensor --tensor-split 500,500" in joined and "--flash-attn on" in joined
    awq = from_catalog("qwen2.5-7b", "awq-int4")
    v = plan_tensor(awq, sim("2x RTX 3090"), "vllm", 4096, 0.9)
    spec = engines.vllm(v)
    joined = " ".join(spec.argv)
    assert "--pipeline-parallel-size 1 --tensor-parallel-size 2" in joined
    assert "VLLM_PP_LAYER_PARTITION" not in spec.env


# -------------------------------------------------------------------------------- CLI


def test_cli_plan_split_modes(capsys: pytest.CaptureFixture[str]) -> None:
    base = ["plan", "--simulate", "2x RTX 5090", "--model", "llama-3.3-70b"]
    assert cli.main([*base, "--split", "tensor", "--interconnect", "nvlink"]) == 0
    out = capsys.readouterr().out
    assert "split: tensor (nvlink all-reduce)" in out and "--split-mode tensor" in out
    assert cli.main([*base, "--split", "auto"]) == 0
    out = capsys.readouterr().out
    assert "split: tensor (host all-reduce)" in out and "layer split considered" in out
    assert cli.main(base) == 0
    assert "split: layer (pipeline)" in capsys.readouterr().out
    fabric = ["plan", "--simulate", "RTX 3090, @b RTX 3090", "--model", "qwen2.5-14b"]
    assert cli.main([*fabric, "--split", "auto"]) == 0
    assert "tensor split not possible" in capsys.readouterr().out
    assert cli.main([*fabric, "--split", "tensor"]) == 2
    assert "one machine" in capsys.readouterr().err


def test_probe_shows_p2p_and_bar(
    monkeypatch: pytest.MonkeyPatch, reference_xml: str, capsys: pytest.CaptureFixture[str]
) -> None:
    from astra.hardware import probe as probe_mod

    topo = " GPU0\tX\tPIX\n GPU1\tPIX\tX\n"
    p2p = " GPU0\tX\tOK\n GPU1\tOK\tX\n"
    fake = FakeRunner(
        {"nvidia-smi": reference_xml, "nvidia-smi topo -m": topo, "nvidia-smi topo -p2p r": p2p}
    )
    monkeypatch.setattr(probe_mod, "SubprocessRunner", lambda: fake)
    assert cli.main(["probe"]) == 0
    out = capsys.readouterr().out
    assert "P2P OK (peer-to-peer supported)" in out
    assert "tensor split would all-reduce via: p2p" in out


# ---------------------------------------------------------------------------- auto A/B


class _Proc:
    def poll(self) -> int | None:
        return None

    def terminate(self) -> None:
        pass

    def wait(self, timeout: float | None = None) -> int:
        return 0


@pytest.mark.parametrize(("tensor_tps", "kept"), [(60.0, True), (40.0, False)])
def test_auto_measures_tensor_split(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    reference_xml: str,
    tensor_tps: float,
    kept: bool,
) -> None:
    from astra import pool
    from astra.hardware import probe as probe_mod
    from astra.planner import tensor

    idle = reference_xml.replace("<used>5210 MiB</used>", "<used>200 MiB</used>").replace(
        "<used>6890 MiB</used>", "<used>300 MiB</used>"
    )
    fake = FakeRunner({"nvidia-smi": idle})
    monkeypatch.setattr(probe_mod, "SubprocessRunner", lambda: fake)
    monkeypatch.setattr(pool, "discover", lambda *a, **k: [])
    monkeypatch.setattr(tensor, "detect_interconnect", lambda budgets: "host")
    name, _ = gguf_download("qwen2.5-7b", "q4_k_m")  # type: ignore[misc]
    models = tmp_path / "lab" / "models"
    models.mkdir(parents=True)
    write_gguf(
        models / name,
        {
            "general.architecture": "qwen2",
            "qwen2.block_count": 28,
            "qwen2.embedding_length": 3584,
            "qwen2.attention.head_count": 28,
            "qwen2.attention.head_count_kv": 4,
        },
        [(f"blk.{i}.w", (3584, 3584 * 21), 12) for i in range(28)],
    )
    started: list[list[str]] = []
    monkeypatch.setattr(
        auto, "_start", lambda spec, log, url: started.append(list(spec.argv)) or _Proc()
    )
    monkeypatch.setattr(
        auto,
        "_bench",
        lambda url: auto.Bench(tensor_tps if "tensor" in started[-1] else 45.0, 2000.0, None),
    )
    monkeypatch.setattr(auto, "_help_text", lambda server: "--spec-type")
    monkeypatch.setattr(auto, "_port_in_use", lambda port: False)
    monkeypatch.setattr(auto, "run_runtime_checks", lambda *a, **k: [])
    monkeypatch.setattr(
        auto.threading, "Thread", lambda **k: type("T", (), {"start": lambda s: None})()
    )
    monkeypatch.setattr(auto.time, "sleep", lambda s: None)
    server = tmp_path / "llama-server.exe"
    server.write_bytes(b"")
    cfg = AstraConfig(source=tmp_path / "astra.toml", chassis=ChassisConfig())
    opts = auto.AutoOptions(
        lab=tmp_path / "lab",
        context=4096,
        model="qwen2.5-7b",
        quant="q4_k_m",
        binary=str(server),
        draft="off",
        ui=False,
    )
    out: list[str] = []
    code = auto.run(cfg, opts, out.append)
    text = "\n".join(out)
    assert "tensor split (host)" in text, text
    result_dir = next((tmp_path / "lab").glob("auto-*"))
    result = json.loads((result_dir / "results.json").read_text(encoding="utf-8"))
    if kept:
        assert "keeping tensor split" in text and result["split_mode"] == "tensor"
        assert len(started) == 2 and code in (0, 1)
    else:
        assert "back to the baseline" in text and result["split_mode"] == "layer"
        assert len(started) == 3 and "tensor" not in started[-1]

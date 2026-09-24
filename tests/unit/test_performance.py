"""v0.6.0: bigger models, speculative decoding, KV cache types, stack sizing (CR-006)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from astra import auto, cli
from astra.config import AstraConfig, ChassisConfig
from astra.errors import AstraError, ParseError, PlanningError
from astra.hardware.models import GpuInfo
from astra.planner import engines
from astra.planner.gguf import read_gguf_model, split_paths
from astra.planner.models import (
    CATALOG,
    catalog_key_for,
    from_catalog,
    from_gguf,
    gguf_download,
    gguf_files,
)
from astra.planner.sizing import candidates, size_stack
from astra.planner.split import (
    SPEC_MAX_TPS,
    DeviceBudget,
    budgets_from_gpus,
    draft_bytes,
    plan,
    select_gpus,
)
from astra.pool import simulated
from astra.units import GIB, MIB
from conftest import FakeRunner, write_gguf


def sim(spec: str, util: float = 0.9) -> list[DeviceBudget]:
    gpus, remote = simulated(spec)
    budgets, _ = budgets_from_gpus(gpus, util, 768)
    if not remote:
        return budgets
    from dataclasses import replace

    return [
        replace(b, node=remote[b.uuid], rpc_endpoint=f"{remote[b.uuid]}:50052")
        if b.uuid in remote
        else b
        for b in budgets
    ]


# --------------------------------------------------------------------- catalog / GGUF


def test_catalog_has_stack_class_models_with_drafts() -> None:
    assert {"qwen2.5-32b", "llama-3.3-70b"} <= set(CATALOG)
    assert CATALOG["qwen2.5-32b"].draft == "qwen2.5-0.5b"
    assert CATALOG["llama-3.3-70b"].draft == "llama-3.2-1b"
    assert CATALOG["mistral-7b"].draft is None  # no small model shares its tokenizer
    for key, arch in CATALOG.items():
        if arch.draft:
            assert CATALOG[arch.draft].n_params < arch.n_params / 5, key


def test_split_downloads_for_large_quants() -> None:
    single = gguf_files("llama-3.3-70b", "q4_k_m")
    assert single == [
        (
            "Llama-3.3-70B-Instruct-Q4_K_M.gguf",
            "https://huggingface.co/bartowski/Llama-3.3-70B-Instruct-GGUF/resolve/main/"
            "Llama-3.3-70B-Instruct-Q4_K_M.gguf",
        )
    ]
    parts = gguf_files("llama-3.3-70b", "q8_0")
    assert [n for n, _ in parts] == [
        "Llama-3.3-70B-Instruct-Q8_0-00001-of-00002.gguf",
        "Llama-3.3-70B-Instruct-Q8_0-00002-of-00002.gguf",
    ]
    assert parts[1][1].endswith("/Llama-3.3-70B-Instruct-Q8_0/" + parts[1][0])
    assert gguf_download("llama-3.3-70b", "q8_0") == parts[0]
    assert gguf_files("qwen2.5-7b", "awq-int4") == [] and gguf_files("nope", "q4_k_m") == []


def _meta(n_layers: int, hidden: int, vocab: int = 0, **extra: object) -> dict[str, object]:
    meta: dict[str, object] = {
        "general.architecture": "llama",
        "llama.block_count": n_layers,
        "llama.embedding_length": hidden,
        "llama.attention.head_count": 8,
        "llama.attention.head_count_kv": 2,
        **extra,
    }
    if vocab:
        meta["tokenizer.ggml.tokens"] = ["t"] * vocab
    return meta


def test_split_gguf_is_read_across_parts(tmp_path: Path) -> None:
    first = tmp_path / "m-00001-of-00002.gguf"
    write_gguf(
        first,
        _meta(2, 64, **{"split.count": 2}),
        [("token_embd.weight", (64, 100), 1), ("blk.0.attn_q.weight", (64, 64), 1)],
    )
    write_gguf(
        tmp_path / "m-00002-of-00002.gguf",
        {"split.count": 2},
        [("blk.1.attn_q.weight", (64, 64), 1), ("output.weight", (64, 100), 1)],
    )
    assert [p.name for p in split_paths(first)] == [
        "m-00001-of-00002.gguf",
        "m-00002-of-00002.gguf",
    ]
    assert split_paths(tmp_path / "plain.gguf") == [tmp_path / "plain.gguf"]
    merged = read_gguf_model(first)
    assert len(merged.tensors) == 4
    profile = from_gguf(first)
    assert profile.layer_bytes == (64 * 64 * 2, 64 * 64 * 2)  # both blocks, one per part
    assert profile.head_bytes == 64 * 100 * 2

    (tmp_path / "m-00002-of-00002.gguf").unlink()
    with pytest.raises(ParseError, match="missing m-00002"):
        read_gguf_model(first)


def test_catalog_match_uses_vocabulary(tmp_path: Path) -> None:
    from astra.planner.gguf import read_gguf

    llama = write_gguf(tmp_path / "a.gguf", _meta(16, 2048, vocab=128256), [])
    assert catalog_key_for(read_gguf(llama)) == "llama-3.2-1b"
    # Mistral 7B and Llama 3.1 8B share 32 x 4096; only the vocabulary tells them apart.
    mistral = write_gguf(tmp_path / "b.gguf", _meta(32, 4096, vocab=32768), [])
    assert catalog_key_for(read_gguf(mistral)) == "mistral-7b"
    unknown = write_gguf(tmp_path / "c.gguf", _meta(32, 4096), [])
    assert catalog_key_for(read_gguf(unknown)) is None


# -------------------------------------------------------------- speculative decoding


def test_draft_memory_matches_measurement() -> None:
    # FT-SPEC-01: Qwen2.5-0.5B Q8_0 as a draft at 8k context added 645 MiB on the GPU.
    measured = 645 * MIB
    estimate = draft_bytes(from_catalog("qwen2.5-0.5b", "q8_0"), 8192)
    assert measured * 0.95 <= estimate <= measured * 1.1


def test_plan_places_draft_on_first_local_gpu_without_slowing_estimate() -> None:
    model = from_catalog("qwen2.5-14b", "q4_k_m")
    draft = from_catalog("qwen2.5-0.5b", "q8_0")
    budgets = sim("RTX 3060:12288, RTX 3060:12288")
    base = plan(model, budgets, "llamacpp", 8192, 0.9, objective="speed")
    spec = plan(model, budgets, "llamacpp", 8192, 0.9, objective="speed", draft=draft)
    assert spec.draft_stage == 0 and spec.placements[0].draft_bytes == draft_bytes(draft, 8192)
    assert spec.placements[0].fixed_bytes >= spec.placements[0].draft_bytes
    assert spec.fits
    assert spec.est_decode_tokens_per_s == pytest.approx(base.est_decode_tokens_per_s, rel=0.1)
    doc = spec.to_dict()
    assert doc["draft"]["arch_key"] == "qwen2.5-0.5b" and doc["draft"]["stage"] == 0
    assert doc["devices"][0]["draft_bytes"] > 0 and base.to_dict()["draft"] is None


def test_draft_never_runs_over_the_network_or_on_vllm() -> None:
    model = from_catalog("qwen2.5-7b", "q4_k_m")
    draft = from_catalog("qwen2.5-0.5b", "q8_0")
    remote_only = sim("@desktop RTX 3060 Ti")
    p = plan(model, remote_only, "llamacpp", 4096, 0.9, draft=draft)
    assert p.draft is None and any("no local GPU" in w for w in p.warnings)
    local = sim("@desktop RTX 3060 Ti, RTX 3060:12288")
    p = plan(model, local, "llamacpp", 4096, 0.9, draft=draft)
    assert p.draft_stage == 1  # the local card, even though it is the second stage
    v = plan(
        from_catalog("qwen2.5-7b", "awq-int4"),
        sim("RTX 3060:12288"),
        "vllm",
        4096,
        0.9,
        draft=draft,
    )
    assert v.draft is None and any("llama.cpp only" in w for w in v.warnings)


def test_slow_plans_suggest_speculative_decoding() -> None:
    slow = plan(
        from_catalog("qwen2.5-14b", "q8_0"),
        sim("2x RTX 3060:12288"),
        "llamacpp",
        8192,
        0.9,
        objective="speed",
    )
    assert (slow.est_decode_tokens_per_s or 0) <= SPEC_MAX_TPS
    assert any("--draft auto" in w for w in slow.warnings)
    fast = plan(
        from_catalog("llama-3.2-3b", "q4_k_m"), sim("RTX 3060:12288"), "llamacpp", 8192, 0.9
    )
    assert not any("--draft" in w for w in fast.warnings)


def test_launch_flags_for_kv_type_and_draft() -> None:
    model = from_catalog("qwen2.5-7b", "q4_k_m", "q8_0")
    draft = from_catalog("qwen2.5-0.5b", "q8_0", "q8_0")
    p = plan(model, sim("RTX 3060:12288"), "llamacpp", 8192, 0.9, draft=draft)
    argv = engines.llamacpp(p, "m.gguf", draft_path="d.gguf").argv
    joined = " ".join(argv)
    assert "--flash-attn on --cache-type-k q8_0 --cache-type-v q8_0" in joined
    assert "-md d.gguf -ngld 99 -devd CUDA0" in joined
    assert "--spec-type draft-simple --spec-draft-n-max 4 --spec-draft-p-min 0.75" in joined
    assert "-ctkd q8_0 -ctvd q8_0" in joined
    legacy = " ".join(engines.llamacpp(p, "m.gguf", spec_style="legacy").argv)
    assert "--draft-max 4 --draft-p-min 0.75" in legacy and "--spec-type" not in legacy
    assert "<path/to/draft.gguf>" in legacy
    plain = plan(from_catalog("qwen2.5-7b", "q4_k_m"), sim("RTX 3060:12288"), "llamacpp", 8192, 0.9)
    plain_argv = engines.llamacpp(plain, "m.gguf").argv
    assert "--cache-type-k" not in plain_argv and "-md" not in plain_argv


def test_draft_device_name_in_fabric_plan() -> None:
    model = from_catalog("qwen2.5-14b", "q4_k_m")
    draft = from_catalog("qwen2.5-0.5b", "q8_0")
    p = plan(model, sim("@desktop RTX 3060 Ti, RTX 3060:12288"), "llamacpp", 4096, 0.9, draft=draft)
    argv = list(engines.llamacpp(p, "m.gguf", draft_path="d.gguf").argv)
    assert argv[argv.index("-devd") + 1] == "CUDA0"
    assert argv[argv.index("--device") + 1] == "RPC0,CUDA0"


def test_spec_flag_style() -> None:
    assert engines.spec_flag_style("... --spec-type none,draft-simple ...") == "current"
    assert engines.spec_flag_style("--draft-max N") == "legacy"


# ------------------------------------------------------------------------ CLI


def test_cli_plan_with_draft_and_kv(capsys: pytest.CaptureFixture[str]) -> None:
    args = ["plan", "--simulate", "2x RTX 3090", "--model", "qwen2.5-32b", "--draft", "auto"]
    assert cli.main([*args, "--kv-type", "q8_0"]) == 0
    out = capsys.readouterr().out
    assert "KV cache q8_0" in out and "draft Qwen2.5 0.5B Instruct (q8_0)" in out
    assert "-md '<path/to/draft.gguf>'" in out and "--cache-type-k q8_0" in out
    base = ["plan", "--simulate", "RTX 3090", "--model", "mistral-7b"]
    assert cli.main([*base, "--draft", "auto"]) == 2
    assert "no known draft" in capsys.readouterr().err
    assert cli.main([*base, "--draft", "qwen2.5-0.5b"]) == 0  # explicit choice is honoured
    capsys.readouterr()
    vllm = ["plan", "--simulate", "RTX 3090", "--model", "qwen2.5-7b", "--engine", "vllm"]
    assert cli.main([*vllm, "--kv-type", "q8_0"]) == 2
    assert "llama.cpp option" in capsys.readouterr().err


def test_compose_env_carries_kv_type(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    env = tmp_path / ".env"
    args = ["plan", "--simulate", "RTX 3090", "--model", "qwen2.5-7b", "--env-file", str(env)]
    assert cli.main([*args, "--kv-type", "q8_0", "--draft", "auto"]) == 0
    text = env.read_text(encoding="utf-8")
    assert "ASTRA_KV_TYPE=q8_0" in text and "ASTRA_FLASH_ATTN=on" in text
    assert "does not run the draft" in text
    assert cli.main(args) == 0
    text = env.read_text(encoding="utf-8")
    assert "ASTRA_KV_TYPE=f16" in text and "ASTRA_FLASH_ATTN=auto" in text
    assert "draft" not in text
    capsys.readouterr()


def test_cli_plan_draft_gguf(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    draft = write_gguf(tmp_path / "d.gguf", _meta(2, 64), [("blk.0.x", (64, 64), 1)])
    args = ["plan", "--simulate", "RTX 3090", "--model", "qwen2.5-7b"]
    assert cli.main([*args, "--draft-gguf", str(draft)]) == 0
    assert f"-md {draft}" in capsys.readouterr().out.replace("'", "")


def test_cli_models_lists_drafts(capsys: pytest.CaptureFixture[str]) -> None:
    assert cli.main(["models"]) == 0
    out = capsys.readouterr().out
    assert "llama-3.3-70b" in out and "draft llama-3.2-1b" in out


# --------------------------------------------------------------------- sizing (S4)


def test_size_70b_at_8_tokens_per_second() -> None:
    model = from_catalog("llama-3.3-70b", "q4_k_m")
    options = size_stack(model, 8192, 8.0, 0.9, 768)
    assert options, "some GPU type must reach 8 tok/s"
    best = options[0]
    assert best.gpu == "RTX 5090" and best.count == 2  # fewest cards first
    for o in options:
        assert (o.tokens_per_s or 0) >= 8.0 and o.plan.fits
    by_gpu = {(o.gpu, o.vram_mib): o for o in options}
    assert by_gpu[("RTX 3090", 24 * 1024)].count == 3
    # The study's 4 x RTX 5060 Ti 16 GB stack estimates ~7.8 tok/s: just under 8, over 7.5.
    assert ("RTX 5060 Ti", 16 * 1024) not in by_gpu
    relaxed = {(o.gpu, o.vram_mib): o for o in size_stack(model, 8192, 7.5, 0.9, 768, only="5060")}
    assert relaxed[("RTX 5060 Ti", 16 * 1024)].count == 4
    assert ("RTX 3060", 12 * 1024) not in by_gpu  # 360 GB/s: 70B stays below 8 tok/s
    assert options == sorted(
        options, key=lambda o: (o.count, o.rated_power_w, -(o.tokens_per_s or 0))
    )
    assert best.to_dict()["gpus_used"] == 2 and best.rated_power_w == 2 * 575


def test_size_with_owned_gpus_and_filters() -> None:
    model = from_catalog("qwen2.5-14b", "q4_k_m")
    owned = size_stack(model, 8192, 20.0, 0.9, 768, have="RTX 3060 Ti", only="RTX 3060 Ti")
    assert [(o.gpu, o.count) for o in owned] == [("RTX 3060 Ti", 1)]  # one more 3060 Ti
    assert len(owned[0].plan.placements) == 2
    assert size_stack(model, 8192, 10_000.0, 0.9, 768, only="RTX 4090") == []
    with pytest.raises(PlanningError):
        size_stack(model, 8192, 0, 0.9, 768)
    assert all(s.model != "GT 1030" for s in candidates())
    assert any(s.model == "GT 1030" for s in candidates(include_legacy=True))


def test_cli_size(capsys: pytest.CaptureFixture[str]) -> None:
    assert cli.main(["size", "--model", "llama-3.3-70b", "--gpu", "RTX 3090", "--top", "3"]) == 0
    out = capsys.readouterr().out
    assert "RTX 3090" in out and "EST tok/s" in out
    assert cli.main(["size", "--model", "qwen2.5-7b", "--gpu", "RTX 3060", "--json"]) == 0
    rows = json.loads(capsys.readouterr().out)
    assert rows and rows[0]["count"] == 1
    assert cli.main(["size", "--model", "llama-3.3-70b", "--min-tps", "500"]) == 1
    assert "no GPU type" in capsys.readouterr().out


# ------------------------------------------------------------------- astra auto


def gpu(i: int, name: str, gib: int, arch: str) -> GpuInfo:
    return GpuInfo(
        i,
        f"GPU-{i}",
        f"NVIDIA GeForce {name}",
        f"00000000:{i + 1:02X}:00.0",
        architecture=arch,
        memory_total_bytes=gib * GIB,
        memory_used_bytes=0,
    )


def test_recommend_uses_q8_kv_only_when_needed() -> None:
    budgets, _ = budgets_from_gpus([gpu(0, "RTX 3060 Ti", 8, "Ampere")], 0.9, 768)
    recs = auto.recommend(budgets, AstraConfig(), 8192, use_draft="off")
    assert all(r.draft is None for r in recs)
    for r in recs:
        if r.kv == "q8_0":  # only when the same model/quant does not fit with f16
            f16 = select_gpus(from_catalog(r.model, r.quant), budgets, "llamacpp", 8192, 0.9)
            assert not f16.fits or min(p.headroom_bytes for p in f16.placements) < auto.MIN_HEADROOM
    assert any(r.kv == "f16" for r in recs)


def test_recommend_adds_draft_on_slow_pools() -> None:
    budgets, _ = budgets_from_gpus(
        [gpu(0, "RTX 3060", 12, "Ampere"), gpu(1, "RTX 3060", 12, "Ampere")], 0.9, 768
    )
    cfg = AstraConfig()
    recs = auto.recommend(budgets, cfg, 8192, models=["qwen2.5-14b"], quants=["q8_0"])
    assert recs[0].draft == "qwen2.5-0.5b" and recs[0].plan.draft is not None
    off = auto.recommend(
        budgets, cfg, 8192, models=["qwen2.5-14b"], quants=["q8_0"], use_draft="off"
    )
    assert off[0].draft is None
    fast = auto.recommend(budgets, cfg, 8192, models=["qwen2.5-7b"], quants=["q4_k_m"])
    assert fast[0].draft is None  # ~70 tok/s: speculative decoding would cost speed
    forced = auto.recommend(
        budgets, cfg, 8192, models=["qwen2.5-7b"], quants=["q4_k_m"], use_draft="on"
    )
    assert forced[0].draft == "qwen2.5-0.5b"


def test_ensure_model_downloads_every_part(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    fetched: list[str] = []

    def fake_download(url: str, dest: Path, echo: Any) -> None:
        fetched.append(url)
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(b"x")

    monkeypatch.setattr(auto, "_download", fake_download)
    first = auto.ensure_model(tmp_path, "llama-3.3-70b", "q6_k", lambda _: None)
    assert first.name == "Llama-3.3-70B-Instruct-Q6_K-00001-of-00002.gguf"
    assert len(fetched) == 2 and (tmp_path / "models" / fetched[1].rsplit("/", 1)[1]).is_file()
    auto.ensure_model(tmp_path, "llama-3.3-70b", "q6_k", lambda _: None)
    assert len(fetched) == 2  # present parts are not downloaded again
    with pytest.raises(AstraError):
        auto.ensure_model(tmp_path, "qwen2.5-7b", "awq-int4", lambda _: None)


class _FakeProc:
    def poll(self) -> int | None:
        return None

    def terminate(self) -> None:
        pass

    def wait(self, timeout: float | None = None) -> int:
        return 0


@pytest.mark.parametrize(("draft_tps", "kept"), [(90.0, True), (70.0, False)])
def test_auto_keeps_speculative_decoding_only_if_faster(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    single_3060ti_xml: str,
    draft_tps: float,
    kept: bool,
) -> None:
    from astra import pool
    from astra.hardware import probe as probe_mod

    idle = single_3060ti_xml.replace("<used>2418 MiB</used>", "<used>900 MiB</used>")
    fake = FakeRunner({"nvidia-smi": idle})
    monkeypatch.setattr(probe_mod, "SubprocessRunner", lambda: fake)
    monkeypatch.setattr(pool, "discover", lambda *a, **k: [])
    models = tmp_path / "lab" / "models"
    models.mkdir(parents=True)
    main_file = gguf_download("qwen2.5-7b", "q4_k_m")
    draft_file = gguf_download("qwen2.5-0.5b", "q8_0")
    assert main_file and draft_file
    # Header-only stand-ins with the real shapes: 28 x 3584 (7B) and 24 x 896 (0.5B).
    qwen = {
        "general.architecture": "qwen2",
        "qwen2.attention.head_count": 28,
        "qwen2.attention.head_count_kv": 4,
    }
    write_gguf(
        models / main_file[0],
        {**qwen, "qwen2.block_count": 28, "qwen2.embedding_length": 3584},
        [(f"blk.{i}.w", (3584, 3584 * 21), 12) for i in range(28)],
    )
    write_gguf(
        models / draft_file[0],
        {
            **qwen,
            "qwen2.block_count": 24,
            "qwen2.embedding_length": 896,
            "qwen2.attention.head_count": 14,
            "qwen2.attention.head_count_kv": 2,
        },
        [(f"blk.{i}.w", (896, 896 * 12), 8) for i in range(24)],
    )
    started: list[list[str]] = []

    def fake_start(spec: engines.LaunchSpec, log: Path, url: str) -> _FakeProc:
        started.append(list(spec.argv))
        return _FakeProc()

    def fake_bench(url: str) -> auto.Bench:
        with_draft = "-md" in started[-1]
        return auto.Bench(draft_tps if with_draft else 75.0, 2800.0, 0.9 if with_draft else None)

    monkeypatch.setattr(auto, "_start", fake_start)
    monkeypatch.setattr(auto, "_bench", fake_bench)
    monkeypatch.setattr(auto, "_help_text", lambda server: "--spec-type")
    monkeypatch.setattr(auto, "_port_in_use", lambda port: False)
    monkeypatch.setattr(auto, "run_runtime_checks", lambda *a, **k: [])
    monkeypatch.setattr(
        auto.threading, "Thread", lambda **k: type("T", (), {"start": lambda self: None})()
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
        draft="on",
        ui=False,
    )
    out: list[str] = []
    code = auto.run(cfg, opts, out.append)
    text = "\n".join(out)
    assert code == 0, text
    assert "baseline" in text and "with speculative decoding" in text
    result_dir = next((tmp_path / "lab").glob("auto-*"))
    result = json.loads((result_dir / "results.json").read_text(encoding="utf-8"))
    plan_doc = json.loads((result_dir / "plan.json").read_text(encoding="utf-8"))
    assert len(result["runs"]) == 2 and result["ttft_2k_prompt_s"] == pytest.approx(0.71, 0.01)
    if kept:
        assert "keeping speculative decoding" in text and len(started) == 2
        assert result["draft"] and plan_doc["draft"] is not None
        assert result["measured_tok_s"] == draft_tps
    else:
        assert "did not pay off" in text and len(started) == 3  # restarted the baseline
        assert "-md" not in started[-1] and result["draft"] is None
        assert plan_doc["draft"] is None

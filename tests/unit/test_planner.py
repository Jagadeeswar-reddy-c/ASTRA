from collections.abc import Callable
from pathlib import Path

import pytest

from astra.errors import ParseError, PlanningError
from astra.hardware.models import GpuInfo
from astra.planner import engines
from astra.planner.gguf import read_gguf, tensor_nbytes
from astra.planner.models import from_catalog, from_gguf
from astra.planner.split import budgets_from_gpus, plan
from conftest import write_gguf

GpuFactory = Callable[..., GpuInfo]


@pytest.fixture
def reference(make_gpu: GpuFactory) -> list[GpuInfo]:
    return [
        make_gpu(0, "NVIDIA GeForce RTX 2060", 6144, "Turing"),
        make_gpu(1, "NVIDIA GeForce RTX 3050", 8192, "Ampere"),
    ]


def _plan(
    gpus: list[GpuInfo],
    model: str,
    quant: str,
    engine: str = "llamacpp",
    context: int = 8192,
    util: float = 0.9,
    reserve: int = 768,
):  # type: ignore[no-untyped-def]
    budgets, warnings = budgets_from_gpus(gpus, util, reserve)
    return plan(from_catalog(model, quant), budgets, engine, context, util, warnings)


# ------------------------------------------------------------------ concept-doc rule


def test_vram_ratio_matches_concept_document(reference: list[GpuInfo]) -> None:
    result = _plan(reference, "llama-3.1-8b", "q8_0")
    assert result.vram_ratio == pytest.approx((6 / 14, 8 / 14))  # 43% : 57%


def test_split_tracks_vram_ratio_when_fixed_costs_are_small(reference: list[GpuInfo]) -> None:
    result = _plan(reference, "llama-3.1-8b", "q8_0", reserve=0)
    share_2060 = result.placements[0].n_layers / result.model.n_layers
    assert share_2060 == pytest.approx(6 / 14, abs=0.05)


def test_layers_are_contiguous_and_complete(reference: list[GpuInfo]) -> None:
    result = _plan(reference, "qwen2.5-14b", "q4_k_m")
    assert sum(result.layer_counts) == 48
    assert result.placements[0].first_layer == 0
    assert result.placements[1].first_layer == result.placements[0].n_layers
    assert all(n >= 1 for n in result.layer_counts)


# ------------------------------------------------------------------------- fitting


def test_llama8b_fp16_does_not_fit_14gb_pool(reference: list[GpuInfo]) -> None:
    """Design review DR-03: the concept's Phase 3 command cannot load unquantized 8B."""
    result = _plan(reference, "llama-3.1-8b", "f16", engine="vllm")
    assert not result.fits
    assert result.max_context == 0
    assert any("smaller quantization" in w for w in result.warnings)


def test_llama8b_awq_fits_with_vllm(reference: list[GpuInfo]) -> None:
    result = _plan(reference, "llama-3.1-8b", "awq-int4", engine="vllm")
    assert result.fits
    assert result.placements[0].fixed_bytes > 0  # vLLM keeps embeddings on stage 0
    assert result.placements[1].fixed_bytes > 0  # and lm_head on the last stage


def test_14b_needs_the_pool(reference: list[GpuInfo]) -> None:
    pooled = _plan(reference, "qwen2.5-14b", "q4_k_m")
    assert pooled.fits
    for single in reference:
        assert not _plan([single], "qwen2.5-14b", "q4_k_m").fits


def test_single_gpu_hint_when_pooling_is_unnecessary(reference: list[GpuInfo]) -> None:
    result = _plan(reference, "llama-3.2-3b", "q4_k_m")
    assert result.fits
    assert any("alone" in w for w in result.warnings)


def test_context_overflow_reports_max_context(reference: list[GpuInfo]) -> None:
    fits = _plan(reference, "qwen2.5-14b", "q4_k_m", context=4096)
    too_long = _plan(reference, "qwen2.5-14b", "q4_k_m", context=fits.max_context + 1)
    at_max = _plan(reference, "qwen2.5-14b", "q4_k_m", context=fits.max_context)
    assert fits.fits and at_max.fits and not too_long.fits
    assert any(str(fits.max_context) in w for w in too_long.warnings)


def test_llamacpp_keeps_embeddings_on_host(reference: list[GpuInfo]) -> None:
    result = _plan(reference, "llama-3.1-8b", "q4_k_m")
    assert result.placements[0].fixed_bytes == 0
    assert result.placements[1].fixed_bytes == result.model.head_bytes


def test_baseline_usage_shrinks_budget(make_gpu: GpuFactory) -> None:
    busy = make_gpu(0, "RTX 3050", 8192, used_mib=2048)
    budgets, warnings = budgets_from_gpus([busy], 0.9, 768)
    assert budgets[0].capacity_bytes == (8192 - 2048) << 20
    assert warnings and "already in use" in warnings[0]


def test_pipeline_order_follows_pci_bus(make_gpu: GpuFactory) -> None:
    a, b = make_gpu(0, "A", 6144), make_gpu(1, "B", 8192)
    budgets, _ = budgets_from_gpus([b, a], 0.9, 768)
    assert [x.name for x in budgets] == ["A", "B"]


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"engine": "tgi"}, "unknown engine"),
        ({"context": 0}, "context"),
        ({"reserve": 99999}, "reserve"),
    ],
)
def test_planning_errors(reference: list[GpuInfo], kwargs: dict[str, object], match: str) -> None:
    with pytest.raises(PlanningError, match=match):
        _plan(reference, "llama-3.1-8b", "q4_k_m", **kwargs)  # type: ignore[arg-type]


def test_unknown_catalog_entries() -> None:
    with pytest.raises(PlanningError, match="unknown model"):
        from_catalog("gpt-5", "q4_k_m")
    with pytest.raises(PlanningError, match="quantization"):
        from_catalog("llama-3.1-8b", "q3_weird")


def test_catalog_size_sanity() -> None:
    f16 = from_catalog("llama-3.1-8b", "f16")
    assert 14.5e9 < f16.weights_bytes < 16.5e9  # ~16 GB BF16 checkpoint
    q4 = from_catalog("llama-3.1-8b", "q4_k_m")
    assert 4.5e9 < q4.weights_bytes < 5.3e9  # published Q4_K_M GGUF ≈ 4.9 GB
    assert f16.kv_bytes_per_token_layer == 2 * 8 * 128 * 2  # 4 KiB/token/layer


# ------------------------------------------------------------------------------ GGUF


def _tiny_model(path: Path, n_layers: int = 4, tied: bool = False) -> Path:
    hidden, vocab = 256, 1000
    tensors = [("token_embd.weight", (hidden, vocab), 1)]
    for i in range(n_layers):
        qtype = 14 if i == 0 else 12  # first block Q6_K, rest Q4_K (mixed like Q4_K_M)
        tensors += [
            (f"blk.{i}.attn_q.weight", (hidden, hidden), qtype),
            (f"blk.{i}.ffn_up.weight", (hidden, 4 * hidden), qtype),
            (f"blk.{i}.attn_norm.weight", (hidden,), 0),
        ]
    tensors.append(("output_norm.weight", (hidden,), 0))
    if not tied:
        tensors.append(("output.weight", (hidden, vocab), 14))
    meta = {
        "general.architecture": "llama",
        "general.name": "tiny-test",
        "llama.block_count": n_layers,
        "llama.embedding_length": hidden,
        "llama.attention.head_count": 8,
        "llama.attention.head_count_kv": 2,
        "tokenizer.ggml.tokens": [f"t{i}" for i in range(50)],
    }
    return write_gguf(path, meta, tensors)


def test_gguf_reader(tmp_path: Path) -> None:
    g = read_gguf(_tiny_model(tmp_path / "m.gguf"))
    assert g.version == 3
    assert g.architecture == "llama"
    assert g.arch_value("block_count") == 4
    assert g.metadata["tokenizer.ggml.tokens"] == {"array_len": 50}
    assert len(g.tensors) == 1 + 4 * 3 + 2


def test_gguf_profile_sizes_each_block(tmp_path: Path) -> None:
    profile = from_gguf(_tiny_model(tmp_path / "m.gguf"))
    assert profile.n_layers == 4
    assert profile.layer_bytes[0] > profile.layer_bytes[1]  # Q6_K block is bigger than Q4_K
    assert profile.layer_bytes[1] == profile.layer_bytes[3]
    assert profile.embed_bytes == 256 * 1000 * 2
    assert profile.head_bytes == tensor_nbytes((256, 1000), 14) + 256 * 4
    assert profile.kv_bytes_per_token_layer == 2 * 2 * 32 * 2  # kv_heads=2, head_dim=256/8


def test_gguf_tied_embeddings_put_embedding_on_output_gpu(tmp_path: Path) -> None:
    profile = from_gguf(_tiny_model(tmp_path / "m.gguf", tied=True))
    assert profile.head_bytes >= profile.embed_bytes


def test_gguf_rejects_other_files(tmp_path: Path) -> None:
    bad = tmp_path / "x.bin"
    bad.write_bytes(b"NOPE" + b"\0" * 32)
    with pytest.raises(ParseError, match="not a GGUF"):
        read_gguf(bad)
    trunc = tmp_path / "t.gguf"
    trunc.write_bytes(b"GGUF\x03\x00\x00\x00")
    with pytest.raises(ParseError):
        read_gguf(trunc)


# --------------------------------------------------------------------------- engines


def test_llamacpp_launch(reference: list[GpuInfo]) -> None:
    result = _plan(reference, "qwen2.5-14b", "q4_k_m")
    spec = engines.llamacpp(result, "/models/qwen14b.gguf", port=9000)
    argv = list(spec.argv)
    assert argv[0] == "llama-server"
    assert argv[argv.index("--tensor-split") + 1] == ",".join(map(str, result.layer_counts))
    assert argv[argv.index("--n-gpu-layers") + 1] == "49"
    assert argv[argv.index("--split-mode") + 1] == "layer"
    assert spec.env == {
        "CUDA_DEVICE_ORDER": "PCI_BUS_ID",
        "CUDA_VISIBLE_DEVICES": "GPU-test-0,GPU-test-1",
    }
    assert "CUDA_DEVICE_ORDER=PCI_BUS_ID" in spec.shell()
    assert "$env:CUDA_VISIBLE_DEVICES" in spec.shell(posix=False)


def test_vllm_launch_uses_pipeline_not_tensor_parallel(reference: list[GpuInfo]) -> None:
    """Design review DR-02: uneven VRAM needs PP with an explicit partition, not TP=2."""
    result = _plan(reference, "llama-3.1-8b", "awq-int4", engine="vllm")
    spec = engines.vllm(result)
    argv = list(spec.argv)
    assert argv[:3] == ["vllm", "serve", "hugging-quants/Meta-Llama-3.1-8B-Instruct-AWQ-INT4"]
    assert argv[argv.index("--pipeline-parallel-size") + 1] == "2"
    assert argv[argv.index("--tensor-parallel-size") + 1] == "1"
    assert argv[argv.index("--dtype") + 1] == "float16"  # Turing has no bf16
    assert argv[argv.index("--quantization") + 1] == "awq"
    assert spec.env["VLLM_PP_LAYER_PARTITION"] == ",".join(map(str, result.layer_counts))


def test_engine_mismatch(reference: list[GpuInfo]) -> None:
    result = _plan(reference, "llama-3.1-8b", "q4_k_m")
    with pytest.raises(PlanningError):
        engines.vllm(result)

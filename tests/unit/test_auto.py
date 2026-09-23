"""`astra auto` (ADR-0013): detection-driven configuration, recommendation, engine
build choice, llama.cpp download, NVIDIA topology, and the dry-run flow."""

import io
import json
import zipfile
from pathlib import Path
from typing import Any

import pytest

from astra import auto, cli
from astra.config import AstraConfig, ChassisConfig, parse_config
from astra.errors import AstraError
from astra.hardware import nvtopo
from astra.hardware.compat import capabilities
from astra.hardware.models import GpuInfo, Inventory
from astra.planner.models import gguf_download
from astra.planner.split import budgets_from_gpus
from astra.validation.checks import LinkContext, check_switch
from astra.validation.report import Status
from conftest import FakeRunner

GIB = 1 << 30


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


# ----------------------------------------------------------------- recommendations


def test_recommend_prefers_largest_model_that_fits() -> None:
    budgets, _ = budgets_from_gpus(
        [gpu(0, "RTX 3060", 12, "Ampere"), gpu(1, "GTX 1660 SUPER", 6, "Turing")], 0.9, 768
    )
    recs = auto.recommend(budgets, AstraConfig(), 8192)
    assert recs[0].model == "qwen2.5-14b"  # 18 GB pool: the 14B beats every 7-8B
    assert [r.params_b for r in recs] == sorted((r.params_b for r in recs), reverse=True)
    top14 = [r.quant for r in recs if r.model == "qwen2.5-14b"]
    assert top14 == sorted(top14, key=lambda q: ["q8_0", "q6_k", "q5_k_m", "q4_k_m"].index(q))


def test_recommend_small_gpu_and_filters() -> None:
    budgets, _ = budgets_from_gpus([gpu(0, "RTX 3060 Ti", 8, "Ampere")], 0.9, 768)
    recs = auto.recommend(budgets, AstraConfig(), 8192)
    assert recs and all(r.model != "qwen2.5-14b" for r in recs)
    assert all(min(p.headroom_bytes for p in r.plan.placements) >= auto.MIN_HEADROOM for r in recs)
    assert auto.recommend(budgets, AstraConfig(), 8192, min_tps=10_000) == []
    only = auto.recommend(budgets, AstraConfig(), 8192, models=["llama-3.2-3b"], quants=["q8_0"])
    assert [(r.model, r.quant) for r in only] == [("llama-3.2-3b", "q8_0")]


def test_gguf_download_mapping() -> None:
    name, url = gguf_download("qwen2.5-14b", "q6_k")  # type: ignore[misc]
    assert name == "Qwen2.5-14B-Instruct-Q6_K.gguf"
    assert url == (
        "https://huggingface.co/bartowski/Qwen2.5-14B-Instruct-GGUF/resolve/main/"
        "Qwen2.5-14B-Instruct-Q6_K.gguf"
    )
    assert gguf_download("qwen2.5-14b", "f16") is None
    assert gguf_download("nope", "q4_k_m") is None


# ------------------------------------------------------------------ engine build


@pytest.mark.parametrize(
    ("gpus", "major"),
    [
        ([("RTX 3060", 12, "Ampere")], 12),
        ([("GTX 1070", 8, "Pascal"), ("RTX 3060", 12, "Ampere")], 12),
        ([("RTX 5070", 12, "Blackwell"), ("RTX 3060", 12, "Ampere")], 13),
    ],
)
def test_pick_cuda_major(gpus: list[tuple[str, int, str]], major: int) -> None:
    caps = [capabilities(gpu(i, *g)) for i, g in enumerate(gpus)]
    assert auto.pick_cuda_major(caps) == major


def test_pick_cuda_major_impossible_mix() -> None:
    caps = [
        capabilities(gpu(0, "RTX 5070", 12, "Blackwell")),
        capabilities(gpu(1, "GTX 1070", 8, "Pascal")),
    ]
    with pytest.raises(AstraError, match="build it from source"):
        auto.pick_cuda_major(caps)


def _zip_bytes(*names: str) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        for n in names:
            zf.writestr(n, b"binary")
    return buf.getvalue()


def test_ensure_llama_server_downloads_matching_build(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    releases = [
        {
            "tag_name": "v0.5.0",
            "assets": [{"name": "nightly-tag.txt", "browser_download_url": "x"}],
        },
        {
            "tag_name": "b9",
            "assets": [
                {"name": "llama-b9-bin-win-cuda-13.4-x64.zip", "browser_download_url": "u13"},
                {"name": "llama-b9-bin-win-cuda-12.4-x64.zip", "browser_download_url": "u12"},
                {"name": "cudart-llama-bin-win-cuda-12.4-x64.zip", "browser_download_url": "rt12"},
            ],
        },
    ]

    class Resp(io.BytesIO):
        def __enter__(self) -> "Resp":
            return self

        def __exit__(self, *a: Any) -> None:
            pass

    monkeypatch.setattr(
        auto.urllib.request, "urlopen", lambda *a, **k: Resp(json.dumps(releases).encode())
    )
    fetched: list[str] = []

    def fake_download(url: str, dest: Path, echo: Any) -> None:
        fetched.append(url)
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(
            _zip_bytes("llama-server.exe") if url == "u12" else _zip_bytes("cudart64_12.dll")
        )

    monkeypatch.setattr(auto, "_download", fake_download)
    caps = [capabilities(gpu(0, "RTX 3060", 12, "Ampere"))]
    path = auto.ensure_llama_server(tmp_path, caps, None, lambda s: None, windows=True)
    assert fetched == ["u12", "rt12"]
    assert (
        Path(path).name == "llama-server.exe" and (tmp_path / "llama" / "cudart64_12.dll").exists()
    )


def test_ensure_llama_server_linux_needs_binary(tmp_path: Path) -> None:
    with pytest.raises(AstraError, match="llama-server not found"):
        auto.ensure_llama_server(tmp_path, [], None, lambda s: None, windows=False)


# ------------------------------------------------------------------------ config


def test_sync_config_create_match_and_change(
    tmp_path: Path, reference_inventory: Inventory
) -> None:
    target = tmp_path / "astra.toml"
    cfg = AstraConfig(source=target)
    msgs: list[str] = []
    auto.sync_config(reference_inventory, cfg, msgs.append)
    assert target.exists() and "created" in msgs[-1]
    import tomllib

    written = parse_config(tomllib.loads(target.read_text()), source=target)
    assert [g.match for g in written.node.expected_gpus] == ["RTX 2060", "RTX 3050"]
    auto.sync_config(reference_inventory, written, msgs.append)
    assert "matches" in msgs[-1]
    changed = Inventory(reference_inventory.gpus[:1], reference_inventory.driver_version)
    auto.sync_config(changed, written, msgs.append)
    assert (tmp_path / "astra.toml.new").exists() and any("differ" in m for m in msgs)


# -------------------------------------------------------------------- nvidia topo

TOPO = (
    "\x1b[4mGPU0\tGPU1\tGPU2\tCPU Affinity\tNUMA Affinity\tGPU NUMA ID\x1b[0m\n"
    "GPU0\t X \tPIX\tPHB\t0-15\t0\t\tN/A\n"
    "GPU1\tPIX\t X \tPHB\t0-15\t0\t\tN/A\n"
    "GPU2\tPHB\tPHB\t X \t0-15\t0\t\tN/A\n\nLegend:\n  X    = Self\n"
)


def test_parse_topology() -> None:
    links = nvtopo.parse_topology(TOPO)
    assert links == {(0, 1): "PIX", (0, 2): "PHB", (1, 2): "PHB"}
    assert nvtopo.parse_topology("\x1b[4mGPU0\tCPU Affinity\x1b[0m\nGPU0\t X \t\t\tN/A\n") == {}
    assert nvtopo.describe("NV4") == "NVLink ×4" and "switch" in nvtopo.describe("PIX")


def test_query_topology_tolerates_missing_tool() -> None:
    assert nvtopo.query(FakeRunner({})) == {}
    assert nvtopo.query(FakeRunner({"nvidia-smi": TOPO})) == {
        (0, 1): "PIX",
        (0, 2): "PHB",
        (1, 2): "PHB",
    }


def test_switch_check_from_nvtopo(reference_inventory: Inventory) -> None:
    from dataclasses import replace

    inv = replace(reference_inventory, paths={}, gpu_links={(0, 1): "PIX"})
    assert check_switch(LinkContext(AstraConfig(), inv, "")).status is Status.PASS
    inv = replace(inv, gpu_links={(0, 1): "PHB"})
    assert check_switch(LinkContext(AstraConfig(), inv, "")).status is Status.FAIL
    relaxed = replace(
        AstraConfig(), interconnect=replace(AstraConfig().interconnect, require_switch=False)
    )
    assert check_switch(LinkContext(relaxed, inv, "")).status is Status.PASS


def test_emit_config_uses_nvtopo(reference_inventory: Inventory) -> None:
    from dataclasses import replace

    inv = replace(reference_inventory, paths={}, gpu_links={(0, 1): "PHB"})
    lines = cli._topology_config(inv, AstraConfig(), [g.uuid for g in inv.gpus])
    assert "require_switch = false" in lines


# ----------------------------------------------------------------- dry-run flow


def test_auto_dry_run(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, reference_xml: str) -> None:
    from astra import pool
    from astra.hardware import probe as probe_mod

    idle = reference_xml.replace("<used>5210 MiB</used>", "<used>200 MiB</used>").replace(
        "<used>6890 MiB</used>", "<used>300 MiB</used>"
    )  # the fixture records a busy node; auto runs on an idle one
    fake = FakeRunner({"nvidia-smi": idle})
    monkeypatch.setattr(probe_mod, "SubprocessRunner", lambda: fake)
    monkeypatch.setattr(pool, "discover", lambda *a, **k: [])
    cfg = AstraConfig(source=tmp_path / "astra.toml", chassis=ChassisConfig())
    out: list[str] = []
    opts = auto.AutoOptions(lab=tmp_path / "lab", context=4096, launch=False)
    assert auto.run(cfg, opts, out.append) == 0
    text = "\n".join(out)
    assert "2 NVIDIA GPU(s)" in text and "RTX 2060" in text and "RTX 3050" in text
    assert "created" in text and "-> " in text
    assert not (tmp_path / "lab").exists()  # dry run downloads nothing

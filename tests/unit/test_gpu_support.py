"""Support tiers, `astra gpus`, and the generated docs table staying in sync."""

import json
from pathlib import Path

import pytest

from astra import cli
from astra.hardware.gpu_specs import (
    TIER_BEST_EFFORT,
    TIER_PINNED,
    TIER_RECOMMENDED,
    TIER_UNSUPPORTED,
    arch_info,
    support_tier,
)

DOC = Path(__file__).resolve().parents[2] / "docs" / "03-solution-design" / "supported-gpus.md"


@pytest.mark.parametrize(
    ("arch", "tier"),
    [
        ("Blackwell", TIER_RECOMMENDED),
        ("Turing", TIER_RECOMMENDED),
        ("Volta", TIER_PINNED),
        ("Pascal", TIER_PINNED),
        ("Maxwell", TIER_BEST_EFFORT),
        ("Kepler", TIER_UNSUPPORTED),
        (None, TIER_BEST_EFFORT),
    ],
)
def test_support_tiers(arch: str | None, tier: str) -> None:
    assert support_tier(arch_info(arch)) == tier


def test_gpu_rows_merge_variants_and_order_newest_first() -> None:
    rows = cli.gpu_support_rows()
    assert rows[0]["architecture"] == "Blackwell" and rows[-1]["architecture"] == "Maxwell"
    rtx3060 = next(r for r in rows if r["model"] == "RTX 3060")
    assert rtx3060["vram_gb"] == [8, 12] and rtx3060["engines"] == ["llamacpp", "vllm"]
    gt1030 = next(r for r in rows if r["model"] == "GT 1030")
    assert gt1030["engines"] == ["llamacpp"] and not gt1030["nvenc"] and gt1030["lanes"] == 4


def test_cli_gpus(capsys: pytest.CaptureFixture[str]) -> None:
    assert cli.main(["gpus"]) == 0
    out = capsys.readouterr().out
    assert "Turing (cc 7.5) - recommended" in out and "GT 1030" in out
    assert cli.main(["gpus", "--json"]) == 0
    assert any(r["model"] == "RTX 5090" for r in json.loads(capsys.readouterr().out))


def test_supported_gpus_doc_is_in_sync() -> None:
    """Regenerate with: astra gpus --markdown, and paste between the markers."""
    text = DOC.read_text(encoding="utf-8")
    start = text.index("<!-- BEGIN GENERATED: astra gpus --markdown -->\n") + len(
        "<!-- BEGIN GENERATED: astra gpus --markdown -->\n"
    )
    end = text.index("<!-- END GENERATED -->")
    assert text[start:end] == cli.gpu_support_markdown()

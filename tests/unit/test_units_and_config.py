from pathlib import Path

import pytest

from astra import units
from astra.config import AstraConfig, load_config, parse_config
from astra.errors import ConfigError

REPO = Path(__file__).resolve().parents[2]


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("8192 MiB", 8192 << 20),
        ("6 GiB", 6 << 30),
        ("1024", 1024),
        ("N/A", None),
        ("[N/A]", None),
        ("12 parsecs", None),
        (None, None),
    ],
)
def test_parse_bytes(text: str | None, expected: int | None) -> None:
    assert units.parse_bytes(text) == expected


def test_parse_misc() -> None:
    assert units.parse_number("29.46 W") == pytest.approx(29.46)
    assert units.parse_number("Requested functionality has been deprecated") is None
    assert units.parse_link_width("16x") == 16
    assert units.parse_link_width("x4") == 4
    assert units.gts_to_gen("8.0 GT/s PCIe") == 3
    assert units.gts_to_gen("16.0 GT/s PCIe") == 4
    assert units.gts_to_gen("Unknown") is None


def test_bandwidth_gen3_x4_is_the_astra_uplink() -> None:
    assert units.link_bandwidth_gbps(3, 4) == pytest.approx(3.94, abs=0.01)
    assert units.link_bandwidth_gbps(4, 4) == pytest.approx(7.88, abs=0.01)
    assert units.link_bandwidth_gbps(None, 4) is None


def test_fmt_bytes() -> None:
    assert units.fmt_bytes(512) == "512 B"
    assert units.fmt_bytes(6 << 30) == "6.00 GiB"
    assert units.fmt_bytes(None) == "n/a"


def test_defaults_without_file(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("ASTRA_CONFIG", raising=False)
    cfg = load_config()
    assert cfg == AstraConfig()
    assert cfg.interconnect.expected_uplink_gen == 3


def test_example_config_parses() -> None:
    cfg = load_config(REPO / "config" / "astra.example.toml")
    assert [g.match for g in cfg.node.expected_gpus] == ["RTX 2060", "RTX 3050"]
    assert cfg.node.expected_gpus[0].vram_mib == 6144
    assert cfg.interconnect.switch_vendor_ids == ("0x10b5",)


def test_env_var_path(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    f = tmp_path / "x.toml"
    f.write_text('[node]\nname = "bench-7"\n')
    monkeypatch.setenv("ASTRA_CONFIG", str(f))
    assert load_config().node.name == "bench-7"


@pytest.mark.parametrize(
    "data",
    [
        {"node": {"nmae": "typo"}},
        {"planner": {"gpu_memory_utilization": 1.5}},
        {"thermal": {"warn_c": 90, "crit_c": 80}},
        {"node": {"expected_gpu": [{"vram_mib": 6144}]}},
        {"validation": {"runtime_tolerance": 0}},
    ],
)
def test_invalid_config_rejected(data: dict[str, object]) -> None:
    with pytest.raises(ConfigError):
        parse_config(data)


def test_missing_explicit_file(tmp_path: Path) -> None:
    with pytest.raises(ConfigError, match="not found"):
        load_config(tmp_path / "nope.toml")


def test_malformed_toml(tmp_path: Path) -> None:
    f = tmp_path / "bad.toml"
    f.write_text("[node\n")
    with pytest.raises(ConfigError):
        load_config(f)

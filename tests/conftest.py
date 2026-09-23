"""Shared fixtures: recorded nvidia-smi output, a fake sysfs tree of the reference node,
and a fake command runner so no test needs real hardware."""

from __future__ import annotations

import struct
from collections.abc import Callable, Sequence
from dataclasses import replace
from pathlib import Path, PurePosixPath

import pytest

from astra.errors import CommandError
from astra.hardware import nvsmi, sysfs
from astra.hardware.models import GpuInfo, Inventory

FIXTURES = Path(__file__).parent / "fixtures"


@pytest.fixture(autouse=True)
def _isolated_user_config(
    tmp_path_factory: pytest.TempPathFactory, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Never read the developer's real per-user astra.toml during tests."""
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path_factory.mktemp("xdg")))


UUID_2060 = "GPU-2060aaaa-0000-4000-8000-000000002060"
UUID_3050 = "GPU-3050bbbb-0000-4000-8000-000000003050"


class FakeRunner:
    """Maps argv[0] to canned stdout (or an exception)."""

    def __init__(self, outputs: dict[str, str | Exception]) -> None:
        self.outputs = outputs
        self.calls: list[list[str]] = []

    def run(self, argv: Sequence[str], timeout: float = 15.0) -> str:
        self.calls.append(list(argv))
        out = self.outputs.get(argv[0])
        if out is None:
            raise CommandError(f"{argv[0]} not found on PATH")
        if isinstance(out, Exception):
            raise out
        return out

    def available(self, program: str) -> bool:
        return program in self.outputs


@pytest.fixture
def reference_xml() -> str:
    return (FIXTURES / "nvsmi_reference_node_legacy_driver.xml").read_text(encoding="utf-8")


@pytest.fixture
def single_3060ti_xml() -> str:
    return (FIXTURES / "nvsmi_single_3060ti_driver610.xml").read_text(encoding="utf-8")


class DictSysfs:
    """In-memory sysfs: {posix path: file content} plus a bdf -> device path index."""

    def __init__(self) -> None:
        self.files: dict[str, str] = {}
        self.devices: dict[str, PurePosixPath] = {}

    def add(self, rel: str, **attrs: str) -> None:
        path = PurePosixPath("/sys") / rel
        self.devices[path.name] = path
        for name, value in attrs.items():
            self.files[str(path / name)] = value

    def available(self) -> bool:
        return True

    def device_path(self, bdf: str) -> PurePosixPath | None:
        return self.devices.get(bdf)

    def read(self, path: PurePosixPath) -> str | None:
        return self.files.get(str(path))


def build_reference_sysfs(
    uplink_speed: str = "8.0 GT/s PCIe",
    uplink_width: str = "4",
    uplink_max_speed: str = "16.0 GT/s PCIe",
    aer_correctable: int = 0,
    aer_fatal: int = 0,
) -> DictSysfs:
    """Host root port (Gen4-capable M.2) -> PEX8747 upstream -> 2 downstream ports -> GPUs."""
    fs = DictSysfs()
    rp = "devices/pci0000:00/0000:00:1b.4"
    up = f"{rp}/0000:02:00.0"
    fs.add(
        rp,
        vendor="0x8086",
        device="0x7ac4",
        **{"class": "0x060400"},
        current_link_speed=uplink_speed,
        current_link_width=uplink_width,
        max_link_speed=uplink_max_speed,
        max_link_width="4",
    )
    fs.add(
        up,
        vendor="0x10b5",
        device="0x8747",
        **{"class": "0x060400"},
        current_link_speed=uplink_speed,
        current_link_width=uplink_width,
        max_link_speed="8.0 GT/s PCIe",
        max_link_width="16",
        aer_dev_correctable=f"RxErr {aer_correctable}\nBadTLP 0\nTOTAL_ERR_COR {aer_correctable}",
        aer_dev_nonfatal="Undefined 0\nTOTAL_ERR_NONFATAL 0",
        aer_dev_fatal=f"Undefined 0\nTOTAL_ERR_FATAL {aer_fatal}",
    )
    for port, gpu, width, dev in (
        ("0000:03:08.0", "0000:04:00.0", "16", "0x1f08"),
        ("0000:03:10.0", "0000:05:00.0", "8", "0x2507"),
    ):
        fs.add(
            f"{up}/{port}",
            vendor="0x10b5",
            device="0x8747",
            **{"class": "0x060400"},
            current_link_speed="8.0 GT/s PCIe",
            current_link_width=width,
            max_link_speed="8.0 GT/s PCIe",
            max_link_width="16",
        )
        fs.add(
            f"{up}/{port}/{gpu}",
            vendor="0x10de",
            device=dev,
            **{"class": "0x030000"},
            current_link_speed="8.0 GT/s PCIe",
            current_link_width=width,
            max_link_speed="8.0 GT/s PCIe",
            max_link_width=width,
        )
    return fs


@pytest.fixture
def reference_sysfs() -> DictSysfs:
    return build_reference_sysfs()


@pytest.fixture
def reference_inventory(reference_xml: str, reference_sysfs: DictSysfs) -> Inventory:
    report = nvsmi.parse_xml(reference_xml)
    paths = {}
    for g in report.gpus:
        p = sysfs.gpu_path(g.short_bdf, reference_sysfs)
        assert p is not None
        paths[g.uuid] = p
    return Inventory(report.gpus, report.driver_version, report.cuda_version, "Linux test", paths)


def with_gpu(inv: Inventory, uuid: str, **changes: object) -> Inventory:
    gpus = tuple(replace(g, **changes) if g.uuid == uuid else g for g in inv.gpus)  # type: ignore[arg-type]
    return replace(inv, gpus=gpus)


@pytest.fixture
def make_gpu() -> Callable[..., GpuInfo]:
    def factory(
        index: int, name: str, mib: int, arch: str = "Ampere", used_mib: int = 0
    ) -> GpuInfo:
        return GpuInfo(
            index=index,
            uuid=f"GPU-test-{index}",
            name=name,
            bus_id=f"00000000:{4 + index:02X}:00.0",
            architecture=arch,
            memory_total_bytes=mib << 20,
            memory_used_bytes=used_mib << 20,
        )

    return factory


# ---------------------------------------------------------------------------- GGUF writer


def _gguf_str(s: str) -> bytes:
    raw = s.encode()
    return struct.pack("<Q", len(raw)) + raw


def write_gguf(
    path: Path, metadata: dict[str, object], tensors: list[tuple[str, tuple[int, ...], int]]
) -> Path:
    """Write a header-only GGUF v3 file (tensor data omitted — the reader never needs it)."""
    out = bytearray(b"GGUF" + struct.pack("<IQQ", 3, len(tensors), len(metadata)))
    for key, value in metadata.items():
        out += _gguf_str(key)
        if isinstance(value, str):
            out += struct.pack("<I", 8) + _gguf_str(value)
        elif isinstance(value, bool):
            out += struct.pack("<I?", 7, value)
        elif isinstance(value, int):
            out += struct.pack("<II", 4, value)
        elif isinstance(value, float):
            out += struct.pack("<If", 6, value)
        elif isinstance(value, list):  # array of strings (e.g. vocabulary)
            out += struct.pack("<IIQ", 9, 8, len(value))
            for item in value:
                out += _gguf_str(str(item))
        else:
            raise TypeError(value)
    offset = 0
    for name, shape, ggml_type in tensors:
        out += _gguf_str(name) + struct.pack("<I", len(shape))
        out += b"".join(struct.pack("<Q", d) for d in shape)
        out += struct.pack("<IQ", ggml_type, offset)
        offset += 32
    path.write_bytes(bytes(out))
    return path

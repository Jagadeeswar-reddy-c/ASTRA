from pathlib import Path, PurePosixPath

import pytest

from astra.errors import CommandError, ParseError
from astra.hardware import kernel_log, nvsmi, sysfs
from astra.hardware.probe import probe
from conftest import UUID_2060, UUID_3050, DictSysfs, FakeRunner, build_reference_sysfs

# ------------------------------------------------------------------------ nvidia-smi


def test_parse_current_driver_schema(single_3060ti_xml: str) -> None:
    report = nvsmi.parse_xml(single_3060ti_xml)
    assert report.driver_version == "610.88"
    assert report.cuda_version == "13.3"
    (g,) = report.gpus
    assert g.name == "NVIDIA GeForce RTX 3060 Ti"
    assert g.architecture == "Ampere"
    assert g.memory_total_bytes == 8192 << 20
    assert (g.link_gen_current, g.link_gen_max) == (1, 4)  # idle downshift, as recorded
    assert (g.link_width_current, g.link_width_max) == (16, 16)
    assert g.power_draw_w == pytest.approx(29.61)
    assert g.power_limit_w == pytest.approx(200.0)
    assert g.replay_counter == 0
    assert g.active_clock_events == ("gpu_idle",)
    assert g.fault_slowdowns == ()
    assert g.short_bdf == "0000:01:00.0"


def test_parse_legacy_driver_schema(reference_xml: str) -> None:
    report = nvsmi.parse_xml(reference_xml)
    assert [g.uuid for g in report.gpus] == [UUID_2060, UUID_3050]
    g2060, g3050 = report.gpus
    assert g2060.architecture == "Turing"
    assert g2060.memory_total_bytes == 6144 << 20
    assert g2060.power_draw_w == pytest.approx(151.2)  # <power_readings><power_draw>
    assert g2060.power_limit_w == pytest.approx(160.0)
    assert g2060.active_clock_events == ("sw_power_cap",)  # clocks_throttle_reasons
    assert g2060.fault_slowdowns == ()  # power cap is normal under load, not a fault
    assert g3050.link_width_max == 8  # RTX 3050 is an x8 card
    assert g2060.utilization_pct == 87


def test_parse_rejects_garbage() -> None:
    with pytest.raises(ParseError):
        nvsmi.parse_xml("<not-xml")
    with pytest.raises(ParseError):
        nvsmi.parse_xml("<other/>")
    with pytest.raises(ParseError, match="UUID"):
        nvsmi.parse_xml(
            "<nvidia_smi_log><gpu id='0'><product_name>x</product_name></gpu></nvidia_smi_log>"
        )


def test_query_uses_runner(reference_xml: str) -> None:
    runner = FakeRunner({"nvidia-smi": reference_xml})
    assert len(nvsmi.query(runner).gpus) == 2
    assert runner.calls == [["nvidia-smi", "-q", "-x"]]


def test_query_propagates_missing_driver() -> None:
    with pytest.raises(CommandError):
        nvsmi.query(FakeRunner({}))


# ---------------------------------------------------------------------------- sysfs


def test_gpu_path_walks_switch_and_finds_uplink_bottleneck(reference_sysfs: DictSysfs) -> None:
    path = sysfs.gpu_path("0000:04:00.0", reference_sysfs)
    assert path is not None
    assert [h.bdf for h in path.chain] == [
        "0000:00:1b.4",
        "0000:02:00.0",
        "0000:03:08.0",
        "0000:04:00.0",
    ]
    bottleneck = path.bottleneck
    assert bottleneck is not None
    assert (bottleneck.current_gen, bottleneck.current_width) == (3, 4)
    assert bottleneck.bandwidth_gbps == pytest.approx(3.94, abs=0.01)
    switches = path.switches(("0x10b5",))
    assert [s.bdf for s in switches] == ["0000:02:00.0", "0000:03:08.0"]


def test_aer_counters_summed_along_path() -> None:
    root = build_reference_sysfs(aer_correctable=3, aer_fatal=1)
    path = sysfs.gpu_path("0000:05:00.0", root)
    assert path is not None
    assert (path.aer.correctable, path.aer.fatal, path.aer.total) == (3, 1, 4)


def test_unknown_device(reference_sysfs: DictSysfs) -> None:
    assert sysfs.gpu_path("0000:99:00.0", reference_sysfs) is None


def test_local_sysfs_absent(tmp_path: Path) -> None:
    fs = sysfs.LocalSysfs(tmp_path)
    assert not fs.available()
    assert fs.device_path("0000:01:00.0") is None
    assert fs.read(PurePosixPath(tmp_path.as_posix()) / "missing") is None


def test_probe_combines_driver_and_topology(reference_xml: str, reference_sysfs: DictSysfs) -> None:
    inv = probe(FakeRunner({"nvidia-smi": reference_xml}), fs=reference_sysfs)
    assert set(inv.paths) == {UUID_2060, UUID_3050}
    data = inv.to_dict()
    assert data["paths"][UUID_3050]["bottleneck"]["current_width"] == 4
    assert data["paths"][UUID_3050]["bottleneck_gbps"] == pytest.approx(3.94, abs=0.01)


def test_probe_without_sysfs(reference_xml: str, tmp_path: Path) -> None:
    inv = probe(FakeRunner({"nvidia-smi": reference_xml}), fs=sysfs.LocalSysfs(tmp_path))
    assert inv.paths == {}


# ------------------------------------------------------------------------ kernel log

KERNEL_LOG = """\
[    1.20] pci 0000:02:00.0: [10b5:8747] type 01 class 0x060400
[    1.30] pci 0000:04:00.0: BAR 1: no space for [mem size 0x10000000 64bit pref]
[   88.10] pcieport 0000:02:00.0: AER: Corrected error received: 0000:03:08.0
[  120.00] NVRM: Xid (PCI:0000:05:00): 79, pid=1234, GPU has fallen off the bus.
[  130.00] nvidia-modeset: Loading
"""


def test_scan_classifies_signatures() -> None:
    findings = kernel_log.scan(KERNEL_LOG)
    assert [f.key for f in findings] == ["bar_assign_failed", "aer_corrected", "gpu_fell_off_bus"]
    assert findings[1].severity is kernel_log.Severity.WARN
    assert findings[2].severity is kernel_log.Severity.FAIL
    assert "Above 4G" in findings[0].hint


def test_scan_clean_log() -> None:
    assert (
        kernel_log.scan("[0.0] Linux version 6.8\n[1.0] nvidia: loading out-of-tree module\n") == []
    )


def test_read_kernel_log_falls_back_to_dmesg() -> None:
    runner = FakeRunner({"journalctl": CommandError("denied"), "dmesg": "hello"})
    assert kernel_log.read_kernel_log(runner) == "hello"


def test_read_kernel_log_unavailable() -> None:
    with pytest.raises(CommandError, match="unavailable"):
        kernel_log.read_kernel_log(FakeRunner({"journalctl": CommandError("denied")}))

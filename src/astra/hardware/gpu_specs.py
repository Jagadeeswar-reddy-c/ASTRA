"""Static capabilities of NVIDIA GPUs that nvidia-smi does not report.

ASTRA is hardware-agnostic (ADR-0009): any mix of GT / GTX / RTX cards may be
installed. Planning and validation need facts the driver does not expose —
memory bandwidth (decode speed), board power (PSU sizing), NVENC presence
(transcode placement), compute capability (engine support) and which driver
branches still support the architecture. They live here.

Lookup order: exact model table (by normalised name, disambiguated by VRAM) →
per-architecture defaults → unknown. Unknown values are ``None`` and every
consumer degrades gracefully (e.g. no speed estimate, spec-less power budget).

Figures are reference-board values for desktop cards (NVIDIA spec pages).
Factory-overclocked boards draw more; the power budget's transient factor and
headroom absorb that. Keep this table data-only so reviews are easy.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

# ---------------------------------------------------------------- architectures


@dataclass(frozen=True)
class ArchInfo:
    name: str
    compute_capability: float
    # Last driver branch that supports the architecture; None = still current.
    # Per NVIDIA's lifecycle notices: Kepler ended with R470; Maxwell, Pascal and
    # Volta end with R580 (and CUDA 13 no longer compiles for them).
    last_driver_branch: int | None
    max_cuda_major: int | None  # newest CUDA toolkit major that still targets it
    min_driver_branch: int  # oldest driver branch that knows the architecture
    bf16: bool


ARCHITECTURES: dict[str, ArchInfo] = {
    a.name.lower(): a
    for a in (
        ArchInfo("Kepler", 3.5, 470, 11, 0, False),
        ArchInfo("Maxwell", 5.2, 580, 12, 0, False),
        ArchInfo("Pascal", 6.1, 580, 12, 0, False),
        ArchInfo("Volta", 7.0, 580, 12, 384, False),
        ArchInfo("Turing", 7.5, None, None, 410, False),
        ArchInfo("Ampere", 8.6, None, None, 455, True),
        ArchInfo("Ada Lovelace", 8.9, None, None, 520, True),
        ArchInfo("Hopper", 9.0, None, None, 520, True),
        ArchInfo("Blackwell", 12.0, None, None, 570, True),
    )
}
_ARCH_ALIASES = {"ada": "ada lovelace", "lovelace": "ada lovelace"}


def arch_info(name: str | None) -> ArchInfo | None:
    if not name:
        return None
    key = name.strip().lower()
    return ARCHITECTURES.get(_ARCH_ALIASES.get(key, key))


# ----------------------------------------------------------------------- models


@dataclass(frozen=True)
class GpuSpec:
    model: str  # canonical marketing name, e.g. "RTX 3060 Ti"
    architecture: str
    vram_mib: int
    mem_bandwidth_gbps: float
    board_power_w: int
    nvenc: bool
    pcie_lanes: int
    aux_power: bool  # False = slot-powered only (<= 75 W)


def _s(
    model: str, arch: str, gib: float, bw: float, watts: int, nvenc: bool = True, lanes: int = 16
) -> GpuSpec:
    return GpuSpec(model, arch, int(gib * 1024), bw, watts, nvenc, lanes, watts > 75)


# fmt: off
SPECS: tuple[GpuSpec, ...] = (
    # Maxwell (GTX 9xx)
    _s("GTX 950", "Maxwell", 2, 106, 90), _s("GTX 960", "Maxwell", 2, 112, 120),
    _s("GTX 960", "Maxwell", 4, 112, 120), _s("GTX 970", "Maxwell", 4, 224, 145),
    _s("GTX 980", "Maxwell", 4, 224, 165), _s("GTX 980 Ti", "Maxwell", 6, 336, 250),
    _s("GTX 750 Ti", "Maxwell", 2, 86, 60), _s("GTX 750", "Maxwell", 1, 80, 55),
    # Pascal (GT 1030, GTX 10xx) — GT 1030 has no NVENC and is an x4 card
    _s("GT 1030", "Pascal", 2, 48, 30, nvenc=False, lanes=4),
    _s("GTX 1050", "Pascal", 2, 112, 75), _s("GTX 1050", "Pascal", 3, 84, 75),
    _s("GTX 1050 Ti", "Pascal", 4, 112, 75),
    _s("GTX 1060", "Pascal", 3, 192, 120), _s("GTX 1060", "Pascal", 6, 192, 120),
    _s("GTX 1070", "Pascal", 8, 256, 150), _s("GTX 1070 Ti", "Pascal", 8, 256, 180),
    _s("GTX 1080", "Pascal", 8, 320, 180), _s("GTX 1080 Ti", "Pascal", 11, 484, 250),
    _s("TITAN Xp", "Pascal", 12, 548, 250),
    # Volta
    _s("TITAN V", "Volta", 12, 653, 250),
    # Turing (GTX 16xx, RTX 20xx)
    _s("GTX 1630", "Turing", 4, 96, 75), _s("GTX 1650", "Turing", 4, 128, 75),
    _s("GTX 1650 SUPER", "Turing", 4, 192, 100), _s("GTX 1660", "Turing", 6, 192, 120),
    _s("GTX 1660 SUPER", "Turing", 6, 336, 125), _s("GTX 1660 Ti", "Turing", 6, 288, 120),
    _s("RTX 2060", "Turing", 6, 336, 160), _s("RTX 2060", "Turing", 12, 336, 185),
    _s("RTX 2060 SUPER", "Turing", 8, 448, 175), _s("RTX 2070", "Turing", 8, 448, 175),
    _s("RTX 2070 SUPER", "Turing", 8, 448, 215), _s("RTX 2080", "Turing", 8, 448, 215),
    _s("RTX 2080 SUPER", "Turing", 8, 496, 250), _s("RTX 2080 Ti", "Turing", 11, 616, 250),
    _s("TITAN RTX", "Turing", 24, 672, 280),
    # Ampere (RTX 30xx) — RTX 3050 6 GB is slot-powered and x8
    _s("RTX 3050", "Ampere", 6, 168, 70, lanes=8), _s("RTX 3050", "Ampere", 8, 224, 130, lanes=8),
    _s("RTX 3060", "Ampere", 8, 240, 170), _s("RTX 3060", "Ampere", 12, 360, 170),
    _s("RTX 3060 Ti", "Ampere", 8, 448, 200), _s("RTX 3070", "Ampere", 8, 448, 220),
    _s("RTX 3070 Ti", "Ampere", 8, 608, 290), _s("RTX 3080", "Ampere", 10, 760, 320),
    _s("RTX 3080", "Ampere", 12, 912, 350), _s("RTX 3080 Ti", "Ampere", 12, 912, 350),
    _s("RTX 3090", "Ampere", 24, 936, 350), _s("RTX 3090 Ti", "Ampere", 24, 1008, 450),
    # Ada Lovelace (RTX 40xx)
    _s("RTX 4060", "Ada Lovelace", 8, 272, 115, lanes=8),
    _s("RTX 4060 Ti", "Ada Lovelace", 8, 288, 160, lanes=8),
    _s("RTX 4060 Ti", "Ada Lovelace", 16, 288, 165, lanes=8),
    _s("RTX 4070", "Ada Lovelace", 12, 504, 200),
    _s("RTX 4070 SUPER", "Ada Lovelace", 12, 504, 220),
    _s("RTX 4070 Ti", "Ada Lovelace", 12, 504, 285),
    _s("RTX 4070 Ti SUPER", "Ada Lovelace", 16, 672, 285),
    _s("RTX 4080", "Ada Lovelace", 16, 717, 320),
    _s("RTX 4080 SUPER", "Ada Lovelace", 16, 736, 320),
    _s("RTX 4090", "Ada Lovelace", 24, 1008, 450),
    # Blackwell (RTX 50xx)
    _s("RTX 5060", "Blackwell", 8, 448, 145, lanes=8),
    _s("RTX 5060 Ti", "Blackwell", 8, 448, 180, lanes=8),
    _s("RTX 5060 Ti", "Blackwell", 16, 448, 180, lanes=8),
    _s("RTX 5070", "Blackwell", 12, 672, 250), _s("RTX 5070 Ti", "Blackwell", 16, 896, 300),
    _s("RTX 5080", "Blackwell", 16, 960, 360), _s("RTX 5090", "Blackwell", 32, 1792, 575),
)
# fmt: on

# Support tiers (ADR-0009). Kept here so docs, CLI and console agree.
TIER_RECOMMENDED = "recommended"
TIER_PINNED = "supported (driver <= R580)"
TIER_BEST_EFFORT = "best effort (driver <= R580)"
TIER_UNSUPPORTED = "unsupported"


def support_tier(arch: ArchInfo | None) -> str:
    """How well ASTRA supports an architecture.

    recommended   Turing and newer: any current driver, llama.cpp and vLLM.
    supported     Pascal / Volta: works, but pins the host to the R580 driver branch
                  and needs CUDA 12 engine builds.
    best effort   Maxwell: as above and below ASTRA's qualified compute capability (6.1).
    unsupported   Kepler and older: no driver that also runs current GPUs.
    """
    if arch is None:
        return TIER_BEST_EFFORT
    if arch.last_driver_branch is None:
        return TIER_RECOMMENDED
    if arch.last_driver_branch <= 470:
        return TIER_UNSUPPORTED
    return TIER_PINNED if arch.compute_capability >= 6.1 else TIER_BEST_EFFORT


_NOISE = re.compile(r"\b(nvidia|geforce|graphics|laptop gpu)\b")


def _norm(name: str) -> str:
    text = _NOISE.sub(" ", name.lower())
    text = re.sub(r"(\d)\s*(gb|g)\b", r"\1gb", text)
    return " ".join(text.split())


_BY_KEY: dict[str, list[GpuSpec]] = {}
for _spec in SPECS:
    _BY_KEY.setdefault(_norm(_spec.model), []).append(_spec)


def lookup(name: str, vram_bytes: int | None = None) -> GpuSpec | None:
    """Find the spec for a product name such as 'NVIDIA GeForce RTX 3060 Ti'.

    The longest model key that appears as a whole-word run wins ('rtx 3060 ti'
    beats 'rtx 3060'); among same-name variants the closest VRAM size wins.
    """
    norm = _norm(name)
    best_key = None
    for key in _BY_KEY:
        if re.search(rf"(?<![\w]){re.escape(key)}(?![\w])", norm) and (
            best_key is None or len(key) > len(best_key)
        ):
            best_key = key
    if best_key is None:
        return None
    variants = _BY_KEY[best_key]
    if vram_bytes is None or len(variants) == 1:
        return variants[0]
    mib = vram_bytes / (1024 * 1024)
    return min(variants, key=lambda s: abs(s.vram_mib - mib))

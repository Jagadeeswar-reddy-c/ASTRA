"""Minimal GGUF (llama.cpp model file) header reader.

Reads the metadata key/values and the tensor table — never the tensor data — so
the planner can size every transformer block exactly, even for mixed-precision
quantizations where block sizes differ (e.g. Q4_K_M keeps some tensors at Q6_K).
Format reference: https://github.com/ggml-org/ggml/blob/master/docs/gguf.md
"""

from __future__ import annotations

import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Any, BinaryIO

from astra.errors import ParseError

GGUF_MAGIC = b"GGUF"

# GGUF metadata value types -> struct format (scalars only).
_SCALARS = {
    0: "<B",
    1: "<b",
    2: "<H",
    3: "<h",
    4: "<I",
    5: "<i",
    6: "<f",
    7: "<?",
    10: "<Q",
    11: "<q",
    12: "<d",
}
_STRING, _ARRAY = 8, 9

# ggml tensor type id -> (elements per block, bytes per block).
GGML_TYPE_SIZES: dict[int, tuple[int, int]] = {
    0: (1, 4),  # F32
    1: (1, 2),  # F16
    2: (32, 18),  # Q4_0
    3: (32, 20),  # Q4_1
    6: (32, 22),  # Q5_0
    7: (32, 24),  # Q5_1
    8: (32, 34),  # Q8_0
    9: (32, 36),  # Q8_1
    10: (256, 84),  # Q2_K
    11: (256, 110),  # Q3_K
    12: (256, 144),  # Q4_K
    13: (256, 176),  # Q5_K
    14: (256, 210),  # Q6_K
    15: (256, 292),  # Q8_K
    16: (256, 66),  # IQ2_XXS
    17: (256, 74),  # IQ2_XS
    18: (256, 98),  # IQ3_XXS
    19: (256, 50),  # IQ1_S
    20: (32, 18),  # IQ4_NL
    21: (256, 110),  # IQ3_S
    22: (256, 82),  # IQ2_S
    23: (256, 136),  # IQ4_XS
    24: (1, 1),  # I8
    25: (1, 2),  # I16
    26: (1, 4),  # I32
    27: (1, 8),  # I64
    28: (1, 8),  # F64
    29: (256, 56),  # IQ1_M
    30: (1, 2),  # BF16
    34: (256, 54),  # TQ1_0
    35: (256, 66),  # TQ2_0
    39: (32, 17),  # MXFP4
}


@dataclass(frozen=True)
class TensorInfo:
    name: str
    shape: tuple[int, ...]
    ggml_type: int
    nbytes: int | None  # None when the tensor type is unknown to this reader


@dataclass(frozen=True)
class GgufFile:
    path: Path
    version: int
    metadata: dict[str, Any]
    tensors: tuple[TensorInfo, ...]
    file_size: int

    @property
    def architecture(self) -> str:
        return str(self.metadata.get("general.architecture", "unknown"))

    def arch_value(self, key: str, default: Any = None) -> Any:
        return self.metadata.get(f"{self.architecture}.{key}", default)


def _read(fh: BinaryIO, fmt: str) -> Any:
    size = struct.calcsize(fmt)
    data = fh.read(size)
    if len(data) != size:
        raise ParseError("unexpected end of GGUF header")
    return struct.unpack(fmt, data)[0]


def _read_string(fh: BinaryIO) -> str:
    length = _read(fh, "<Q")
    if length > 1 << 24:
        raise ParseError(f"implausible GGUF string length {length}")
    raw = fh.read(length)
    if len(raw) != length:
        raise ParseError("unexpected end of GGUF string")
    return raw.decode("utf-8", errors="replace")


def _read_value(fh: BinaryIO, vtype: int, keep_arrays: bool) -> Any:
    if vtype in _SCALARS:
        return _read(fh, _SCALARS[vtype])
    if vtype == _STRING:
        return _read_string(fh)
    if vtype == _ARRAY:
        item_type = _read(fh, "<I")
        count = _read(fh, "<Q")
        if item_type in _SCALARS and not keep_arrays:
            fh.seek(struct.calcsize(_SCALARS[item_type]) * count, 1)
            return {"array_len": count}
        items = [_read_value(fh, item_type, keep_arrays) for _ in range(count)]
        return items if keep_arrays else {"array_len": count}
    raise ParseError(f"unknown GGUF value type {vtype}")


def tensor_nbytes(shape: tuple[int, ...], ggml_type: int) -> int | None:
    if ggml_type not in GGML_TYPE_SIZES:
        return None
    block_elems, block_bytes = GGML_TYPE_SIZES[ggml_type]
    elements = 1
    for dim in shape:
        elements *= dim
    return (elements // block_elems) * block_bytes


def read_gguf(path: str | Path, keep_arrays: bool = False) -> GgufFile:
    """Parse the header of a GGUF v2/v3 file. Large arrays (the vocabulary) are skipped."""
    p = Path(path)
    with p.open("rb") as fh:
        if fh.read(4) != GGUF_MAGIC:
            raise ParseError(f"{p} is not a GGUF file")
        version = _read(fh, "<I")
        if version not in (2, 3):
            raise ParseError(f"unsupported GGUF version {version}")
        tensor_count = _read(fh, "<Q")
        kv_count = _read(fh, "<Q")

        metadata: dict[str, Any] = {}
        for _ in range(kv_count):
            key = _read_string(fh)
            vtype = _read(fh, "<I")
            metadata[key] = _read_value(fh, vtype, keep_arrays)

        tensors = []
        for _ in range(tensor_count):
            name = _read_string(fh)
            n_dims = _read(fh, "<I")
            shape = tuple(_read(fh, "<Q") for _ in range(n_dims))
            ggml_type = _read(fh, "<I")
            _read(fh, "<Q")  # data offset — not needed for sizing
            tensors.append(TensorInfo(name, shape, ggml_type, tensor_nbytes(shape, ggml_type)))

    return GgufFile(p, version, metadata, tuple(tensors), p.stat().st_size)

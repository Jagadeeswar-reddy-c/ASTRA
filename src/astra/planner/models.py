"""Model memory profiles: how many bytes each transformer block, the embeddings,
the output head and the KV cache need.

Profiles come either from the built-in architecture catalog combined with a
quantization format, or exactly from a GGUF file's tensor table.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, replace
from pathlib import Path

from astra.errors import PlanningError
from astra.planner.gguf import GgufFile, read_gguf


@dataclass(frozen=True)
class Architecture:
    key: str
    display: str
    hf_repo: str
    n_params: float
    n_layers: int
    hidden: int
    n_heads: int
    n_kv_heads: int
    head_dim: int
    vocab: int
    tied_embeddings: bool = False
    awq_repo: str | None = None  # pre-quantized checkpoint for vLLM, when one is published
    # GGUF downloads for llama.cpp: https://huggingface.co/<gguf_repo>/resolve/main/<stem>-<Q>.gguf
    gguf_repo: str | None = None
    gguf_stem: str | None = None


@dataclass(frozen=True)
class Quant:
    key: str
    layer_bits: float  # effective bits per weight in transformer blocks
    embed_bits: float  # token embedding
    head_bits: float  # output projection (lm_head)
    engines: tuple[str, ...]


CATALOG: dict[str, Architecture] = {
    a.key: a
    for a in (
        Architecture(
            "llama-3.2-3b",
            "Llama 3.2 3B Instruct",
            "meta-llama/Llama-3.2-3B-Instruct",
            3.21e9,
            28,
            3072,
            24,
            8,
            128,
            128256,
            tied_embeddings=True,
        ),
        Architecture(
            "llama-3.1-8b",
            "Llama 3.1 8B Instruct",
            "meta-llama/Llama-3.1-8B-Instruct",
            8.03e9,
            32,
            4096,
            32,
            8,
            128,
            128256,
            awq_repo="hugging-quants/Meta-Llama-3.1-8B-Instruct-AWQ-INT4",
        ),
        Architecture(
            "mistral-7b",
            "Mistral 7B Instruct v0.3",
            "mistralai/Mistral-7B-Instruct-v0.3",
            7.25e9,
            32,
            4096,
            32,
            8,
            128,
            32768,
        ),
        Architecture(
            "qwen2.5-7b",
            "Qwen2.5 7B Instruct",
            "Qwen/Qwen2.5-7B-Instruct",
            7.62e9,
            28,
            3584,
            28,
            4,
            128,
            152064,
            awq_repo="Qwen/Qwen2.5-7B-Instruct-AWQ",
        ),
        Architecture(
            "qwen2.5-14b",
            "Qwen2.5 14B Instruct",
            "Qwen/Qwen2.5-14B-Instruct",
            14.77e9,
            48,
            5120,
            40,
            8,
            128,
            152064,
            awq_repo="Qwen/Qwen2.5-14B-Instruct-AWQ",
        ),
    )
}

_GGUF_SOURCES = {
    "llama-3.2-3b": ("bartowski/Llama-3.2-3B-Instruct-GGUF", "Llama-3.2-3B-Instruct"),
    "llama-3.1-8b": ("bartowski/Meta-Llama-3.1-8B-Instruct-GGUF", "Meta-Llama-3.1-8B-Instruct"),
    "mistral-7b": ("bartowski/Mistral-7B-Instruct-v0.3-GGUF", "Mistral-7B-Instruct-v0.3"),
    "qwen2.5-7b": ("bartowski/Qwen2.5-7B-Instruct-GGUF", "Qwen2.5-7B-Instruct"),
    "qwen2.5-14b": ("bartowski/Qwen2.5-14B-Instruct-GGUF", "Qwen2.5-14B-Instruct"),
}
CATALOG = {
    k: replace(a, gguf_repo=_GGUF_SOURCES[k][0], gguf_stem=_GGUF_SOURCES[k][1])
    if k in _GGUF_SOURCES
    else a
    for k, a in CATALOG.items()
}
# Quantizations with published GGUF files (the file-name tag for each).
GGUF_QUANT_TAGS = {"q4_k_m": "Q4_K_M", "q5_k_m": "Q5_K_M", "q6_k": "Q6_K", "q8_0": "Q8_0"}


def gguf_download(arch_key: str, quant_key: str) -> tuple[str, str] | None:
    """(file name, URL) of a published GGUF for a catalog model, or None."""
    arch = CATALOG.get(arch_key)
    tag = GGUF_QUANT_TAGS.get(quant_key)
    if arch is None or tag is None or not arch.gguf_repo or not arch.gguf_stem:
        return None
    name = f"{arch.gguf_stem}-{tag}.gguf"
    return name, f"https://huggingface.co/{arch.gguf_repo}/resolve/main/{name}"


QUANTS: dict[str, Quant] = {
    q.key: q
    for q in (
        Quant("f16", 16.0, 16.0, 16.0, ("vllm", "llamacpp")),
        Quant("q8_0", 8.5, 8.5, 8.5, ("llamacpp",)),
        Quant("q6_k", 6.56, 6.56, 6.56, ("llamacpp",)),
        Quant("q5_k_m", 5.69, 5.5, 6.56, ("llamacpp",)),
        Quant("q4_k_m", 4.85, 4.5, 6.56, ("llamacpp",)),
        # AWQ/GPTQ keep embeddings and lm_head in FP16.
        Quant("awq-int4", 4.25, 16.0, 16.0, ("vllm",)),
        Quant("gptq-int4", 4.25, 16.0, 16.0, ("vllm",)),
    )
}

KV_BYTES = {"f16": 2.0, "q8_0": 34 / 32, "q4_0": 18 / 32}


@dataclass(frozen=True)
class ModelProfile:
    name: str
    source: str  # "catalog:<arch>/<quant>" or "gguf:<path>"
    n_layers: int
    layer_bytes: tuple[int, ...]
    embed_bytes: int
    head_bytes: int  # output projection + final norm; 0 when tied to the embedding
    kv_bytes_per_token_layer: float  # K+V for one token in one layer at the chosen KV dtype
    hf_repo: str | None = None
    quant: str | None = None

    @property
    def weights_bytes(self) -> int:
        return sum(self.layer_bytes) + self.embed_bytes + self.head_bytes

    def kv_bytes(self, n_layers: int, context: int) -> int:
        return int(self.kv_bytes_per_token_layer * n_layers * context)


def _kv_per_token_layer(n_kv_heads: int, head_dim: int, kv_dtype: str) -> float:
    if kv_dtype not in KV_BYTES:
        raise PlanningError(f"unknown KV cache type '{kv_dtype}' (choose from {sorted(KV_BYTES)})")
    return 2 * n_kv_heads * head_dim * KV_BYTES[kv_dtype]


def from_catalog(arch_key: str, quant_key: str, kv_dtype: str = "f16") -> ModelProfile:
    arch = CATALOG.get(arch_key)
    if arch is None:
        raise PlanningError(f"unknown model '{arch_key}' (catalog: {', '.join(sorted(CATALOG))})")
    quant = QUANTS.get(quant_key)
    if quant is None:
        raise PlanningError(f"unknown quantization '{quant_key}' (choose from {', '.join(QUANTS)})")

    embed_params = arch.vocab * arch.hidden
    head_params = 0 if arch.tied_embeddings else embed_params
    per_layer_params = (arch.n_params - embed_params - head_params) / arch.n_layers
    layer = int(per_layer_params * quant.layer_bits / 8)
    embed = int(embed_params * quant.embed_bits / 8)
    # Tied models still need the embedding matrix on the output GPU for the logits.
    head = int((head_params or embed_params) * quant.head_bits / 8)
    return ModelProfile(
        name=f"{arch.display} ({quant.key})",
        source=f"catalog:{arch.key}/{quant.key}",
        n_layers=arch.n_layers,
        layer_bytes=(layer,) * arch.n_layers,
        embed_bytes=embed,
        head_bytes=head,
        kv_bytes_per_token_layer=_kv_per_token_layer(arch.n_kv_heads, arch.head_dim, kv_dtype),
        hf_repo=arch.awq_repo if quant.key == "awq-int4" and arch.awq_repo else arch.hf_repo,
        quant=quant.key,
    )


_BLOCK = re.compile(r"^blk\.(\d+)\.")


def from_gguf_file(gguf: GgufFile, kv_dtype: str = "f16") -> ModelProfile:
    n_layers = gguf.arch_value("block_count")
    hidden = gguf.arch_value("embedding_length")
    n_heads = gguf.arch_value("attention.head_count")
    if not isinstance(n_layers, int) or not isinstance(hidden, int) or not n_heads:
        raise PlanningError(
            f"{gguf.path.name}: missing {gguf.architecture}.block_count/embedding_length"
        )
    n_kv_heads = gguf.arch_value("attention.head_count_kv", n_heads)
    if isinstance(n_heads, list) or isinstance(n_kv_heads, list):
        raise PlanningError("per-layer head counts are not supported yet")
    head_dim = gguf.arch_value("attention.key_length", hidden // int(n_heads))

    layers = [0] * n_layers
    embed = head = 0
    unknown = 0
    for t in gguf.tensors:
        if t.nbytes is None:
            unknown += 1
            continue
        m = _BLOCK.match(t.name)
        if m and int(m.group(1)) < n_layers:
            layers[int(m.group(1))] += t.nbytes
        elif t.name.startswith("token_embd"):
            embed += t.nbytes
        elif t.name.startswith("output"):
            head += t.nbytes
    if unknown:
        # Fall back to spreading the file size evenly when the reader lacks a type.
        per_layer = max(0, gguf.file_size - embed - head) // n_layers
        layers = [per_layer] * n_layers
    if head == 0 or not any(t.name == "output.weight" for t in gguf.tensors):
        head += embed  # tied embeddings: logits reuse token_embd on the output GPU

    return ModelProfile(
        name=str(gguf.metadata.get("general.name", gguf.path.stem)),
        source=f"gguf:{gguf.path}",
        n_layers=n_layers,
        layer_bytes=tuple(layers),
        embed_bytes=embed,
        head_bytes=head,
        kv_bytes_per_token_layer=_kv_per_token_layer(int(n_kv_heads), int(head_dim), kv_dtype),
        quant=None,
    )


def from_gguf(path: str | Path, kv_dtype: str = "f16") -> ModelProfile:
    return from_gguf_file(read_gguf(path), kv_dtype)

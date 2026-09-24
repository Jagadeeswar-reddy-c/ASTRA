"""Tensor parallelism: every GPU works on every layer at once (ADR-0016).

Pipeline (layer) splitting adds memory but not single-stream speed: the stages run one
after another. Tensor splitting divides every weight matrix across the GPUs, so each
token reads 1/k of the weights per GPU in parallel, at the price of two all-reduces of
the hidden state per layer. How much that costs depends on the link between the GPUs:

====================  ===========================================  ==================
Link                  How the all-reduce travels                   Cost (2 GPUs)
====================  ===========================================  ==================
``host``              GPU -> CPU memory -> GPU (GeForce default)   45 us
``p2p``               GPU -> GPU over PCIe (switch or P2P driver)  25 us (estimate)
``nvlink``            NVLink bridge (RTX 3090, 2080 Ti, ...)       12 us
====================  ===========================================  ==================

Calibration (docs/05-testing/reports/tensor-parallel-study.md). ``host`` is fitted to
single-stream data: vLLM TP=2 on 2 x RTX 3090 over PCIe 4.0 x8 without NVLink measured
+16-35 %, and a 16 KB GPU -> host -> GPU round trip measured 14 us on the dev PC (an
all-reduce needs that plus a host-side reduce and synchronisation). ``p2p`` and ``nvlink``
are estimates from link latency, to be confirmed on hardware (TC-HW-14); published vLLM
throughput runs show NVLink +48 % over PCIe at TP=2. Across machines an all-reduce costs
~0.9 ms (measured over llama.cpp RPC), so tensor splitting stays inside one host.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

from astra.errors import PlanningError
from astra.planner.models import CATALOG, ModelProfile
from astra.planner.split import (
    SPEED_FILL_LIMIT,
    DeviceBudget,
    Placement,
    Plan,
    draft_bytes,
)

ALLREDUCE_S = {"host": 45e-6, "p2p": 25e-6, "nvlink": 12e-6}
INTERCONNECTS = tuple(ALLREDUCE_S)
TENSOR_KV_TYPES = ("f16",)  # llama.cpp tensor mode rejects quantized KV caches
MAX_CONTEXT_SEARCH = 1 << 20


def allreduce_s(link: str, k: int) -> float:
    """Seconds per all-reduce among ``k`` GPUs (a ring needs more steps with more GPUs)."""
    if link not in ALLREDUCE_S:
        raise PlanningError(f"unknown interconnect '{link}' (choose from {', '.join(ALLREDUCE_S)})")
    return ALLREDUCE_S[link] * (1 + 0.5 * max(0, k - 2))


def classify_links(
    indices: Sequence[int],
    gpu_links: Mapping[tuple[int, int], str],
    p2p: Mapping[tuple[int, int], str] | None = None,
) -> str:
    """'nvlink' if every pair has NVLink, 'p2p' if every pair reports P2P OK, else 'host'.

    ``gpu_links`` is ``nvidia-smi topo -m``, ``p2p`` is ``nvidia-smi topo -p2p r``.
    """
    pairs = [(a, b) for i, a in enumerate(indices) for b in indices[i + 1 :]]
    if not pairs:
        return "host"

    def link(a: int, b: int) -> str:
        return gpu_links.get((min(a, b), max(a, b)), "")

    if all(link(a, b).startswith("NV") for a, b in pairs):
        return "nvlink"
    if p2p and all(p2p.get((min(a, b), max(a, b))) == "OK" for a, b in pairs):
        return "p2p"
    return "host"


def _shares(caps: list[float], weights: list[float]) -> tuple[list[float], bool]:
    """Water-filling: shares proportional to ``weights`` (bandwidth), none above its cap."""
    n = len(caps)
    share = [0.0] * n
    free = set(range(n))
    remaining = 1.0
    while free:
        total = sum(weights[i] for i in free) or 1.0
        over = [i for i in free if remaining * weights[i] / total > caps[i]]
        if not over:
            for i in free:
                share[i] = remaining * weights[i] / total
            return share, True
        for i in over:
            share[i] = caps[i]
            remaining -= caps[i]
            free.discard(i)
        if remaining <= 0:
            break
    if remaining > 1e-9:
        # Does not fit: spread by capacity so the shortfall shows on every GPU.
        cap_total = sum(caps) or 1.0
        return [c / cap_total for c in caps], False
    return share, True


def plan_tensor(
    model: ModelProfile,
    budgets: list[DeviceBudget],
    engine: str,
    context: int,
    gpu_memory_utilization: float,
    interconnect: str = "host",
    warnings: list[str] | None = None,
    draft: ModelProfile | None = None,
) -> Plan:
    """Tensor-parallel plan over exactly these local GPUs."""
    notes = list(warnings or [])
    k = len(budgets)
    if k < 2:
        raise PlanningError("tensor parallelism needs at least 2 GPUs")
    if any(b.is_remote for b in budgets):
        raise PlanningError(
            "tensor parallelism is limited to GPUs in one machine: an all-reduce over the "
            "network costs ~1 ms, twice per layer per token"
        )
    if engine == "llamacpp" and model.kv_dtype not in TENSOR_KV_TYPES:
        raise PlanningError("llama.cpp tensor mode needs an f16 KV cache (drop --kv-type)")
    arch = CATALOG.get(model.arch_key or "")
    if engine == "vllm":
        if len({b.total_bytes for b in budgets}) > 1:
            notes.append("vLLM splits evenly: the smallest GPU limits every GPU's share")
        if arch and (arch.n_heads % k or (arch.n_kv_heads % k and k % arch.n_kv_heads)):
            raise PlanningError(
                f"{arch.display}: {arch.n_heads} attention heads / {arch.n_kv_heads} KV heads "
                f"cannot be split {k} ways"
            )
        draft = None
    if draft is not None and engine != "llamacpp":
        draft = None

    # Per-token bytes that are split: blocks + output head (+ embeddings on vLLM).
    split_weights = sum(model.layer_bytes) + model.head_bytes
    if engine == "vllm":
        split_weights += model.embed_bytes
    extra = [draft_bytes(draft, context) if draft is not None and i == 0 else 0 for i in range(k)]

    def layout(ctx: int) -> tuple[list[float], bool]:
        per_share = split_weights + model.kv_bytes(model.n_layers, ctx)
        usable = [b.usable_bytes - extra[i] for i, b in enumerate(budgets)]
        caps = [max(0.0, u / per_share) for u in usable]
        if engine == "vllm":
            even = min(caps) * k >= 1.0
            return [1.0 / k] * k, even
        return _shares(caps, [b.bandwidth_gbps or 1.0 for b in budgets])

    shares, fits = layout(context)
    kv_total = model.kv_bytes(model.n_layers, context)
    placements = tuple(
        Placement(
            device=b,
            first_layer=0,
            n_layers=model.n_layers,
            weight_bytes=int(shares[i] * split_weights),
            kv_bytes=int(shares[i] * kv_total),
            fixed_bytes=extra[i],
            draft_bytes=extra[i],
        )
        for i, b in enumerate(budgets)
    )
    fits = fits and all(p.headroom_bytes >= 0 for p in placements)

    lo, hi = 0, MAX_CONTEXT_SEARCH
    while lo < hi:
        mid = (lo + hi + 1) // 2
        if layout(mid)[1]:
            lo = mid
        else:
            hi = mid - 1
    if not fits:
        notes.append(
            f"does not fit at context {context}; largest context that fits is {lo}"
            if lo
            else "model weights alone exceed these GPUs"
        )
    worst = max(p.model_bytes / p.device.usable_bytes for p in placements)
    if fits and worst > SPEED_FILL_LIMIT:
        notes.append(f"a GPU is {worst:.0%} full; little room for longer prompts")
    if interconnect == "host":
        notes.append(
            "tensor split over PCIe via host memory (no P2P/NVLink): each layer waits for two "
            "all-reduces; an NVLink bridge or a P2P-capable PCIe switch makes it faster"
        )
    if engine == "llamacpp":
        notes.append("llama.cpp tensor mode is experimental: `astra auto` measures it first")
    return Plan(
        model,
        engine,
        context,
        gpu_memory_utilization,
        placements,
        lo,
        tuple(notes),
        objective="speed",
        draft=draft,
        draft_stage=0 if draft is not None else None,
        split_mode="tensor",
        allreduce_s=allreduce_s(interconnect, k),
        interconnect=interconnect,
    )


def detect_interconnect(budgets: Sequence[DeviceBudget]) -> str:
    """Link class of the local GPUs, from ``nvidia-smi topo -m`` and ``topo -p2p r``."""
    from astra.hardware import nvtopo
    from astra.hardware.runner import SubprocessRunner

    local = [b.index for b in budgets if not b.is_remote]
    if len(local) < 2:
        return "host"
    runner = SubprocessRunner()
    return classify_links(local, nvtopo.query(runner), nvtopo.query_p2p(runner))


def better(a: Plan, b: Plan, margin: float = 0.05) -> Plan:
    """The plan that fits and is faster by more than ``margin``; ``a`` wins ties."""
    if not b.fits:
        return a
    if not a.fits:
        return b
    ta, tb = a.est_decode_tokens_per_s or 0.0, b.est_decode_tokens_per_s or 0.0
    return b if tb > ta * (1 + margin) else a

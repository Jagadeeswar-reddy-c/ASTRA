"""Pipeline-parallel placement of a model across heterogeneous GPUs (ADR-0003, ADR-0010).

Each GPU's *usable* memory (capacity minus a CUDA/compute reserve) must hold its
contiguous run of transformer blocks, the KV cache for those blocks, and any fixed
tensors the engine pins to that stage (embeddings on the first stage, output head
on the last). A dynamic programme over (GPU, first block) picks the split for one
of two objectives:

* ``balanced`` — minimise the worst per-GPU memory utilisation. With negligible
  fixed costs this is the concept document's VRAM ratio (6:8 ≈ 43 %:57 %).
* ``speed`` — minimise single-stream decode time, Σ bytes_i / bandwidth_i, while
  keeping every GPU ≤ 92 % full, then spend up to 3 % of that time on a more even
  split. Decoding is memory-bandwidth bound, so slow cards (a GT 1030 reads
  48 GB/s, an RTX 3060 360 GB/s) only receive what must spill onto them.

:func:`select_gpus` also tries every subset of the eligible GPUs and keeps the
best one that fits — adding a slow card to a pipeline can make it slower.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from functools import cache
from itertools import combinations, pairwise
from typing import Any

from astra.errors import PlanningError
from astra.hardware.compat import ENGINE_MIN_CC, capabilities
from astra.hardware.models import GpuInfo
from astra.planner.models import ModelProfile
from astra.units import MIB, fmt_bytes

ENGINES = ("llamacpp", "vllm")
OBJECTIVES = ("speed", "balanced")
MAX_CONTEXT_SEARCH = 1 << 20
BANDWIDTH_EFFICIENCY = 0.75  # achievable share of peak DRAM bandwidth during decode
HOP_LATENCY_S = 150e-6  # per stage boundary per token: activation copy via host over PCIe
NETWORK_HOP_S = 2e-3  # per stage boundary crossing machines (llama.cpp RPC round trips, LAN)
MAX_AUTO_GPUS = 8
SPEED_TIE = 0.05  # GPU sets whose speed estimates differ by less are treated as equal
SPEED_SLACK = 0.03  # decode-time slack traded for a more even memory split
SPEED_FILL_LIMIT = 0.92  # speed plans keep every GPU at or below this utilisation
# Speculative decoding (FT-SPEC-01, RTX 3060 Ti, Qwen2.5-7B + 0.5B draft): +21-38 % when
# the target decodes at ~20 tok/s, -5 to -15 % at ~74 tok/s. Only suggested below this.
SPEC_MAX_TPS = 30.0
DRAFT_OVERHEAD_BYTES = 64 * MIB  # draft compute buffers on top of its weights and KV


@dataclass(frozen=True)
class DeviceBudget:
    index: int
    uuid: str
    name: str
    bus_id: str
    architecture: str | None
    total_bytes: int
    capacity_bytes: int  # what the engine may use on this GPU
    reserve_bytes: int  # kept free for CUDA context + compute buffers
    baseline_used_bytes: int  # memory already used by other processes at planning time
    bandwidth_gbps: float | None = None  # peak DRAM bandwidth (gpu_specs); None if unknown
    compute_capability: float | None = None
    display_active: bool | None = None
    node: str = "local"  # fabric node that hosts this GPU (ADR-0011)
    rpc_endpoint: str | None = None  # host:port of the node's llama.cpp rpc-server; None = local

    @property
    def is_remote(self) -> bool:
        return self.rpc_endpoint is not None

    @property
    def usable_bytes(self) -> int:
        return self.capacity_bytes - self.reserve_bytes

    def supports(self, engine: str) -> bool:
        if self.is_remote and engine != "llamacpp":
            return False  # only llama.cpp can reach GPUs on other machines (RPC backend)
        cc = self.compute_capability
        return cc is None or cc >= ENGINE_MIN_CC[engine]


@dataclass(frozen=True)
class Placement:
    device: DeviceBudget
    first_layer: int
    n_layers: int
    weight_bytes: int
    kv_bytes: int
    fixed_bytes: int  # embeddings / output head (and a draft model) pinned to this stage
    draft_bytes: int = 0  # part of fixed_bytes: a speculative-decoding draft on this GPU

    @property
    def required_bytes(self) -> int:
        return self.weight_bytes + self.kv_bytes + self.fixed_bytes + self.device.reserve_bytes

    @property
    def headroom_bytes(self) -> int:
        return self.device.capacity_bytes - self.required_bytes

    @property
    def model_bytes(self) -> int:
        return self.weight_bytes + self.kv_bytes + self.fixed_bytes


@dataclass(frozen=True)
class Candidate:
    """One GPU subset evaluated by :func:`select_gpus`."""

    gpu_indices: tuple[int, ...]
    names: tuple[str, ...]
    fits: bool
    est_decode_tokens_per_s: float | None
    worst_utilisation: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "gpu_indices": list(self.gpu_indices),
            "names": list(self.names),
            "fits": self.fits,
            "est_decode_tokens_per_s": self.est_decode_tokens_per_s,
            "worst_utilisation": round(self.worst_utilisation, 4),
        }


@dataclass(frozen=True)
class Plan:
    model: ModelProfile
    engine: str
    context: int
    gpu_memory_utilization: float
    placements: tuple[Placement, ...]
    max_context: int
    warnings: tuple[str, ...] = field(default_factory=tuple)
    objective: str = "balanced"
    alternatives: tuple[Candidate, ...] = field(default_factory=tuple)
    network_hop_s: float = NETWORK_HOP_S
    draft: ModelProfile | None = None  # speculative-decoding draft model
    draft_stage: int | None = None  # index into placements of the GPU holding the draft
    split_mode: str = "layer"  # "layer" (pipeline) or "tensor" (ADR-0016)
    allreduce_s: float = 0.0  # tensor mode: seconds per all-reduce between the GPUs
    interconnect: str | None = None  # tensor mode: "host", "p2p" or "nvlink"

    @property
    def fits(self) -> bool:
        return all(p.headroom_bytes >= 0 for p in self.placements)

    @property
    def layer_counts(self) -> tuple[int, ...]:
        return tuple(p.n_layers for p in self.placements)

    @property
    def vram_ratio(self) -> tuple[float, ...]:
        """The concept document's raw ratio: VRAM_i / VRAM_total."""
        total = sum(p.device.total_bytes for p in self.placements)
        return tuple(p.device.total_bytes / total for p in self.placements)

    @property
    def planned_share(self) -> tuple[float, ...]:
        """Fraction of the model's resident bytes placed on each GPU."""
        total = sum(p.model_bytes for p in self.placements) or 1
        return tuple(p.model_bytes / total for p in self.placements)

    @property
    def est_decode_tokens_per_s(self) -> float | None:
        if self.split_mode == "tensor":
            return estimate_tensor_tps(self.placements, self.model.n_layers, self.allreduce_s)
        return estimate_decode_tps(self.placements, self.network_hop_s)

    @property
    def tensor_split(self) -> tuple[float, ...]:
        """llama.cpp --tensor-split values: layers per GPU, or weight shares in tensor mode."""
        if self.split_mode == "tensor":
            total = sum(p.weight_bytes for p in self.placements) or 1
            return tuple(p.weight_bytes / total for p in self.placements)
        return tuple(float(n) for n in self.layer_counts)

    def to_dict(self) -> dict[str, Any]:
        return {
            "model": {
                "name": self.model.name,
                "source": self.model.source,
                "quant": self.model.quant,
                "hf_repo": self.model.hf_repo,
                "n_layers": self.model.n_layers,
                "weights_bytes": self.model.weights_bytes,
            },
            "engine": self.engine,
            "split_mode": self.split_mode,
            "interconnect": self.interconnect,
            "objective": self.objective,
            "est_decode_tokens_per_s": self.est_decode_tokens_per_s,
            "context": self.context,
            "max_context": self.max_context,
            "gpu_memory_utilization": self.gpu_memory_utilization,
            "fits": self.fits,
            "layer_counts": list(self.layer_counts),
            "vram_ratio": [round(r, 4) for r in self.vram_ratio],
            "planned_share": [round(s, 4) for s in self.planned_share],
            "warnings": list(self.warnings),
            "devices": [
                {
                    "index": p.device.index,
                    "uuid": p.device.uuid,
                    "name": p.device.name,
                    "bus_id": p.device.bus_id,
                    "total_bytes": p.device.total_bytes,
                    "capacity_bytes": p.device.capacity_bytes,
                    "reserve_bytes": p.device.reserve_bytes,
                    "baseline_used_bytes": p.device.baseline_used_bytes,
                    "first_layer": p.first_layer,
                    "n_layers": p.n_layers,
                    "weight_bytes": p.weight_bytes,
                    "kv_bytes": p.kv_bytes,
                    "fixed_bytes": p.fixed_bytes,
                    "draft_bytes": p.draft_bytes,
                    "required_bytes": p.required_bytes,
                    "headroom_bytes": p.headroom_bytes,
                    "bandwidth_gbps": p.device.bandwidth_gbps,
                    "compute_capability": p.device.compute_capability,
                    "node": p.device.node,
                    "rpc_endpoint": p.device.rpc_endpoint,
                }
                for p in self.placements
            ],
            "alternatives": [a.to_dict() for a in self.alternatives],
            "draft": None
            if self.draft is None or self.draft_stage is None
            else {
                "name": self.draft.name,
                "source": self.draft.source,
                "arch_key": self.draft.arch_key,
                "quant": self.draft.quant,
                "bytes": draft_bytes(self.draft, self.context),
                "stage": self.draft_stage,
                "uuid": self.placements[self.draft_stage].device.uuid,
            },
        }


def estimate_decode_tps(
    placements: Sequence[Placement], network_hop_s: float = NETWORK_HOP_S
) -> float | None:
    """Rough single-stream decode ceiling: every token reads each stage's weights and,
    on average, half its KV cache at the GPU's DRAM bandwidth; plus one PCIe hop per
    stage boundary. None if any bandwidth is unknown. Batched serving is faster."""
    seconds = 0.0
    for p in placements:
        bw = p.device.bandwidth_gbps
        if not bw:
            return None
        # The draft is read by its own (cheap) passes, not per target token.
        seconds += (p.weight_bytes + p.fixed_bytes - p.draft_bytes + p.kv_bytes / 2) / (
            bw * 1e9 * BANDWIDTH_EFFICIENCY
        )
    for a, b in pairwise(placements):
        if a.device.node == b.device.node:
            seconds += HOP_LATENCY_S
        elif a.device.is_remote and b.device.is_remote:
            seconds += 2 * network_hop_s  # llama.cpp RPC relays remote-to-remote via the head
        else:
            seconds += network_hop_s
    # The head node runs the graph: it sends the input to the first stage and reads the
    # logits back from the last one, so a remote first or last stage costs a hop too.
    if placements and placements[0].device.is_remote:
        seconds += network_hop_s
    if placements and placements[-1].device.is_remote:
        seconds += network_hop_s
    return 1.0 / seconds if seconds > 0 else None


def estimate_tensor_tps(
    placements: Sequence[Placement], n_layers: int, allreduce_s: float
) -> float | None:
    """Tensor split: all GPUs read their share at once (the slowest sets the pace), then
    two all-reduces of the hidden state per layer (ADR-0016)."""
    slowest = 0.0
    for p in placements:
        bw = p.device.bandwidth_gbps
        if not bw:
            return None
        read = p.weight_bytes + p.fixed_bytes - p.draft_bytes + p.kv_bytes / 2
        slowest = max(slowest, read / (bw * 1e9 * BANDWIDTH_EFFICIENCY))
    seconds = slowest + 2 * n_layers * allreduce_s
    return 1.0 / seconds if seconds > 0 else None


def budgets_from_gpus(
    gpus: list[GpuInfo] | tuple[GpuInfo, ...],
    gpu_memory_utilization: float,
    reserve_mib: int,
) -> tuple[list[DeviceBudget], list[str]]:
    """Per-GPU budgets in pipeline order (PCI bus order, matching CUDA_DEVICE_ORDER=PCI_BUS_ID)."""
    warnings: list[str] = []
    budgets = []
    for gpu in sorted(gpus, key=lambda g: g.bus_id):
        caps = capabilities(gpu)
        if gpu.memory_total_bytes is None:
            raise PlanningError(f"{gpu.name} ({gpu.uuid}) did not report total memory")
        total = gpu.memory_total_bytes
        used = gpu.memory_used_bytes or 0
        capacity = int(total * gpu_memory_utilization)
        free = total - used
        if free < capacity:
            warnings.append(
                f"GPU {gpu.index} {gpu.name}: {fmt_bytes(used)} already in use by other processes; "
                f"budget reduced from {fmt_bytes(capacity)} to {fmt_bytes(free)}"
            )
            capacity = free
        budgets.append(
            DeviceBudget(
                index=gpu.index,
                uuid=gpu.uuid,
                name=gpu.name,
                bus_id=gpu.bus_id,
                architecture=caps.architecture,
                total_bytes=total,
                capacity_bytes=capacity,
                reserve_bytes=reserve_mib * MIB,
                baseline_used_bytes=used,
                bandwidth_gbps=caps.mem_bandwidth_gbps,
                compute_capability=caps.compute_capability,
                display_active=gpu.display_active,
            )
        )
    return budgets, warnings


def draft_bytes(draft: ModelProfile, context: int) -> int:
    """GPU memory of a speculative-decoding draft: blocks and output head (llama.cpp keeps
    token embeddings in host RAM), its own KV cache, and buffers. Measured: Qwen2.5-0.5B
    Q8_0 at 8k context adds 645 MiB (FT-SPEC-01); this formula gives ~650 MiB."""
    return (
        sum(draft.layer_bytes)
        + draft.head_bytes
        + draft.kv_bytes(draft.n_layers, context)
        + DRAFT_OVERHEAD_BYTES
    )


def draft_stage(budgets: Sequence[DeviceBudget]) -> int | None:
    """The draft runs on the first local GPU of the pipeline (never over the network)."""
    return next((i for i, b in enumerate(budgets) if not b.is_remote), None)


def _fixed_costs(
    model: ModelProfile,
    engine: str,
    budgets: Sequence[DeviceBudget],
    draft: ModelProfile | None = None,
    context: int = 0,
) -> list[int]:
    fixed = [0] * len(budgets)
    fixed[-1] += model.head_bytes
    if engine == "vllm":
        fixed[0] += model.embed_bytes  # llama.cpp keeps token embeddings in host RAM
    stage = draft_stage(budgets) if draft is not None and engine == "llamacpp" else None
    if draft is not None and stage is not None:
        fixed[stage] += draft_bytes(draft, context)
    return fixed


def _partition(
    model: ModelProfile,
    budgets: list[DeviceBudget],
    fixed: list[int],
    context: int,
    objective: str = "balanced",
) -> tuple[float, list[int]]:
    """Contiguous split. Returns (worst memory utilisation, block counts per GPU).

    ``balanced`` minimises the worst utilisation. ``speed`` builds the Pareto front
    of (decode time, worst utilisation) over splits that keep every GPU at or below
    ``SPEED_FILL_LIMIT``, then takes the most even split whose time is within
    ``SPEED_SLACK`` of the fastest — so near-equal cards share the load while a slow
    card only receives what must spill onto it. Falls back to ``balanced`` when no
    split respects the fill limit or a bandwidth is unknown.
    """
    n, k = model.n_layers, len(budgets)
    kv_layer = model.kv_bytes_per_token_layer * context
    prefix = [0.0]
    for b in model.layer_bytes:
        prefix.append(prefix[-1] + b + kv_layer)

    def util(dev: int, start: int, end: int) -> float:
        usable = budgets[dev].usable_bytes
        demand = fixed[dev] + prefix[end] - prefix[start]
        return demand / usable if usable > 0 else float("inf")

    @cache
    def balanced(dev: int, start: int) -> tuple[float, tuple[int, ...]]:
        if dev == k - 1:
            return util(dev, start, n), (n - start,)
        result: tuple[float, tuple[int, ...]] = (float("inf"), ())
        for end in range(start + 1, n - (k - dev - 1) + 1):
            tail_cost, tail = balanced(dev + 1, end)
            worst = max(util(dev, start, end), tail_cost)
            if worst < result[0]:
                result = (worst, (end - start, *tail))
        return result

    worst, counts = balanced(0, 0)
    if objective == "balanced" or any(not b.bandwidth_gbps for b in budgets):
        return worst, list(counts)

    def seconds(dev: int, start: int, end: int) -> float:
        bw = budgets[dev].bandwidth_gbps or 1.0
        return (fixed[dev] + prefix[end] - prefix[start]) / (bw * 1e9)

    Option = tuple[float, float, tuple[int, ...]]  # (time, worst utilisation, counts)

    def pareto(options: list[Option]) -> tuple[Option, ...]:
        options.sort(key=lambda o: (o[0], o[1]))
        front: list[Option] = []
        for o in options:
            if not front or o[1] < front[-1][1]:
                front.append(o)
        return tuple(front)

    @cache
    def fronts(dev: int, start: int) -> tuple[Option, ...]:
        if dev == k - 1:
            u = util(dev, start, n)
            return ((seconds(dev, start, n), u, (n - start,)),) if u <= SPEED_FILL_LIMIT else ()
        options: list[Option] = []
        for end in range(start + 1, n - (k - dev - 1) + 1):
            u = util(dev, start, end)
            if u > SPEED_FILL_LIMIT:
                break  # later ends only add blocks to this GPU
            head = seconds(dev, start, end)
            options += [
                (head + t, max(u, w), (end - start, *c)) for t, w, c in fronts(dev + 1, end)
            ]
        return pareto(options)

    front = fronts(0, 0)
    if not front:
        return worst, list(counts)
    fastest = front[0][0]
    _, speed_worst, speed_counts = min(
        (o for o in front if o[0] <= fastest * (1 + SPEED_SLACK)), key=lambda o: o[1]
    )
    return speed_worst, list(speed_counts)


def _max_context(model: ModelProfile, budgets: list[DeviceBudget], fixed: list[int]) -> int:
    if _partition(model, budgets, fixed, 1)[0] > 1.0:
        return 0
    lo, hi = 1, MAX_CONTEXT_SEARCH
    while lo < hi:
        mid = (lo + hi + 1) // 2
        if _partition(model, budgets, fixed, mid)[0] <= 1.0:
            lo = mid
        else:
            hi = mid - 1
    return lo


def _placements(
    model: ModelProfile,
    budgets: list[DeviceBudget],
    fixed: list[int],
    counts: list[int],
    context: int,
    draft: ModelProfile | None = None,
) -> tuple[Placement, ...]:
    stage = draft_stage(budgets) if draft is not None else None
    out, start = [], 0
    for dev, count in enumerate(counts):
        out.append(
            Placement(
                device=budgets[dev],
                first_layer=start,
                n_layers=count,
                weight_bytes=sum(model.layer_bytes[start : start + count]),
                kv_bytes=model.kv_bytes(count, context),
                fixed_bytes=fixed[dev],
                draft_bytes=draft_bytes(draft, context) if draft and dev == stage else 0,
            )
        )
        start += count
    return tuple(out)


def plan(
    model: ModelProfile,
    budgets: list[DeviceBudget],
    engine: str,
    context: int,
    gpu_memory_utilization: float,
    warnings: list[str] | None = None,
    objective: str = "balanced",
    hint_single_gpu: bool = True,
    network_hop_s: float = NETWORK_HOP_S,
    draft: ModelProfile | None = None,
) -> Plan:
    """Plan on exactly these GPUs, in this order. ``draft`` adds a speculative-decoding
    draft model on the first local GPU (llama.cpp only)."""
    if engine not in ENGINES:
        raise PlanningError(f"unknown engine '{engine}' (choose from {', '.join(ENGINES)})")
    if objective not in OBJECTIVES:
        raise PlanningError(
            f"unknown objective '{objective}' (choose from {', '.join(OBJECTIVES)})"
        )
    if not budgets:
        raise PlanningError("no GPUs available to plan on")
    if len(budgets) > model.n_layers:
        raise PlanningError(f"{len(budgets)} GPUs but the model only has {model.n_layers} layers")
    if context < 1:
        raise PlanningError("context must be >= 1 token")
    for b in budgets:
        if not b.supports(engine):
            if b.is_remote:
                raise PlanningError(
                    f"GPU {b.index} {b.name} is on node {b.node}; only llamacpp can use remote GPUs"
                )
            raise PlanningError(
                f"GPU {b.index} {b.name} (compute capability {b.compute_capability}) is not "
                f"supported by {engine}, which needs >= {ENGINE_MIN_CC[engine]}"
            )
        if b.usable_bytes <= 0:
            raise PlanningError(
                f"GPU {b.index} {b.name}: reserve ({fmt_bytes(b.reserve_bytes)}) exceeds "
                f"its budget ({fmt_bytes(b.capacity_bytes)})"
            )

    notes = list(warnings or [])
    if draft is not None and (engine != "llamacpp" or draft_stage(budgets) is None):
        why = "llama.cpp only" if engine != "llamacpp" else "no local GPU in the pipeline"
        notes.append(f"speculative decoding skipped: {why}")
        draft = None
    fixed = _fixed_costs(model, engine, budgets, draft, context)
    _, counts = _partition(model, budgets, fixed, context, objective)
    placements = _placements(model, budgets, fixed, counts, context, draft)
    max_ctx = _max_context(model, budgets, fixed)
    fits = all(p.headroom_bytes >= 0 for p in placements)

    if not fits:
        if max_ctx > 0:
            notes.append(
                f"does not fit at context {context}; largest context that fits is {max_ctx}"
            )
        else:
            notes.append("model weights alone exceed the pool; choose a smaller quantization")
    if hint_single_gpu and len(budgets) > 1 and fits:
        for b in budgets:
            if _partition(model, [b], _fixed_costs(model, engine, [b]), context)[0] <= 1.0:
                notes.append(
                    f"fits on GPU {b.index} {b.name} alone; a single-GPU deployment avoids "
                    "inter-GPU activation transfers over the shared uplink"
                )
                break
    for b in budgets:
        if b.display_active:
            notes.append(f"GPU {b.index} {b.name} drives a display; its free memory can change")
        if b.bandwidth_gbps is None:
            notes.append(f"GPU {b.index} {b.name}: memory bandwidth unknown — no speed estimate")
    if engine == "vllm" and any((b.compute_capability or 0) < 8.0 for b in budgets):
        notes.append("pre-Ampere GPU in the pool: use --dtype float16 (no bfloat16 support)")
    tps = estimate_decode_tps(placements, network_hop_s)
    if draft is None and engine == "llamacpp" and fits and tps and tps <= SPEC_MAX_TPS:
        notes.append(
            f"~{tps:.0f} tok/s is slow enough for speculative decoding to help "
            "(+20-40 % measured): add --draft auto"
        )
    return Plan(
        model,
        engine,
        context,
        gpu_memory_utilization,
        placements,
        max_ctx,
        tuple(notes),
        objective=objective,
        network_hop_s=network_hop_s,
        draft=draft,
        draft_stage=draft_stage(budgets) if draft is not None else None,
    )


def _excluded(b: DeviceBudget, exclude: Sequence[str]) -> bool:
    return any(x and (x == b.uuid or x.lower() in b.name.lower()) for x in exclude)


def select_gpus(
    model: ModelProfile,
    budgets: list[DeviceBudget],
    engine: str,
    context: int,
    gpu_memory_utilization: float,
    warnings: list[str] | None = None,
    objective: str = "speed",
    exclude: Sequence[str] = (),
    network_hop_s: float = NETWORK_HOP_S,
    draft: ModelProfile | None = None,
) -> Plan:
    """Try every subset of eligible GPUs (kept in PCI order) and plan on the best one.

    Ranking: sets that fit; then sets that fit with margin (every GPU <= 92 %);
    then the highest estimated decode speed (``speed`` — sets within 5 % of the
    fastest count as equal and the one with fewer GPUs wins) or the lowest
    worst-case memory pressure (``balanced``). When nothing fits, the plan covers
    all eligible GPUs so the shortfall is visible.
    """
    if engine not in ENGINES:
        raise PlanningError(f"unknown engine '{engine}' (choose from {', '.join(ENGINES)})")
    notes = list(warnings or [])
    eligible = []
    for b in budgets:
        if _excluded(b, exclude):
            notes.append(f"GPU {b.index} {b.name} skipped: excluded by config")
        elif b.is_remote and engine != "llamacpp":
            notes.append(f"GPU {b.index} {b.name} on node {b.node} skipped: {engine} is local-only")
        elif not b.supports(engine):
            notes.append(
                f"GPU {b.index} {b.name} skipped: compute capability {b.compute_capability} "
                f"< {ENGINE_MIN_CC[engine]} required by {engine}"
            )
        elif b.usable_bytes <= 0:
            notes.append(f"GPU {b.index} {b.name} skipped: no memory left after the reserve")
        else:
            eligible.append(b)
    if not eligible:
        raise PlanningError(f"no GPU is eligible for {engine}")
    if len(eligible) > MAX_AUTO_GPUS:
        order = {id(b): i for i, b in enumerate(eligible)}  # keep caller's (local-first) order
        eligible = sorted(eligible, key=lambda b: b.usable_bytes, reverse=True)[:MAX_AUTO_GPUS]
        eligible.sort(key=lambda b: order[id(b)])

    scored: list[tuple[tuple[float, ...], list[DeviceBudget], Candidate]] = []
    for size in range(1, min(len(eligible), model.n_layers) + 1):
        for subset in combinations(eligible, size):
            chosen = list(subset)
            fixed = _fixed_costs(model, engine, chosen, draft, context)
            worst, counts = _partition(model, chosen, fixed, context, objective)
            tps = estimate_decode_tps(
                _placements(model, chosen, fixed, counts, context, draft), network_hop_s
            )
            fits = worst <= 1.0
            comfortable = worst <= SPEED_FILL_LIMIT
            rank = -(tps or 0.0) if objective == "speed" else worst
            key = (0.0 if fits else 1.0, 0.0 if comfortable else 1.0, rank, worst, float(size))
            cand = Candidate(
                tuple(b.index for b in chosen), tuple(b.name for b in chosen), fits, tps, worst
            )
            scored.append((key, chosen, cand))
    scored.sort(key=lambda s: s[0])

    if objective == "speed" and scored[0][2].fits and scored[0][2].est_decode_tokens_per_s:
        # Estimates closer than SPEED_TIE are within their own error: among sets in the
        # same (fits, comfortable) class, prefer fewer GPUs (fewer hops and failure
        # points), then the faster one.
        head = scored[0]
        top = head[2].est_decode_tokens_per_s or 0.0
        tied = [
            s
            for s in scored
            if s[0][:2] == head[0][:2]
            and (s[2].est_decode_tokens_per_s or 0.0) >= top * (1 - SPEED_TIE)
        ]
        winner = min(tied, key=lambda s: (len(s[1]), -(s[2].est_decode_tokens_per_s or 0.0)))
        scored.remove(winner)
        scored.insert(0, winner)

    best = scored[0][1]
    if not scored[0][2].fits:
        best = eligible
        notes.append("no GPU combination fits; showing the plan for all eligible GPUs")
    elif len(best) < len(eligible):
        left_out = ", ".join(f"{b.index} {b.name}" for b in eligible if b not in best)
        why = "slower or not needed" if objective == "speed" else "not needed"
        notes.append(
            f"auto-selected GPU(s) {', '.join(str(b.index) for b in best)}; "
            f"left out {left_out} ({why})"
        )
    result = plan(
        model,
        best,
        engine,
        context,
        gpu_memory_utilization,
        notes,
        objective,
        hint_single_gpu=False,
        network_hop_s=network_hop_s,
        draft=draft,
    )
    return Plan(
        result.model,
        result.engine,
        result.context,
        result.gpu_memory_utilization,
        result.placements,
        result.max_context,
        result.warnings,
        result.objective,
        tuple(c for _, _, c in scored[:6]),
        network_hop_s,
        result.draft,
        result.draft_stage,
    )

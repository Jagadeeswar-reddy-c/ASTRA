"""Turn a :class:`Plan` into a concrete engine launch (argv + environment).

GPUs are addressed by UUID with ``CUDA_DEVICE_ORDER=PCI_BUS_ID`` (ADR-0004) so
the pipeline order never depends on CUDA's default "fastest first" ordering or
on whether the host has its own display GPU.
"""

from __future__ import annotations

import shlex
from dataclasses import dataclass, field

from astra.errors import PlanningError
from astra.planner.split import Plan


@dataclass(frozen=True)
class LaunchSpec:
    argv: tuple[str, ...]
    env: dict[str, str]
    notes: tuple[str, ...] = field(default_factory=tuple)

    def shell(self, posix: bool = True) -> str:
        if posix:
            prefix = " ".join(f"{k}={shlex.quote(v)}" for k, v in self.env.items())
            return f"{prefix} {shlex.join(self.argv)}".strip()
        sets = "; ".join(f"$env:{k}='{v}'" for k, v in self.env.items())
        return f"{sets}; & {' '.join(self.argv)}"


def _gpu_env(plan: Plan) -> dict[str, str]:
    """Local GPUs only; remote GPUs are reached through llama.cpp's RPC backend."""
    return {
        "CUDA_DEVICE_ORDER": "PCI_BUS_ID",
        "CUDA_VISIBLE_DEVICES": ",".join(
            p.device.uuid for p in plan.placements if not p.device.is_remote
        ),
    }


def llamacpp_devices(plan: Plan) -> tuple[list[str], list[str]]:
    """(RPC endpoints, llama.cpp device names) in pipeline order.

    Local GPUs are CUDA0..n in CUDA_VISIBLE_DEVICES order; each remote GPU is the
    device its rpc-server exposes, named RPC[host:port].
    """
    endpoints: list[str] = []
    devices: list[str] = []
    local = 0
    for p in plan.placements:
        if p.device.rpc_endpoint:
            if p.device.rpc_endpoint not in endpoints:
                endpoints.append(p.device.rpc_endpoint)
            devices.append(f"RPC[{p.device.rpc_endpoint}]")
        else:
            devices.append(f"CUDA{local}")
            local += 1
    return endpoints, devices


def llamacpp(
    plan: Plan,
    model_path: str,
    host: str = "0.0.0.0",
    port: int = 8080,
    binary: str = "llama-server",
) -> LaunchSpec:
    if plan.engine != "llamacpp":
        raise PlanningError("plan was made for a different engine")
    argv = [
        binary,
        "--model",
        model_path,
        "--n-gpu-layers",
        str(plan.model.n_layers + 1),  # +1 offloads the output head too
        "--split-mode",
        "layer",
        "--tensor-split",
        ",".join(str(n) for n in plan.layer_counts),
        "--main-gpu",
        "0",
        "--ctx-size",
        str(plan.context),
        "--host",
        host,
        "--port",
        str(port),
    ]
    notes = []
    endpoints, devices = llamacpp_devices(plan)
    if endpoints:
        # --device fixes the pipeline order across local and remote GPUs; --tensor-split
        # follows the same order.
        argv += ["--rpc", ",".join(endpoints), "--device", ",".join(devices)]
        notes.append(
            "fabric plan: check the device names with "
            f"`{binary} --rpc {','.join(endpoints)} --list-devices` (llama.cpp built with GGML_RPC)"
        )
    if not plan.model.source.startswith("gguf:"):
        notes.append("catalog-based estimate: re-plan with --gguf <file> for an exact layer split")
    return LaunchSpec(tuple(argv), _gpu_env(plan), tuple(notes))


def vllm(
    plan: Plan,
    model_ref: str | None = None,
    host: str = "0.0.0.0",
    port: int = 8000,
    binary: str = "vllm",
) -> LaunchSpec:
    if plan.engine != "vllm":
        raise PlanningError("plan was made for a different engine")
    model = model_ref or plan.model.hf_repo
    if not model:
        raise PlanningError("vLLM needs a Hugging Face model id; pass --model-ref")
    stages = len(plan.placements)
    argv = [
        binary,
        "serve",
        model,
        "--pipeline-parallel-size",
        str(stages),
        "--tensor-parallel-size",
        "1",
        "--gpu-memory-utilization",
        f"{plan.gpu_memory_utilization:.2f}",
        "--max-model-len",
        str(plan.context),
        "--host",
        host,
        "--port",
        str(port),
    ]
    # bfloat16 needs Ampere (cc 8.0)+; float16 works on every vLLM-capable GPU.
    if any((p.device.compute_capability or 0) < 8.0 for p in plan.placements):
        argv += ["--dtype", "float16"]
    if plan.model.quant in ("awq-int4", "gptq-int4"):
        argv += ["--quantization", plan.model.quant.split("-")[0]]
    env = _gpu_env(plan)
    if stages > 1:
        # Uneven layer partition across pipeline stages (vLLM splits evenly otherwise).
        env["VLLM_PP_LAYER_PARTITION"] = ",".join(str(n) for n in plan.layer_counts)
    notes = (
        "vLLM's --gpu-memory-utilization is applied uniformly per GPU; "
        "VLLM_PP_LAYER_PARTITION carries the uneven split.",
    )
    return LaunchSpec(tuple(argv), env, notes)

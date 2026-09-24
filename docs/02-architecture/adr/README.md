# Architecture Decision Records

Format: context → decision → consequences (after M. Nygard). An accepted ADR is
immutable; to change a decision, write a new ADR that supersedes it.

| ADR | Title | Status |
|---|---|---|
| [0001](0001-native-pcie-over-oculink.md) | Native PCIe over OCuLink instead of USB4/Thunderbolt | Accepted |
| [0002](0002-pcie-packet-switch.md) | PCIe packet switch instead of host bifurcation | Accepted |
| [0003](0003-pipeline-parallel-vram-pooling.md) | Pipeline-parallel VRAM pooling; llama.cpp primary engine | Accepted; §2/§4 amended by 0010 |
| [0004](0004-gpu-identity-by-uuid.md) | Address GPUs by UUID in PCI-bus order | Accepted |
| [0005](0005-stdlib-control-plane.md) | Stdlib-only Python control plane using nvidia-smi XML and sysfs | Accepted |
| [0006](0006-ubuntu-lts-host.md) | Ubuntu 24.04 LTS as the production host OS | Accepted |
| [0007](0007-purpose-built-exporter.md) | Purpose-built Prometheus exporter instead of DCGM | Accepted |
| [0008](0008-containers-and-systemd.md) | Docker Compose profiles for workloads, systemd for node services | Accepted |
| [0009](0009-hardware-agnostic-gpu-support.md) | Hardware-agnostic GPU support: capability catalog + compatibility gate | Accepted (CR-001) |
| [0010](0010-speed-aware-gpu-selection.md) | Speed-aware placement and automatic GPU-set selection | Accepted (CR-001) |
| [0011](0011-astra-fabric.md) | ASTRA Fabric: exo-style GPU pooling across machines (llama.cpp RPC) | Accepted (CR-003) |
| [0012](0012-web-console.md) | Web console (`astra ui`): topology, planning, health, chat | Accepted (CR-004) |
| [0013](0013-zero-input-automation.md) | Zero-input automation (`astra auto`) with NVIDIA topology detection | Accepted (CR-005) |
| [0014](0014-astra-stack-bricks.md) | ASTRA Stack: 1-GPU bricks on a PCIe star | Accepted (CR-006) |
| [0015](0015-measured-speed-tuning.md) | Speed features (speculative decoding, KV cache type) are measured, not assumed | Accepted |
| [0016](0016-tensor-parallel-split.md) | Tensor-parallel split inside one machine (P2P / NVLink aware) | Accepted |

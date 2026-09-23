# CR-005 — Detect everything automatically

| | |
|---|---|
| Raised | 2026-09-24 by the Sponsor ("make it automatic: it should pick how many GPUs there are instead of us saying what we have; use NVIDIA tools") |
| Status | **Approved and implemented** in v0.5.0 (ADR-0013) |
| Affects | PRD FR-20, new `astra auto`, `astra.asbuilt`, `hardware/nvtopo.py`, per-user config path, check L04 on Windows |

## Impact
* **Requirements:** FR-20: one command detects all GPUs (local and on the network),
  configures, recommends, prepares, launches and verifies without operator input.
* **Software:** `astra/auto.py`; `nvidia-smi topo -m` parser; GGUF download sources for
  every catalog model (20 URLs verified); per-user config location.
* **Tests:** `tests/unit/test_auto.py` (recommendation, CUDA build choice, llama.cpp
  release selection, config sync, topology, dry run); a real run on the field-test PC.

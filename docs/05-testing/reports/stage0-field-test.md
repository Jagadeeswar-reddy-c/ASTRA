# Field Test Report — Stage 0 (single PC)

| | |
|---|---|
| Report ID | ASTRA-QA-FT-000 |
| Date | 2026-09-23/24 |
| Plan | [field-test-plan.md](../field-test-plan.md), Stage 0 (+ Stage 1 on loopback) |
| System | Windows 11, NVIDIA GeForce RTX 3060 Ti 8 GB (Ampere, display GPU), driver 610.88 |
| Engine | llama.cpp b11149, Windows CUDA 12.4 build (`llama-server.exe`, `ggml-rpc-server.exe`) |
| ASTRA | v0.4.1 → fixes released in v0.4.2 |
| Verdict | **PASS**: every result within tolerance; 4 defects found and fixed, 1 open improvement |

## Results

| Test | Model | Planned | Measured | Δ | Result |
|---|---|---|---|---|---|
| GGUF sizing | Llama-3.2-3B Q4_K_M | 28 layers, 2.17 GiB | llama.cpp loaded 28 + output layers | — | PASS |
| Decode speed, local | Llama-3.2-3B Q4_K_M | ~150 tok/s | 139.4 / 143.8 tok/s | −4…−7 % | PASS (±30 %) |
| Engine memory, local | Llama-3.2-3B Q4_K_M | 2.33 GiB model + 0.77 GiB reserve | +2.52 GiB | inside the reserve | PASS |
| Runtime gate R01–R05 | Llama-3.2-3B Q4_K_M | — | 5/5 PASS, 59 °C | — | PASS |
| Decode speed, local (tight fit, 584 MiB headroom) | Qwen2.5-7B Q4_K_M | ~75 tok/s | 73.2 / 74.2 tok/s | −1…−2 % | PASS |
| Engine memory, local | Qwen2.5-7B Q4_K_M | 4.30 GiB model + 0.77 GiB reserve | +4.56 GiB | inside the reserve | PASS |
| Runtime gate R01–R05 | Qwen2.5-7B Q4_K_M | — | 5/5 PASS, 63 °C | — | PASS |
| Console chat (proxy → real engine) | Llama-3.2-3B Q4_K_M | — | streamed reply | — | PASS |
| **Fabric over llama.cpp RPC** (real agent, supervised `ggml-rpc-server`, `plan --peers`, generated `--rpc … --device RPC0`) | Llama-3.2-3B Q4_K_M | ~94 tok/s (2 × 2 ms hops) | 112.3 / 115.6 tok/s | +20 % (loopback is faster than the LAN default) | PASS |

**Network-hop calibration:** local 7.0 ms/token vs RPC over loopback 8.8 ms/token gives
≈ 0.9 ms per hop on loopback. The 2 ms default (`fabric.network_hop_ms`) is kept for
1 GbE; re-measure in Stage 1 on a real LAN.

## Defects found (all fixed in v0.4.2)

| ID | Severity | Finding | Fix |
|---|---|---|---|
| FT-01 | **Sev-2** | llama.cpp names remote devices `RPC0, RPC1, …` (in `--rpc` order), not `RPC[host:port]`. Every generated fabric command would have failed | `engines.llamacpp_devices` emits `RPC<i>`; verified end-to-end over RPC |
| FT-02 | Sev-3 | The RPC server binary is now `ggml-rpc-server` | `find_rpc_server()` tries the configured name, then `rpc-server` and `ggml-rpc-server` |
| FT-03 | Sev-3 | Generated engine commands bound to `0.0.0.0` (exposed on the LAN), contradicting NFR-09 | Default `--host 127.0.0.1`; exposing is explicit |
| FT-04 | Sev-4 | Console form did not reflect plans made from a GGUF file | Form shows "<file> (from CLI plan)" |

## Open improvement

| ID | Finding | Proposal |
|---|---|---|
| FT-05 | On Windows (WDDM) `nvidia-smi` reports ~2.4 GiB used while CUDA reports ~1.05 GiB used on the same display GPU. Budgets are therefore ~1.3 GiB conservative on Windows display GPUs | Optionally take free memory from CUDA (`llama-server --list-devices`) when available; Linux/headless GPUs are unaffected |

## Evidence
`plan-3b.json`, `plan-7b.json`, `plan-rpc.json`, `stage0-3b.md`, `stage0-7b.md` and
console screenshots were captured on the test machine (`L:\astra-lab`, not
committed: they contain machine-specific GPU UUIDs).

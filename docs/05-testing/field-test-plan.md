# ASTRA — Field Test Plan (real hardware)

| | |
|---|---|
| Document ID | ASTRA-QA-002 |
| Owner | QA Lead · with Hardware Engineer |
| Goal | Prove ASTRA on real hardware in **four stages of increasing cost**, so money is only spent once the previous stage has passed |

| Stage | What it proves | Hardware needed | Cost |
|---|---|---|---|
| **0 — Today** | Real engine, real model, planner accuracy, runtime gate, console chat | Your current PC (RTX 3060 Ti, Windows) | none |
| **1 — Two PCs** | ASTRA Fabric over the LAN (agent, discovery, llama.cpp RPC) | Current PC + any second PC with an NVIDIA GPU, wired LAN | none (borrow) |
| **2 — Two GPUs, one PC** | Heterogeneous pooling on local PCIe, Linux-only checks | Any desktop with two PCIe x16-size slots + a second GPU | 1–2 GPUs |
| **3 — The ASTRA node** | External chassis: OCuLink link, power sync, thermals, full TC-HW | Head mini PC + chassis parts (BOM) | full BOM |

Send the listed evidence back after each stage. It is recorded in `test-cases.md`
and closes the matching TC IDs.

---

## Stage 0 — on your current PC (no purchase)

**One command does all of Stage 0** (downloads llama.cpp and the model if missing,
plans, starts the engine, measures, runs the runtime gate, writes a report):

```powershell
powershell -ExecutionPolicy Bypass -File scripts\stage0-test.ps1              # 3B model
powershell -ExecutionPolicy Bypass -File scripts\stage0-test.ps1 -Model 7b    # 7B model
```

Results go to `%USERPROFILE%\astra-lab\results-<time>\` (`report.md`, `results.json`,
`runtime-gate.md`, `plan.json`). The same script works after adding GPUs: it plans across
every GPU ASTRA finds. The manual steps below do the same thing by hand.

1. **llama.cpp for Windows with CUDA.** From https://github.com/ggml-org/llama.cpp/releases
   download the Windows CUDA build (`llama-<build>-bin-win-cuda-<ver>-x64.zip`)
   and, if listed separately, the matching `cudart-…` zip. Unzip both into
   `C:\llama`. Check: `C:\llama\llama-server.exe --list-devices` shows the RTX 3060 Ti.
2. **A model (GGUF).** Start small, then go bigger:
   * `Llama-3.2-3B-Instruct-Q4_K_M.gguf` (~2 GB)
   * `Qwen2.5-7B-Instruct-Q4_K_M.gguf` (~4.7 GB), from the model publishers'
     GGUF repositories on Hugging Face. Save them to `C:\models\`.
3. **Plan it** (PowerShell, in `L:\ASTRA`):
   ```powershell
   .\.venv\Scripts\astra plan --gguf C:\models\Llama-3.2-3B-Instruct-Q4_K_M.gguf --context 4096 --binary C:\llama\llama-server.exe --output plan.json
   ```
   Copy the printed **PowerShell launch line** and run it in a second terminal.
4. **Console:** `.\.venv\Scripts\astra ui`, then open http://127.0.0.1:9838/. The engine
   pill should turn green. Chat in the Chat tab and note the measured tok/s.
5. **Runtime gate while chatting:**
   `.\.venv\Scripts\astra validate --phase runtime --plan plan.json --format md --output stage0.md`
6. Repeat 3–5 with the 7B model.

**Pass:** chat works in the console; R01–R03 and R05 pass (R04 may WARN when warm);
measured tok/s is within ±30 % of the plan's estimate.
**Send back:** `stage0.md`, `plan.json`, the measured tok/s, a console screenshot.

## Stage 1 — two PCs over the LAN (no purchase)

Any second PC with an NVIDIA GPU works (desktop or laptop; Turing or newer is
simplest). Use a wired network.

**Second PC (worker):**
1. Install Python 3.11, clone the repo, `pip install -e .`, and llama.cpp as in
   Stage 0 (or run `scripts\stage0-test.ps1` on it once). The RPC server is
   `ggml-rpc-server.exe` in current llama.cpp releases; the agent finds it by itself.
   If it is missing, build llama.cpp with `-DGGML_CUDA=ON -DGGML_RPC=ON`.
2. Give it a name: create `astra.toml` with `[node]` and `name = "pc2"`.
3. Firewall (Private network only): allow UDP 9836 and TCP 9837, 50052.
4. `astra agent --rpc --rpc-binary C:\llama\ggml-rpc-server.exe`

**Your PC (head):**
1. `astra cluster` should list `pc2` with its GPU "rpc <ip>:50052". If discovery
   is blocked, use `--peers <ip of pc2>`.
2. Plan a model too big for either GPU alone, e.g. Qwen2.5-14B Q4_K_M (~9 GB):
   `astra plan --cluster --gguf C:\models\Qwen2.5-14B-Instruct-Q4_K_M.gguf --binary C:\llama\llama-server.exe --output plan.json`
3. Check the device names once:
   `C:\llama\llama-server.exe --rpc <ip>:50052 --list-devices`. They must match the
   plan's `--device` list.
4. Run the launch line, then `astra ui --cluster` and chat.

**Pass:** both machines appear in the topology, the plan uses GPUs on both, chat
works, and measured tok/s is within ±30 % of the estimate (TC-RT-08).
**Send back:** `astra cluster --json`, `plan.json`, measured tok/s, a screenshot.

## Stage 2 — two different GPUs in one desktop (first purchase)

How to connect them (slots, x16/x8/x4, risers, power): [connecting-gpus.md](../03-solution-design/connecting-gpus.md).

Proves pooling across mixed GPUs on local PCIe, plus the Linux-only checks, before
buying any chassis parts.

* **PC:** any ATX desktop with two PCIe x16-size slots (the second is often x4
  electrically, which is fine) and a **750 W+** PSU for the whole PC.
* **GPUs:** buy **one RTX 3060 12 GB** and pair it with a **GTX 1660 SUPER** (or your
  RTX 3060 Ti). Both are "recommended" tier, so there is no driver pin. Together they
  run Qwen2.5-14B Q6_K with ~21k context (`astra plan --simulate "GTX 1660 SUPER, RTX 3060:12288" --model qwen2.5-14b --quant q6_k`).
* **OS:** Ubuntu 24.04 (a spare SSD or dual boot) → `deploy/host/setup-host.sh`.
* **Run:** `astra compat`, `astra validate`, TC-RT-01 (pooled llama.cpp), TC-RT-02 (vLLM
  AWQ), TC-RT-05 (restart), TC-HW-06 (bandwidth) and TC-HW-07 (thermal soak).

## Stage 3 — the ASTRA node (chassis)

### Head PC: what to buy

The head PC has **no GPU of its own**. It drives the chassis and serves the API.

| Item | Minimum | Recommended | Why |
|---|---|---|---|
| Link to the chassis | **Path A:** built-in **OCuLink** port (PCIe 4.0 x4), common on recent mini PCs; or an M.2 slot for an OCuLink adapter | **Path B:** mini-ITX desktop with a **PCIe x16 slot that supports bifurcation (x4x4x4x4)** | A uses the PEX8747 switch design as-is; B needs no switch and gives each GPU its own x4 link (CR-002, pending your choice) |
| CPU | 6 cores with integrated graphics | Ryzen 5 7600 / 8600G or Core i5-12400/13400 (non-F) | iGPU for setup; the GPUs do the inference |
| RAM | 32 GB | 64 GB | Models load through system RAM; 14B+ and CPU offload need headroom |
| Storage | 1 TB NVMe | 2 TB NVMe Gen4 | Models are 2–20 GB each |
| Network | 1 GbE | **2.5 GbE** | Fabric traffic to other PCs |
| BIOS | Above 4G Decoding | + PCIe bifurcation (Path B) | Needed for several GPUs behind one link |
| Chassis power sync | spare SATA/Molex power for the sync relay | — | **Most mini PCs have none.** Then power the chassis on **before** the PC and off **after** it, or use a sync module that senses a USB port's 5 V |
| OS | Ubuntu 24.04 LTS | — | ADR-0006 |

Before buying, check the **exact motherboard manual** for "bifurcation" (Path B) or
the spec sheet for "OCuLink PCIe 4.0 x4" (Path A).

### Chassis parts
`hardware/bom.csv` (BOM-01…11). The PSU is sized with `astra compat` for the GPUs you
fit: 550 W is enough for GTX 1660 SUPER + RTX 3060, and 650 W is the default.

### Tests
The hardware design §6 assembly SOP, then TC-HW-01…12 and TC-RT-01…09.

---

## What ASTRA supports today
See [supported-gpus.md](../03-solution-design/supported-gpus.md), or run `astra gpus`.
In short: **GTX 16xx and every RTX card (20/30/40/50) are fully supported.**
GT 1030 and GTX 9xx/10xx work but force the R580 driver. GTX 6xx/7xx Kepler cards
are not supported.

# Test Report — New-user acceptance (fresh clone → running model)

| | |
|---|---|
| Report ID | ASTRA-QA-FT-001 |
| Date | 2026-09-24 |
| Procedure | [GETTING-STARTED.md](../../GETTING-STARTED.md), steps 1–4, followed literally |
| System | Windows 11, RTX 3060 Ti 8 GB (display GPU), driver 610.88, Python 3.11 |
| Starting state | New folder; `git clone` from GitHub (v0.5.1); **empty** config directory; **no** llama.cpp, **no** model, no `astra-lab` folder |
| Verdict | **PASS** |

## Steps and results

| Step | Command | Result |
|---|---|---|
| 1 Get the code | `git clone https://github.com/Jagadeeswar-reddy-c/ASTRA.git` | OK |
| 2 Install + run | `powershell -ExecutionPolicy Bypass -File scripts\install.ps1` (with `-AutoArgs '--no-ui'` so the test ends by itself) | Found Python 3.11, created `.venv`, installed astra 0.5.1, found the GPU |
| 3a Detect | (automatic) | 1 GPU: RTX 3060 Ti 8 GB, Ampere, PCIe Gen4 x16; no other machines on the LAN |
| 3b Configure | (automatic) | Created a fresh `astra.toml` in the user's config directory |
| 3c Check | (automatic) | Driver window R455–current; PSU OK |
| 3d Choose | (automatic) | Llama-3.2-3B **Q8_0** (the 7B no longer fit with margin because other programs held more of the display GPU's memory at that moment: correct, live-memory behaviour) |
| 3e Download | (automatic) | llama.cpp b11149 CUDA 12.4 (242 MB + 373 MB runtime), model 3.19 GB, all with progress |
| 3f Launch + verify | (automatic) | Engine healthy; **98.2 tok/s measured vs 87 estimated (+14 %)**; runtime gate **R01–R05 PASS**; `RESULT: PASS` |

## Findings

| ID | Finding | Action |
|---|---|---|
| NU-01 | Stage-0 script (`stage0-test.ps1`) downloaded straight to the final file name; an interrupted download left a truncated model that a re-run would treat as complete | Fixed: downloads go to `.part` and are renamed on completion (as `astra auto` already did) |
| NU-02 | Platform shown as "Windows 10" on Windows 11 (Python reports release 10) | Fixed: detected from the build number |
| NU-03 | The chosen model depends on free GPU memory at the moment of the run (display GPU) | By design. Documented in the guide; `--model/--quant` pin a choice |

# Getting started with ASTRA

From zero to a running, verified AI model on your NVIDIA GPU(s) in about 15 minutes,
most of it download time. No configuration needed: ASTRA detects your hardware itself.

---

## 0. What you need

| | Requirement | Check with |
|---|---|---|
| GPU | Any NVIDIA GTX 16xx / RTX 20–50 series (GTX 9xx/10xx work too: see [supported-gpus](03-solution-design/supported-gpus.md)) | — |
| Driver | NVIDIA driver installed | `nvidia-smi` lists your GPU |
| OS | Windows 10/11, or Ubuntu 22.04/24.04 | — |
| Python | 3.11 or newer | `py -3.11 --version` (Windows) / `python3 --version` |
| Git | any | `git --version` |
| Disk | ~10 GB free (llama.cpp ~0.7 GB + a model 2–9 GB) | — |
| Internet | for the first run (downloads) | — |

Missing something?
* **Windows:** `winget install Python.Python.3.12`, `winget install Git.Git`, and the
  driver from nvidia.com/drivers.
* **Ubuntu:** `sudo apt install python3 python3-venv git` and `sudo ubuntu-drivers install`.

Reboot after installing a driver.

## 1. Get the code

```bash
git clone https://github.com/Jagadeeswar-reddy-c/ASTRA.git
cd ASTRA
```

## 2. Install and run: one command

**Windows (PowerShell):**
```powershell
powershell -ExecutionPolicy Bypass -File scripts\install.ps1
```

**Ubuntu:**
```bash
bash scripts/install.sh
```
On Linux, ASTRA also needs llama.cpp's server once (NVIDIA publishes prebuilt CUDA
builds only for Windows):
```bash
sudo apt install -y build-essential cmake nvidia-cuda-toolkit
git clone https://github.com/ggml-org/llama.cpp && cd llama.cpp
cmake -B build -DGGML_CUDA=ON -DGGML_RPC=ON && cmake --build build -j --config Release
sudo cp build/bin/llama-server build/bin/*rpc-server /usr/local/bin/
```
(Ubuntu's `nvidia-cuda-toolkit` is fine up to RTX 40-series; RTX 50-series needs CUDA
12.8+ from NVIDIA's CUDA repository.)

The installer creates `.venv`, installs ASTRA, checks the driver and starts
**`astra auto`**. Next time, just run:

```text
Windows:  .venv\Scripts\astra auto
Ubuntu:   .venv/bin/astra auto
```

## 3. What you will see

```text
[1/6] Detecting GPUs (nvidia-smi) and the network fabric
    1 NVIDIA GPU(s), driver 610.88
    GPU0  RTX 3060 Ti   8.00 GiB  Ampere  PCIe Gen4 x16
[2/6] Configuration
    created C:\Users\<you>\AppData\Roaming\astra\astra.toml
[3/6] Compatibility and power
    driver branch needed: R455 to current; PSU ok: yes
[4/6] Choosing a model for this pool (context 8192)
    qwen2.5-7b    q4_k_m   ~72 tok/s  on RTX 3060 Ti
    -> Qwen2.5 7B Instruct Q4_K_M
[5/6] Preparing llama.cpp and the model          <- downloads on the first run only
[6/6] Launching and verifying
    speed: 74.4 tok/s measured vs ~73 estimated (+2 %)
    runtime gate: R01 PASS, R02 PASS, R03 PASS, R04 PASS, R05 PASS
    RESULT: PASS
    console: http://127.0.0.1:9838/  (Ctrl+C stops the engine and console)
```

The model keeps running while this window is open. **Ctrl+C** stops everything.

## 4. Check that it works

| # | Check | How | Expected |
|---|---|---|---|
| 1 | Automatic test passed | the `astra auto` output | `RESULT: PASS` |
| 2 | Console | open http://127.0.0.1:9838/ | your GPU(s) on the map, a coloured layer strip, the engine pill **online** |
| 3 | Chat | console → **Chat** tab → ask anything | an answer streams in, with the measured tok/s |
| 4 | API from another program | the commands below | JSON with an answer |
| 5 | Health checks | `astra validate` (second terminal) | `Result: PASS` (Windows skips the Linux-only checks) |
| 6 | Saved results | the folder shown after `RESULT:` | `plan.json`, `results.json`, `llama-server.log` |

API test (PowerShell):
```powershell
Invoke-RestMethod http://127.0.0.1:8080/v1/chat/completions -Method Post -ContentType application/json `
  -Body '{"messages":[{"role":"user","content":"Say hello in five words"}]}' | Select-Object -Expand choices
```
API test (bash):
```bash
curl -s http://127.0.0.1:8080/v1/chat/completions -H 'Content-Type: application/json' \
  -d '{"messages":[{"role":"user","content":"Say hello in five words"}]}'
```

## 5. Use it from your own apps

The model speaks the **OpenAI API** at `http://127.0.0.1:8080/v1`, so any OpenAI client works:

```python
from openai import OpenAI                       # pip install openai
client = OpenAI(base_url="http://127.0.0.1:8080/v1", api_key="not-needed")
print(client.chat.completions.create(
    model="local", messages=[{"role": "user", "content": "Hello!"}]
).choices[0].message.content)
```

Tools that let you set an "OpenAI base URL" (Open WebUI, VS Code extensions,
LangChain, …) work the same way.

## 6. Useful options

| Want to… | Run |
|---|---|
| See what it would do, without downloading or launching | `astra auto --no-launch` |
| Pick the model yourself | `astra auto --model qwen2.5-14b --quant q4_k_m` (list: `astra models`) |
| Longer conversations | `astra auto --context 16384` |
| Keep downloads somewhere else | `astra auto --lab-dir D:\astra-lab` |
| Verify and exit (no console) | `astra auto --no-ui` |
| Just see your GPUs | `astra probe` · `astra gpus` · `astra compat` |

## 7. Run it permanently

* **Windows (start at logon):**
  ```powershell
  schtasks /Create /TN "ASTRA" /SC ONLOGON /TR "\"$PWD\.venv\Scripts\astra.exe\" auto"
  ```
  Remove it with `schtasks /Delete /TN "ASTRA" /F`.
* **Ubuntu (a server, starts at boot):** follow [deployment-guide.md](06-deployment/deployment-guide.md):
  `deploy/host/setup-host.sh` installs the systemd services (self-test, exporter,
  agent, console), or use the Docker Compose stack with monitoring.

## 8. Add more GPUs or another PC

* **Another GPU in this PC:** power off, install it (see
  [connecting-gpus.md](03-solution-design/connecting-gpus.md) for slots and power),
  then run **`astra auto`** again. It detects the new card and picks a bigger model if
  the pool allows.
* **A second PC with a GPU:** install ASTRA there too and run `astra agent --rpc`. Your
  main PC's `astra auto` finds it on the network and uses both.

## 9. Update or uninstall

```bash
git pull && .venv/Scripts/pip install -e .      # update   (Ubuntu: .venv/bin/pip)
```
Uninstall: delete the `ASTRA` folder, the `astra-lab` folder in your home directory
(llama.cpp + models), and the config (`%APPDATA%\astra` or `~/.config/astra`).

## 10. If something goes wrong

| Message / symptom | Fix |
|---|---|
| `Python 3.11+ not found` | Install Python (step 0), open a **new** terminal |
| `no NVIDIA GPU found` / `nvidia-smi not found` | Install the NVIDIA driver, reboot, check `nvidia-smi` |
| `running scripts is disabled on this system` | Use the exact command with `-ExecutionPolicy Bypass` from step 2 |
| A smaller model than expected was chosen | ASTRA plans against the GPU memory free *right now*: close GPU-heavy apps and run again, or pin it with `--model/--quant` |
| `nothing in the catalog fits` | Your GPU has little free memory: close games/browsers using the GPU, or `astra auto --context 4096` |
| Download slow or interrupted | Run `astra auto` again; finished files are reused |
| `port 8080 is already in use` | Another engine is running: close it, or `astra auto --port 8090` |
| A GPU is missing after adding it | See the troubleshooting table in [connecting-gpus.md](03-solution-design/connecting-gpus.md#7-troubleshooting) |
| `llama-server not found` (Linux) | Build llama.cpp (step 2) or pass `--binary /path/to/llama-server` |
| Anything else | Run `astra probe --json` and `astra compat`, and open an issue with the output |

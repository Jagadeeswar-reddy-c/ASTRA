# ASTRA — Developer Setup

## Prerequisites
* Python 3.11 or 3.12. An NVIDIA GPU and driver are optional: the hardware tests
  skip themselves without one.
* Docker (optional) for the image and Compose checks.

## Setup

Linux / macOS:
```bash
python3.11 -m venv .venv && . .venv/bin/activate
pip install -e ".[dev]"
```

Windows (PowerShell):
```powershell
py -3.11 -m venv .venv; .\.venv\Scripts\Activate.ps1
pip install -e ".[dev]"
```

## Everyday commands

| Task | Command |
|---|---|
| All tests + coverage | `pytest --cov` |
| Unit tests only | `pytest tests/unit` |
| Hardware tests only | `pytest -m hardware` |
| Lint / format | `ruff check src tests` · `ruff format src tests` |
| Types | `mypy` |
| Try the planner without hardware | `astra plan --simulate reference --model qwen2.5-14b` |
| Plan for other cards | `astra plan --simulate "RTX 4060:8192:Ada,RTX 3060:12288:Ampere" --model qwen2.5-14b --quant q5_k_m` |
| Build the image | `docker build -f deploy/docker/Dockerfile -t astra-control:dev .` |
| Validate Compose | `cp deploy/compose/.env.example deploy/compose/.env && docker compose -f deploy/compose/compose.yaml --profile pooled config --quiet` |

## Adding a recorded fixture from a real node

```bash
nvidia-smi -q -x > /tmp/raw.xml
# Sanitise: replace <uuid> values, strip <processes> and <supported_clocks>, then save as
# tests/fixtures/nvsmi_<description>_driver<branch>.xml and add a parser test.
```

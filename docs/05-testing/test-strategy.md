# ASTRA — Test Strategy

| | |
|---|---|
| Document ID | ASTRA-QA-001 |
| Version | 1.0 (2026-09-23) |
| Owner | QA Lead · Approver: Enterprise Architect |

## 1. Objectives
Show that every M-priority requirement in the PRD is met, that the design-review
fixes work, and that the node is safe to put into service. Testing is
**risk-based**: the high risks R-01, R-02, R-03 and R-05 get both automated and
manual coverage.

## 2. Test levels

| Level | ID prefix | What | Where | Automated | Gate |
|---|---|---|---|---|---|
| Unit | TC-SW | Parsers, planner, checks, exporter, CLI, config — through ports with fixtures | CI (Ubuntu + Windows, py3.11/3.12) | Yes | G4 |
| Integration | TC-SW-12 | Parser, checks and planner against the real driver | Any machine with a GPU; the node | Yes (`-m hardware`) | G4/G5 |
| Deployment artefacts | TC-SW-13/14 | Image build and smoke test, compose config, promtool, shellcheck | CI | Yes | G4 |
| Hardware acceptance | TC-HW | Assembly, power, enumeration, link, bandwidth, thermal, grounding | Assembled node | Partly (link gate) | G5 |
| Runtime acceptance | TC-RT | Pooled/partitioned serving, runtime gate, soak, recovery | Assembled node | Partly (runtime gate) | G5/G7 |

## 3. Environments

| Env | Hardware | Purpose |
|---|---|---|
| CI | GitHub-hosted runners, no GPU | Unit tests, lint/types, artefacts; hardware tests self-skip |
| Dev workstation | Any NVIDIA GPU (e.g. RTX 3060 Ti, Windows 11) | Integration tests against a real driver; planning |
| **Reference node** | Host + ASTRA chassis (RTX 2060 + RTX 3050 behind a PEX8747), Ubuntu 24.04 | TC-HW, TC-RT, soak |

Staged real-hardware testing (what to buy and in which order) is in
[field-test-plan.md](field-test-plan.md).

## 4. Entry and exit criteria

| Phase | Entry | Exit |
|---|---|---|
| Unit/integration (G4) | Code complete for the scope | 100 % of TC-SW pass; coverage ≥ 85 % line+branch; lint and strict types clean |
| Hardware acceptance (G5) | BOM received; assembly SOP done | All TC-HW pass; link gate PASS on 10/10 cold boots |
| Runtime acceptance (G5) | Link gate PASS | TC-RT-01…03, 05 and 06 pass |
| Soak (G7) | Runtime acceptance passed | TC-RT-04: 24 h with no Xid/AER/replay growth and no thermal slowdown |

## 5. Defect management

| Severity | Definition | Example | Blocks gate? |
|---|---|---|---|
| Sev-1 | Safety risk or hardware damage risk; data/model corruption | GPU fed from two PSUs; link drops under load | Yes — fix before any further testing |
| Sev-2 | A must-have requirement fails; no workaround | Runtime split off by 40 %; GPU missing on 1 in 10 boots | Yes |
| Sev-3 | Degraded function with a workaround | Link gate WARN on idle downshift | No; must be triaged |
| Sev-4 | Cosmetic, docs | Table misalignment | No |

Defects are GitHub issues labelled `defect` and `sev-N`, each linking the failing
TC ID and its evidence (report JSON/JUnit, logs).

## 6. Evidence

* Automated: JUnit XML from `pytest` and from `astra validate --format junit`,
  archived per CI run or per node test session.
* Manual: the TC record in `test-cases.md` with date, tester, result and attached
  measurements or photos. Results roll up in `traceability-matrix.md`.

## 7. Tools

pytest, pytest-cov, ruff, mypy; `astra validate` (link/runtime gates);
`nvidia-smi`, `lspci -tvv`; `nvbandwidth` or CUDA samples `bandwidthTest`;
`gpu-burn` for thermal load; a load generator for the OpenAI API (e.g. `llama-bench`,
`vllm bench serve`); a multimeter and PSU tester.

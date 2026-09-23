# ASTRA — Project Charter

| | |
|---|---|
| Document ID | ASTRA-GOV-001 |
| Version | 1.0 (2026-09-23) |
| Status | Draft — awaiting sponsor approval (Gate G0) |
| Owner | Project Manager |

## 1. Purpose

ASTRA (Asynchronous Scalable Topology for Resource Aggregation) builds an **external,
modular GPU compute node**. The node takes GPUs out of the host PC and puts them in
an expansion chassis connected over native PCIe-over-cable (OCuLink 4i). Software
then pools GPUs of different models into one inference target. The reference build
pools an RTX 2060 (6 GB, Turing) and an RTX 3050 (8 GB, Ampere) into 14 GB.

## 2. Objectives and success measures

| # | Objective | Measure of success |
|---|---|---|
| O1 | Reliable external PCIe link | Link gate (`astra validate`) passes on 10/10 cold boots; 0 uncorrectable AER in a 24 h soak |
| O2 | Usable pooled VRAM | A model that fits neither GPU alone (Qwen2.5-14B Q4_K_M) serves requests with the planned split within ±15 % |
| O3 | Safe power design | Synchronized power-on/off; chassis PSU sustained load ≤ 70 %; single-source rule verified |
| O4 | Operable service | Metrics, alerts and runbooks live; self-test blocks unsafe starts |
| O5 | Cost | Hardware ≤ €420 excluding GPUs already owned and one-off tooling (BOM: €285–420) |

## 3. Scope

**In scope:** chassis hardware design and assembly procedure; host integration (BIOS
and OS); the `astra` control plane (discovery, validation, planner, launcher,
exporter); container and systemd deployment; monitoring; test and acceptance.

**Out of scope (v1):** multi-host clustering; hot-plug of the external link; training
workloads; Windows as the production host (supported for tooling only); GPUs
beyond 4 per chassis.

## 4. Organisation and RACI

Roles are functions, not people. One person may hold several roles, but the
**author and the approver of a gate deliverable must be different roles**.

| Deliverable | Sponsor | PM | Enterprise Architect | Solution Architect | Hardware Eng. | Software Dev | QA | DevOps/SRE |
|---|---|---|---|---|---|---|---|---|
| Charter, risk register | A | R | C | C | C | I | I | I |
| PRD (requirements) | A | R | C | C | C | C | C | C |
| System architecture, ADRs | I | I | A/R | C | C | C | C | C |
| Solution design (HW + SW) | I | I | A | R | R | C | C | C |
| Control-plane code | I | I | C | A | C | R | C | C |
| Test strategy and cases | I | I | C | C | C | C | A/R | C |
| Hardware assembly and HW tests | I | I | I | C | R | I | A | I |
| Deployment artefacts | I | I | C | C | I | C | C | A/R |
| Runbooks, monitoring | I | I | I | C | C | C | C | A/R |

R = responsible, A = accountable, C = consulted, I = informed.

## 5. Delivery phases and gates

Each phase ends in a gate review. The gate passes only when its exit criteria are
met and the accountable role signs the gate record (`docs/00-governance/gate-log.md`).

| Gate | Phase | Exit criteria |
|---|---|---|
| **G0** | Initiation | Charter and risk register approved |
| **G1** | Requirements | PRD baselined; every requirement has an ID, a priority and a verification method |
| **G2** | Architecture | System architecture and ADRs approved; design review findings dispositioned |
| **G3** | Solution design | HW + SW design approved; BOM frozen; traceability REQ → design complete |
| **G4** | Development | CI green (lint, strict types, tests, ≥ 85 % coverage); code reviewed |
| **G5** | Verification | All TC-SW pass; TC-HW and TC-RT executed on the assembled node with no open Sev-1/Sev-2 defects |
| **G6** | Deployment | Node provisioned from the deployment guide by someone other than its author; rollback rehearsed |
| **G7** | Operations handover | Monitoring and alerts live; runbooks walked through; 24 h soak passed |

## 6. Change control

* Requirements and architecture are **baselined** at G1 and G2. After that, any
  change needs a change request (a GitHub issue labelled `change-request`) with an
  impact analysis covering requirements, design, tests and cost. The Enterprise
  Architect approves it.
* Architecture decisions are recorded as ADRs. An ADR is never edited after
  acceptance; a new ADR supersedes it.
* The BOM is frozen at G3. Part substitutions need Hardware Engineer and QA
  approval and a re-run of the affected TC-HW cases.

## 7. Cadence and communication

* Weekly status: progress against gates, top 5 risks, blockers.
* Gate reviews are held at phase end with all accountable roles present.
* Defects and changes are tracked in the issue tracker. Severity definitions are in
  `docs/05-testing/test-strategy.md`.

## 8. Key assumptions and constraints

* The GPUs (RTX 2060, RTX 3050) are already owned. The host has a free M.2 Key-M
  NVMe slot wired for PCIe x4.
* The production host OS is Ubuntu 24.04 LTS (ADR-0006).
* The node is a lab or edge appliance on a trusted LAN. Endpoints bind to localhost
  unless placed behind a reverse proxy.

# ASTRA — Gate Log

Record of gate reviews. A gate passes only with the accountable role's sign-off, and
the author of a deliverable may not approve it (see the charter, §4).

| Gate | Date | Deliverables reviewed | Decision | Approver (role) | Notes / conditions |
|---|---|---|---|---|---|
| G0 Initiation | — | project-charter.md, risk-register.md | Pending | Sponsor | |
| G1 Requirements | — | PRD.md | Pending | Sponsor | |
| G2 Architecture | — | system-architecture.md, design-review.md, ADR-0001…0008 | Pending | Enterprise Architect | 15 design-review findings need disposition |
| G3 Solution design | — | hardware-design.md, software-design.md, hardware/bom.csv | Pending | Enterprise Architect | BOM freeze, except the PSU (sized per GPU set, CR-001) |
| CR-001 | 2026-09-23 | change-requests/CR-001-any-nvidia-gpu.md | Approved | Enterprise Architect | Any NVIDIA GPU mix; ADR-0009/0010; FR-13–15 |
| G4 Development | — | `src/`, CI run | Pending | Solution Architect | v0.2.0 (CR-001): 143 automated tests green, 92 % coverage |
| G5 Verification | — | test reports TC-SW / TC-HW / TC-RT | Pending | QA Lead | Needs the assembled node |
| G6 Deployment | — | deployment rehearsal record | Pending | DevOps/SRE | |
| G7 Handover | — | 24 h soak report, runbook walkthrough | Pending | DevOps/SRE | |

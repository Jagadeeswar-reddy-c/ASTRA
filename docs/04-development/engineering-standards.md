# ASTRA — Engineering Standards

| | |
|---|---|
| Document ID | ASTRA-DEV-001 |
| Owner | Solution Architect |

## 1. Source control

* **Trunk-based development.** `main` is always releasable and protected: no direct
  pushes, one approving review, and CI must pass.
* Branches: `feat/<topic>`, `fix/<topic>`, `docs/<topic>`, `hw/<topic>`. Branches
  are short-lived (< 3 days).
* **Conventional Commits**: `feat(planner): …`, `fix(nvsmi): …`, `docs(adr): …`,
  `test: …`, `ci: …`, `chore: …`. A breaking change is marked `!` with a
  `BREAKING CHANGE:` footer.
* Squash-merge PRs. The PR title becomes the commit subject.

## 2. Pull request checklist (Definition of Done)

A change is done when:

- [ ] It traces to a requirement, ADR, defect or change request (link it in the PR).
- [ ] `ruff check`, `ruff format --check`, `mypy` (strict) and `pytest` pass locally
      and in CI, and coverage is ≥ 85 %.
- [ ] New behaviour has unit tests. Hardware-facing logic is tested through the
      `CommandRunner`/`SysfsReader` ports with recorded or synthetic fixtures.
- [ ] Docs are updated wherever they're affected: the CLI contract (software-design
      §4), the metric catalog, test cases, runbooks.
- [ ] `CHANGELOG.md` has an entry under *Unreleased*.
- [ ] Any change to deployment artefacts has passed `docker compose config`,
      `promtool` and `shellcheck`.

## 3. Code standards

* Python ≥ 3.11 with full type hints. `mypy --strict` is clean, and `Any` is used
  only at JSON boundaries.
* The core has no third-party runtime dependencies (ADR-0005). Adding one requires
  an ADR.
* Data records are frozen dataclasses. Functions are pure where practical, and I/O
  goes through ports.
* Expected errors raise an `AstraError` subclass with an actionable message. Never
  swallow exceptions silently.
* Comments explain *why* (a constraint, a hardware quirk, a reference), not *what*.
* User-facing strings name the next action ("check cable seating", "enable Above
  4G Decoding").

## 4. Testing standards

* **Unit** (`tests/unit`): fast, hermetic, and OS-independent. The fake sysfs is in
  memory because sysfs names contain `:`, which Windows rejects.
* **Integration** (`tests/integration`, marker `hardware`): runs against the real
  driver and skips itself where nvidia-smi is absent.
* **Fixtures**: recorded real outputs are sanitised (UUIDs replaced, processes
  stripped). Synthetic fixtures are labelled as synthetic in their header.
* Every validation check (L/R) has at least one passing and one failing test.

## 5. Versioning and releases

* Semantic Versioning for the `astra-node` package and the container image
  (`astra-control:<semver>`).
* The release procedure is in `docs/06-deployment/release-process.md`.

## 6. Documentation

* Docs live next to the code and are reviewed in the same PR.
* Architecture decisions are ADRs (immutable once accepted).
* Diagrams are Mermaid inside Markdown, so they diff like code.

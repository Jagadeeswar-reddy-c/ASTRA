# Contributing to ASTRA

1. Read the [engineering standards](docs/04-development/engineering-standards.md)
   and follow the [developer setup](docs/04-development/dev-setup.md).
2. Open or link an issue: a requirement, defect or change request. Changes to
   baselined requirements or architecture need a `change-request` issue approved by
   the Enterprise Architect (see the [charter](docs/00-governance/project-charter.md) §6).
3. Branch from `main` (`feat/…`, `fix/…`, `docs/…`, `hw/…`) and use Conventional
   Commits.
4. Before pushing:
   ```bash
   ruff check src tests && ruff format --check src tests && mypy && pytest --cov
   ```
5. Open a PR that references the issue and requirement IDs, and complete the
   Definition of Done checklist.

Hardware changes (the BOM, wiring, BIOS settings) follow the same flow. The PR must
list which TC-HW cases need re-running.

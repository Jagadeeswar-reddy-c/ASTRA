# ASTRA — Release Process

1. **Freeze:** every PR for the release is merged; `main` CI is green.
2. **Version:** bump `version` in `pyproject.toml` and `src/astra/__init__.py`
   (SemVer). Move *Unreleased* in `CHANGELOG.md` to the new version and date.
3. **Verify on the reference node:**
   `pytest -m hardware`, then `astra validate` (link gate) and TC-RT-01's runtime
   gate with the release candidate installed. Attach the JUnit reports to the
   release PR.
4. **Tag:** `git tag -a vX.Y.Z -m "ASTRA vX.Y.Z"` and push it.
5. **Artefacts:** build `astra-control:X.Y.Z` from the tag
   (`docker build -f deploy/docker/Dockerfile -t astra-control:X.Y.Z .`), push it
   to the registry, and record the digest in the release notes.
6. **Deploy:** follow the deployment guide §5/§6 on each node, then run the link
   and runtime gates.
7. **Post-release:** watch the dashboards for 24 h. Roll back per deployment guide
   §9 on any Sev-1/Sev-2.

Hotfixes branch from the tag (`fix/…`), are released as a patch version, and are
merged back to `main`.

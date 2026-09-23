# ASTRA — Deployment Guide

| | |
|---|---|
| Document ID | ASTRA-OPS-001 |
| Owner | DevOps/SRE |
| Audience | The person provisioning a node. Gate G6 requires that someone other than this document's author can follow it. |

**Order matters.** Hardware (§1) → BIOS (§2) → OS and host setup (§3) → link gate
(§4) → workload (§5 or §6) → runtime gate (§7) → monitoring (§8).

## 1. Hardware
Follow the assembly SOP in the [hardware design](../03-solution-design/hardware-design.md) §6
and record TC-HW-01…03 and TC-HW-11 in `docs/05-testing/test-cases.md`.
**Never connect or disconnect the OCuLink cable while either system is powered.**

## 2. BIOS / UEFI
Apply the checklist in the hardware design §5: Above 4G Decoding **on**, UEFI only,
Fast Boot off, S3/ErP off, M.2 link Auto (or Gen3 if errors appear).

## 3. Operating system and host setup
1. Install **Ubuntu 24.04 LTS** (server or desktop) on the host's own boot drive.
2. Install Docker Engine from Docker's apt repository
   (https://docs.docker.com/engine/install/ubuntu/).
3. **Choose the driver branch for your GPU mix first.** Install Python 3.11 and the
   `astra` CLI (`pip install .`), then run
   `astra compat --simulate "<your cards, e.g. RTX 3060, GT 1030>"`. If the window ends
   at R580 (any Maxwell, Pascal or Volta card), install `nvidia-driver-580` and hold
   it, and use `--skip-driver` below. Otherwise the default driver is fine.
4. Clone this repository to the host, then run a dry run:
   ```bash
   sudo deploy/host/setup-host.sh            # prints every action, changes nothing
   ```
5. Apply the changes. The kernel options are only needed if §4 shows the matching
   symptom:
   ```bash
   sudo deploy/host/setup-host.sh --apply
   sudo reboot                               # needed after a driver install
   sudo deploy/host/setup-host.sh --apply    # second pass after the driver reboot
   ```
   This installs the driver and the NVIDIA Container Toolkit, enables persistence
   mode, **masks sleep/hibernate**, installs `astra` to `/opt/astra/venv`, and
   installs `/etc/astra/astra.toml` and the systemd units.
6. Generate the as-built config from the installed GPUs, then review it (in particular
   `chassis.psu_watts`):
   `astra probe --emit-config | sudo tee /etc/astra/astra.toml`

## 4. Link gate (Phase 2)
```bash
/opt/astra/venv/bin/astra probe
/opt/astra/venv/bin/astra validate --format md --output /var/lib/astra/link-gate.md
sudo systemctl start astra-selftest && systemctl status astra-selftest
```

| Result | Action |
|---|---|
| All PASS | Continue |
| L05 WARN (idle downshift) | Re-run while `nvidia-smi dmon` or any GPU load is running. It must PASS under load |
| L02/L04 FAIL, GPU missing | RB-01 |
| L05/L06 FAIL (width) | RB-02 |
| L07/L08/L09 FAIL | RB-03. For "BAR n: no space", re-run setup with `--pci-realloc`. For corrected AER at idle, use `--disable-aspm` |
| L10 FAIL | RB-04 |

Then record TC-HW-04…10.

## 5. Workload via Docker Compose (recommended)
```bash
cd /path/to/astra
cp deploy/compose/.env.example deploy/compose/.env     # site settings: passwords, dirs
# Plan against the live GPUs; this writes the ASTRA_* split into .env (it overwrites
# the file, so re-append the site settings, or keep them in a separate env file).
/opt/astra/venv/bin/astra plan --gguf /srv/models/Qwen2.5-14B-Instruct-Q4_K_M.gguf \
    --context 8192 --output /var/lib/astra/plan.json --env-file deploy/compose/.env.plan
cat deploy/compose/.env.plan deploy/compose/.env.site > deploy/compose/.env

docker compose -f deploy/compose/compose.yaml --profile pooled up -d
docker compose -f deploy/compose/compose.yaml ps
curl -s localhost:8080/v1/models
```

* Other profiles: `--profile vllm` (plan with `--engine vllm`) and
  `--profile partitioned` (set `ASTRA_GPU_INFER` and `ASTRA_GPU_TRANSCODE` to UUIDs
  from `astra probe`).
* **When to pool:** if `astra plan` prints "fits on GPU n alone", prefer
  `--profile partitioned` or `--gpus n`. Pooling then only adds latency (DR-13).

## 6. Workload via systemd (hosts without Docker)
```bash
sudo editor /etc/astra/inference.env        # ASTRA_GGUF, ASTRA_CONTEXT, ASTRA_PORT, ASTRA_BINARY
sudo -u astra /opt/astra/venv/bin/astra launch --gguf "$ASTRA_GGUF" --dry-run   # review the plan
sudo systemctl enable --now astra-inference
```
`astra-inference` **will not start if `astra-selftest` failed** (FR-10).

## 7. Runtime gate (Phase 3)
Put the engine under load (for example, a loop of chat completions), then:
```bash
/opt/astra/venv/bin/astra validate --phase runtime --plan /var/lib/astra/plan.json \
    --format junit --output /var/lib/astra/runtime-gate.xml
```
R01–R05 must PASS. Record TC-RT-01/02/03.

## 8. Monitoring
Compose starts Prometheus (127.0.0.1:9090) and Grafana (127.0.0.1:3000; the
*ASTRA node* dashboard is provisioned). With systemd, `astra-exporter` serves
:9835; point an existing Prometheus at it and load `deploy/prometheus/astra-alerts.yml`.
See [monitoring.md](../07-operations/monitoring.md).

## 8a. Changing GPUs (any NVIDIA card)
1. Before buying: `astra compat --simulate "<new mix>" --driver <installed version>`.
   It must not FAIL. Note the recommended PSU.
2. Power both systems off, swap or add the card, and connect a dedicated 8-pin cable
   if the card needs one.
3. Boot, then `astra compat` → `astra probe --emit-config` → update
   `/etc/astra/astra.toml` → `astra validate` (L02, L11, L12 must pass).
4. Re-plan (`astra plan … --output --env-file`), restart the workload, and run the
   runtime gate. Record TC-HW-12.

## 10. Fabric: adding GPUs from other machines (ADR-0011)
Every machine (the ASTRA host, an old desktop, a second chassis) contributes its
NVIDIA GPUs to one pool. One machine is the **head**: it runs `llama-server` and
serves the API, and it may have no GPU of its own.

1. **On every GPU machine:**
   * Install the NVIDIA driver (use `astra compat` for the branch) and the `astra` CLI.
   * Build llama.cpp with RPC and put `rpc-server` on PATH:
     `cmake -B build -DGGML_CUDA=ON -DGGML_RPC=ON -DCMAKE_CUDA_ARCHITECTURES="<archs from astra compat>" && cmake --build build -j`
   * Give the node a unique name (`[node] name = "desktop"` in `/etc/astra/astra.toml`).
   * Start the agent: `sudo systemctl enable --now astra-agent` (or `astra agent --rpc`).
2. **Firewall (NFR-12):** allow **only from the cluster subnet**: UDP 9836
   (discovery), TCP 9837 (agent API), TCP 50052+ (one rpc-server per GPU). The
   rpc-server has no authentication, so never expose these ports beyond the LAN.
   Example (ufw): `ufw allow from 192.168.10.0/24 to any port 9836:9837 proto udp`,
   `ufw allow from 192.168.10.0/24 to any port 9837 proto tcp`,
   `ufw allow from 192.168.10.0/24 to any port 50052:50059 proto tcp`.
3. **On the head:** `astra cluster` should list every node and GPU as "rpc host:port".
   Where UDP broadcast is blocked, set `[fabric] peers = ["192.168.10.7"]` or pass
   `--peers`.
4. **Plan and launch:**
   `astra plan --cluster --gguf /srv/models/<model>.gguf --output /var/lib/astra/plan.json`,
   then `astra launch --cluster --gguf …`. The plan shows `@node` for remote GPUs and
   the generated `llama-server --rpc … --device …` command. Check device names once
   with `llama-server --rpc <endpoints> --list-devices`.
5. **Network:** 2.5 GbE or faster is recommended. The first load sends each remote
   slice over the LAN; after that the rpc-server cache (`--cache`) reloads it from
   the node's disk.
6. **Validate:** `astra validate` (link gate) on every node; the runtime gate on each
   node checks that node's GPUs (TC-RT-08).

## 11. Web console (ADR-0012)
* Try it without hardware: `astra ui --simulate fabric-demo`, then open
  http://127.0.0.1:9838/.
* On the head node: `sudo systemctl enable --now astra-ui` (localhost only,
  `--cluster`, chat → `http://127.0.0.1:8080`).
* From your laptop: `ssh -L 9838:127.0.0.1:9838 <head-node>`, then open
  http://127.0.0.1:9838/. Do not bind the console to a LAN address without an
  authenticating reverse proxy (NFR-13).

## 9. Rollback

| Change | Rollback |
|---|---|
| Control plane upgrade | `sudo /opt/astra/venv/bin/pip install astra-node==<previous>` (or check out the previous tag and reinstall); `systemctl restart astra-exporter` |
| Image / engine upgrade | Set the previous tag (`ASTRA_IMAGE`, `VLLM_VERSION`, or pin the llama.cpp image digest) in `.env`; `docker compose up -d` |
| Plan change | Restore the previous `plan.json` / `.env` (keep them under `/var/lib/astra/` with dates); `docker compose up -d` |
| Kernel arguments | Remove them from `/etc/default/grub`; `update-grub`; reboot |
| Whole node | Power off both systems; disconnect the OCuLink cable; the host runs normally without the chassis |

A rollback is rehearsed at least once before G6 and recorded in the gate log.

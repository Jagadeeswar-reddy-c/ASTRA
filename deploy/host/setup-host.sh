#!/usr/bin/env bash
# ASTRA host provisioning for Ubuntu 22.04 / 24.04 LTS.
#
# Default is a dry run that prints every action. Re-run with --apply to execute.
#
#   sudo deploy/host/setup-host.sh --apply [--disable-aspm] [--pci-realloc] [--skip-driver]
#
# Idempotent: safe to re-run after a partial failure or an upgrade.
set -euo pipefail

APPLY=0 DISABLE_ASPM=0 PCI_REALLOC=0 SKIP_DRIVER=0
for arg in "$@"; do
  case "$arg" in
    --apply) APPLY=1 ;;
    --disable-aspm) DISABLE_ASPM=1 ;;   # only if the link gate shows AER errors at idle
    --pci-realloc) PCI_REALLOC=1 ;;     # only if dmesg shows "BAR n: no space"
    --skip-driver) SKIP_DRIVER=1 ;;
    -h|--help) sed -n '2,10p' "$0"; exit 0 ;;
    *) echo "unknown option: $arg" >&2; exit 2 ;;
  esac
done

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PREFIX=/opt/astra

run() {
  echo "+ $*"
  if [[ $APPLY -eq 1 ]]; then "$@"; fi
}

step() { echo; echo "==> $*"; }

[[ $APPLY -eq 1 && $EUID -ne 0 ]] && { echo "--apply needs root (sudo)" >&2; exit 1; }
# shellcheck source=/dev/null
. /etc/os-release
[[ "${ID:-}" == "ubuntu" && "${VERSION_ID:-}" =~ ^(22.04|24.04)$ ]] \
  || echo "WARNING: tested on Ubuntu 22.04/24.04 only (found ${PRETTY_NAME:-unknown})"
[[ $APPLY -eq 0 ]] && echo "DRY RUN — nothing will change. Re-run with --apply."

step "1. NVIDIA driver"
if [[ $SKIP_DRIVER -eq 1 ]]; then
  echo "skipped (--skip-driver)"
elif command -v nvidia-smi >/dev/null && nvidia-smi >/dev/null 2>&1; then
  echo "driver already working: $(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1)"
else
  run apt-get update
  run apt-get install -y ubuntu-drivers-common
  run ubuntu-drivers install
  echo "A reboot is required after driver installation; re-run this script afterwards."
fi

step "2. NVIDIA Container Toolkit (docker must already be installed)"
if ! command -v docker >/dev/null; then
  echo "docker not found: install Docker Engine first (docs/06-deployment/deployment-guide.md §3)"
else
  if dpkg -s nvidia-container-toolkit >/dev/null 2>&1; then
    echo "nvidia-container-toolkit already installed"
  else
    run bash -c 'curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey \
      | gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg'
    run bash -c 'curl -fsSL https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list \
      | sed "s#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g" \
      > /etc/apt/sources.list.d/nvidia-container-toolkit.list'
    run apt-get update
    run apt-get install -y nvidia-container-toolkit
  fi
  # Registers the `nvidia` runtime that compose.yaml's `runtime: nvidia` relies on.
  run nvidia-ctk runtime configure --runtime=docker
  run systemctl restart docker
fi

step "3. Persistence daemon (keeps GPUs initialised; faster, stabler enumeration)"
run systemctl enable --now nvidia-persistenced

step "4. Disable system sleep (DR-05: host S3 cuts the relay -> chassis PSU -> GPUs vanish)"
run systemctl mask sleep.target suspend.target hibernate.target hybrid-sleep.target

step "5. Kernel command line"
ARGS=()
[[ $DISABLE_ASPM -eq 1 ]] && ARGS+=("pcie_aspm=off")
[[ $PCI_REALLOC -eq 1 ]] && ARGS+=("pci=realloc")
if [[ ${#ARGS[@]} -eq 0 ]]; then
  echo "no kernel arguments requested"
else
  for a in "${ARGS[@]}"; do
    if grep -q -- "$a" /etc/default/grub; then
      echo "$a already present"
    else
      run sed -i "s/^GRUB_CMDLINE_LINUX_DEFAULT=\"/&$a /" /etc/default/grub
    fi
  done
  run update-grub
  echo "Reboot required for kernel arguments to take effect."
fi

step "6. ASTRA control plane in $PREFIX"
id astra >/dev/null 2>&1 || run useradd --system --home-dir /var/lib/astra --shell /usr/sbin/nologin astra
run install -d -o root -g root "$PREFIX" /etc/astra
run install -d -o astra -g astra /var/lib/astra
run python3 -m venv "$PREFIX/venv"
run "$PREFIX/venv/bin/pip" install --upgrade "$REPO_DIR"
run cp -r "$REPO_DIR/docs" "$PREFIX/"
if [[ -f /etc/astra/astra.toml ]]; then
  echo "/etc/astra/astra.toml exists — keeping it"
else
  run install -m 0644 "$REPO_DIR/config/astra.example.toml" /etc/astra/astra.toml
fi
[[ -f /etc/astra/inference.env ]] \
  || run install -m 0644 "$REPO_DIR/deploy/systemd/inference.env.example" /etc/astra/inference.env

step "7. systemd units"
for unit in astra-selftest.service astra-exporter.service astra-inference.service astra-agent.service astra-ui.service; do
  run install -m 0644 "$REPO_DIR/deploy/systemd/$unit" /etc/systemd/system/$unit
done
run systemctl daemon-reload
run systemctl enable astra-selftest.service astra-exporter.service
echo "astra-inference.service installed but NOT enabled — enable it after Phase 3 sign-off."

step "Done"
echo "Next: sudo systemctl start astra-selftest && /opt/astra/venv/bin/astra validate"

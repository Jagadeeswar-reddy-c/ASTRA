#!/usr/bin/env bash
# Install ASTRA on Linux and (optionally) run it: one step for new users.
#
#   bash scripts/install.sh            # install, then run `astra auto`
#   bash scripts/install.sh --no-run   # install only
#   bash scripts/install.sh -- --no-ui # extra arguments after -- go to `astra auto`
set -euo pipefail

RUN=1
AUTO_ARGS=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --no-run) RUN=0; shift ;;
    --) shift; AUTO_ARGS=("$@"); break ;;
    -h|--help) sed -n '2,6p' "$0"; exit 0 ;;
    *) echo "unknown option: $1" >&2; exit 2 ;;
  esac
done

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
step() { printf '\n==> %s\n' "$*"; }
fail() { printf '\nERROR: %s\n' "$*" >&2; exit 1; }

step "Finding Python 3.11 or newer"
PY=""
for c in python3.13 python3.12 python3.11 python3; do
  if command -v "$c" >/dev/null 2>&1 && "$c" -c 'import sys; sys.exit(0 if sys.version_info >= (3, 11) else 1)'; then
    PY="$c"; break
  fi
done
[[ -n "$PY" ]] || fail "Python 3.11+ not found. Ubuntu 24.04: sudo apt install python3 python3-venv"
echo "    using: $PY ($("$PY" --version))"

step "Creating the virtual environment and installing ASTRA"
if [[ ! -x "$REPO/.venv/bin/python" ]]; then
  "$PY" -m venv "$REPO/.venv" || fail "venv failed. Ubuntu: sudo apt install python3-venv"
fi
"$REPO/.venv/bin/python" -m pip install --quiet --upgrade pip
"$REPO/.venv/bin/python" -m pip install --quiet -e "$REPO"
ASTRA="$REPO/.venv/bin/astra"
echo "    $("$ASTRA" --version)  ->  $ASTRA"

step "Checking the NVIDIA driver"
if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi --query-gpu=index,name,memory.total,driver_version --format=csv,noheader | sed 's/^/    GPU /'
else
  echo "    nvidia-smi not found: install the NVIDIA driver (sudo ubuntu-drivers install) before running ASTRA."
  RUN=0
fi
if ! command -v llama-server >/dev/null 2>&1; then
  echo "    note: llama-server not on PATH. On Linux build llama.cpp once (docs/GETTING-STARTED.md, step 2)"
  echo "          or pass --binary to astra auto."
fi

printf '\nInstalled. Use ASTRA with:\n'
echo "    $ASTRA auto          # detect GPUs, choose + download a model, launch, verify, open console"
echo "    $ASTRA --help        # all commands"
[[ $RUN -eq 1 ]] || exit 0

step "Running: astra auto ${AUTO_ARGS[*]:-}"
exec "$ASTRA" auto "${AUTO_ARGS[@]}"

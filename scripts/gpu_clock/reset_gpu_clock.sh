#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage: sudo scripts/gpu_clock/reset_gpu_clock.sh [gpu_index]

Reset one GPU's graphics clocks to defaults using nvidia-smi -rgc.

Defaults:
  gpu_index: 2

Example:
  sudo scripts/gpu_clock/reset_gpu_clock.sh
  sudo scripts/gpu_clock/reset_gpu_clock.sh 2
EOF
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

if [[ $# -gt 1 ]]; then
  usage >&2
  exit 2
fi

if [[ "${EUID}" -ne 0 ]]; then
  echo "This command resets GPU clocks and must be run with sudo." >&2
  echo "Run: sudo $0 ${1:-}" >&2
  exit 1
fi

gpu_index="${1:-2}"

if ! [[ "${gpu_index}" =~ ^[0-9]+$ ]]; then
  echo "gpu_index must be an integer." >&2
  exit 2
fi

echo "Before:"
nvidia-smi -i "${gpu_index}" \
  --query-gpu=index,name,pstate,clocks.gr,clocks.max.gr,clocks_event_reasons.applications_clocks_setting \
  --format=csv

nvidia-smi -i "${gpu_index}" -rgc

echo "After:"
nvidia-smi -i "${gpu_index}" \
  --query-gpu=index,name,pstate,clocks.gr,clocks.max.gr,clocks_event_reasons.applications_clocks_setting \
  --format=csv

echo "Reset GPU ${gpu_index} graphics clocks to defaults."

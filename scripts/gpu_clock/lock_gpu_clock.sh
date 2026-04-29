#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage: sudo scripts/gpu_clock/lock_gpu_clock.sh <clock_mhz> [gpu_index]

Lock one GPU's graphics clock to a fixed MHz using nvidia-smi -lgc.

Defaults:
  gpu_index: 2

Example:
  sudo scripts/gpu_clock/lock_gpu_clock.sh 1320
  sudo scripts/gpu_clock/lock_gpu_clock.sh 1200 2

Use scripts/gpu_clock/reset_gpu_clock.sh afterwards to restore defaults.
EOF
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

if [[ $# -lt 1 || $# -gt 2 ]]; then
  usage >&2
  exit 2
fi

if [[ "${EUID}" -ne 0 ]]; then
  echo "This command changes GPU clocks and must be run with sudo." >&2
  echo "Run: sudo $0 $*" >&2
  exit 1
fi

clock_mhz="$1"
gpu_index="${2:-2}"

if ! [[ "${clock_mhz}" =~ ^[0-9]+$ ]]; then
  echo "clock_mhz must be an integer MHz value." >&2
  exit 2
fi

if ! [[ "${gpu_index}" =~ ^[0-9]+$ ]]; then
  echo "gpu_index must be an integer." >&2
  exit 2
fi

max_clock="$(
  nvidia-smi -i "${gpu_index}" \
    --query-gpu=clocks.max.gr \
    --format=csv,noheader,nounits \
    | tr -d '[:space:]'
)"

if [[ -z "${max_clock}" ]]; then
  echo "Failed to query max graphics clock for GPU ${gpu_index}." >&2
  exit 1
fi

if (( clock_mhz >= max_clock )); then
  echo "Requested clock ${clock_mhz} MHz is not below max ${max_clock} MHz." >&2
  echo "Choose a value lower than ${max_clock} MHz, e.g. 1320." >&2
  exit 2
fi

echo "Before:"
nvidia-smi -i "${gpu_index}" \
  --query-gpu=index,name,pstate,clocks.gr,clocks.max.gr \
  --format=csv

nvidia-smi -i "${gpu_index}" -pm 1
nvidia-smi -i "${gpu_index}" -lgc "${clock_mhz},${clock_mhz}"

echo "After:"
nvidia-smi -i "${gpu_index}" \
  --query-gpu=index,name,pstate,clocks.gr,clocks.max.gr,clocks_event_reasons.applications_clocks_setting \
  --format=csv

echo "Locked GPU ${gpu_index} graphics clock to ${clock_mhz} MHz."
echo "Restore with: sudo scripts/gpu_clock/reset_gpu_clock.sh ${gpu_index}"

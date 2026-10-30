#!/usr/bin/env bash
set -euo pipefail
PROJECT_ROOT="$(realpath -- "${1:?Pass the mounted competition_local directory}")"
WORK=/home/eai/ppsuc
source "$PROJECT_ROOT/wsl_setup/network_env.sh"
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export WANDB_MODE=disabled
export OPENPI_DATA_HOME="$PROJECT_ROOT/cloud_snapshot/cache/openpi"
export PPSUC_OPENPI_ROOT="$WORK/openpi"
export PPSUC_RUN_ROOT="$WORK/runs"
export TOKENIZERS_PARALLELISM=false
GPU_SMI="$(command -v nvidia-smi || true)"
[[ -n "$GPU_SMI" ]] || GPU_SMI=/usr/lib/wsl/lib/nvidia-smi
FREE_MIB="$("$GPU_SMI" --query-gpu=memory.free --format=csv,noheader,nounits | head -1 | tr -d ' ')"
if (( FREE_MIB < ${PPSUC_MIN_FREE_MIB:-9000} )); then
  echo "Probe not started: only $FREE_MIB MiB free; keep existing user GPU programs untouched and retry after memory is available."
  exit 3
fi
exec "$WORK/.venv/bin/python" "$PROJECT_ROOT/pi05_low_memory/local_lora_probe.py" --run --steps "${2:-10}"

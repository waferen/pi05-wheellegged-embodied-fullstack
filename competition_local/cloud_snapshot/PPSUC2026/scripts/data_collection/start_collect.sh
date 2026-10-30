#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."
: "${DISPLAY:?Run in the remote desktop or set the existing display, e.g. DISPLAY=:20}"
python=/opt/conda/envs/isaaclab/bin/python
status_dir=$(mktemp -d /tmp/ppsuc-stage.XXXXXX)
export PPSUC_STAGE_STATUS="$status_dir/status.json"
window_pid=
sim_pid=
cleanup() {
  trap - EXIT INT TERM
  if [[ -n "$sim_pid" ]]; then kill "$sim_pid" 2>/dev/null || true; wait "$sim_pid" 2>/dev/null || true; fi
  if [[ -n "$window_pid" ]]; then kill "$window_pid" 2>/dev/null || true; wait "$window_pid" 2>/dev/null || true; fi
  rm -f "$status_dir/status.json" "$status_dir/status.tmp"
  rmdir "$status_dir" 2>/dev/null || true
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
"$python" scripts/data_collection/stage_window.py "$PPSUC_STAGE_STATUS" "$$" &
window_pid=$!
"$python" scripts/rsl_rl/teleop_dual_cerebellum_arm.py \
  --enable_cameras --vx 1.0 --wz 1.0 \
  --record_dir datasets/staged_task "$@" &
sim_pid=$!
result=0
wait "$sim_pid" || result=$?
sim_pid=
exit "$result"

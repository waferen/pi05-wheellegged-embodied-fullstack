#!/usr/bin/env bash
set -euo pipefail
position=4
execute_steps=25
usage() {
  echo "Usage: bash run_full.sh [--can03_position {1,2,3,4}] [--execute_steps N]"
  echo "execute_steps: actions per prediction before replanning, 1-50 (default: 25)."
}
while (( $# > 0 )); do
  case "$1" in
    --can03_position)
      if (( $# < 2 )); then usage >&2; exit 2; fi
      case "$2" in
        1|2|3|4) position=$2 ;;
        *) echo "can03_position must be 1, 2, 3 or 4" >&2; exit 2 ;;
      esac
      shift 2
      ;;
    --execute_steps)
      if (( $# < 2 )); then usage >&2; exit 2; fi
      if [[ ! "$2" =~ ^([1-9]|[1-4][0-9]|50)$ ]]; then
        echo "execute_steps must be an integer from 1 to 50" >&2
        exit 2
      fi
      execute_steps=$2
      shift 2
      ;;
    -h|--help) usage; exit 0 ;;
    *) usage >&2; exit 2 ;;
  esac
done
cd /root/gpufree-data/PPSUC2026
export DISPLAY=${DISPLAY:-:20}
export PYTHONPATH=/root/gpufree-data/openpi/packages/openpi-client/src${PYTHONPATH:+:$PYTHONPATH}
python=/opt/conda/envs/isaaclab/bin/python
status_dir=$(mktemp -d /tmp/ppsuc-full.XXXXXX)
export PPSUC_FULL_STATUS="$status_dir/status.json"
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
"$python" scripts/rsl_rl/full_status_window.py "$PPSUC_FULL_STATUS" "$$" &
window_pid=$!
"$python" scripts/rsl_rl/teleop_dual_cerebellum_arm.py --policy_full --policy_host 127.0.0.1 --execute_steps "$execute_steps" --device cuda:0 --can03_position "$position" --vx 1.0 --wz 1.0 --max_steps 0 &
sim_pid=$!
result=0
wait "$sim_pid" || result=$?
sim_pid=
exit "$result"

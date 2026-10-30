#!/usr/bin/env bash
set -euo pipefail
task_root='/mnt/d/EAI Competition Of PPSUC/competition_local'
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1
exec /home/eai/ppsuc/.venv/bin/python "$task_root/mujoco_hotel/convert_mujoco_to_lerobot.py" \
 --source "$task_root/datasets/mujoco_raw/episode_000003.hdf5" \
 --output "$task_root/datasets/mujoco_success_ep3_v1" --repo-id local/mujoco_success_ep3

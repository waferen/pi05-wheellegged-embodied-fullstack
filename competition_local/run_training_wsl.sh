#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
if [[ ! -x .venv-linux/bin/python ]]; then
  python3 -m venv .venv-linux
  .venv-linux/bin/python -m pip install -r requirements.lock.txt
fi
.venv-linux/bin/python mujoco_training/build_scene.py
if [[ -z "${DISPLAY:-}" && -z "${WAYLAND_DISPLAY:-}" ]]; then
  .venv-linux/bin/python mujoco_training/train_smoke.py --steps "${1:-65536}" --no-video
else
  .venv-linux/bin/python mujoco_training/train_smoke.py --steps "${1:-65536}"
fi

#!/usr/bin/env bash
set -euo pipefail
PROJECT_ROOT="$(realpath -- "${1:?Pass the mounted competition_local directory}")"
SETUP_ROOT="$PROJECT_ROOT/wsl_setup"
exec > >(tee -a "$SETUP_ROOT/linux_bootstrap.log") 2>&1
source "$SETUP_ROOT/network_env.sh"
phase() { printf '{"phase":"%s","updated_utc":"%s"}\n' "$1" "$(date -u +%FT%TZ)" > "$SETUP_ROOT/linux_bootstrap_status.json"; }
trap 'phase failed' ERR
[[ "$(id -u)" == 0 ]] || { echo 'Run the one-time bootstrap as root inside the dedicated WSL distribution.'; exit 1; }
[[ -f "$PROJECT_ROOT/cloud_snapshot/openpi/pyproject.toml" ]] || { echo 'Missing migrated source'; exit 1; }
phase installing_system_packages
export DEBIAN_FRONTEND=noninteractive
# The public wheel CDN was measured at only a few KiB/s on this connection.
# Keep the project-specific CPU Torch index; use the HTTPS TUNA mirror for PyPI.
export PIP_INDEX_URL="${PPSUC_PYPI_INDEX:-https://pypi.tuna.tsinghua.edu.cn/simple}"
export UV_DEFAULT_INDEX="$PIP_INDEX_URL"
apt-get update
apt-get install -y ca-certificates curl git ffmpeg libegl1 libgl1 libglfw3 libosmesa6 python3-venv python3-pip rsync build-essential pkg-config
id -u eai >/dev/null 2>&1 || useradd --create-home --shell /bin/bash eai
if [[ ! -f /etc/wsl.conf ]]; then printf '[user]\ndefault=eai\n' > /etc/wsl.conf; fi
if [[ ! -x /opt/ppsuc-tools/bin/uv ]]; then
  python3 -m venv /opt/ppsuc-tools
  /opt/ppsuc-tools/bin/python -m pip install --disable-pip-version-check uv
fi
UV=/opt/ppsuc-tools/bin/uv
WORK=/home/eai/ppsuc
mkdir -p "$WORK/openpi" "$WORK/runs"
if [[ ! -f "$WORK/openpi/pyproject.toml" ]]; then
  rsync -a --exclude=.git --exclude=.venv --exclude=wandb --exclude=checkpoints --exclude=__pycache__ "$PROJECT_ROOT/cloud_snapshot/openpi/" "$WORK/openpi/"
fi
chown -R eai:eai "$WORK"
phase preparing_python_dependencies
runuser -u eai -- "$UV" python install 3.11
PYTHON="$(runuser -u eai -- "$UV" python find 3.11)"
runuser -u eai -- "$PYTHON" "$SETUP_ROOT/configure_linux_project.py" --project "$WORK/openpi"
runuser -u eai -- env GIT_LFS_SKIP_SMUDGE=1 UV_PROJECT_ENVIRONMENT="$WORK/.venv" "$UV" sync --project "$WORK/openpi" --python 3.11 --no-dev
runuser -u eai -- "$UV" pip install --python "$WORK/.venv/bin/python" 'mujoco==3.17.0'
runuser -u eai -- "$UV" pip freeze --python "$WORK/.venv/bin/python" > "$SETUP_ROOT/linux_requirements.lock.txt"
phase checking_cuda_and_physics
runuser -u eai -- env PPSUC_ASSET_ROOT="$PROJECT_ROOT" XLA_PYTHON_CLIENT_PREALLOCATE=false "$WORK/.venv/bin/python" "$SETUP_ROOT/check_linux_gpu.py"
phase environment_ready
echo 'WSL Python/CUDA/MuJoCo checks passed. Run run_lora_probe.sh for the separate short model probe.'

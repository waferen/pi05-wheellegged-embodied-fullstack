#!/usr/bin/env bash
set -euo pipefail
if (( $# != 0 )); then echo "Usage: bash serve_full.sh" >&2; exit 2; fi
cd /root/gpufree-data/openpi
checkpoint=$(.venv/bin/python - <<'PY'
from pathlib import Path
p=Path("checkpoints/pi05_b2w_piper/run01")
candidates=[d for d in p.glob("*") if d.is_dir() and d.name.isdigit()
            and (d/"params").is_dir() and (d/"assets/b2w_piper_c4_transport/norm_stats.json").is_file()]
if not candidates:
    raise SystemExit("No completed pi05_b2w_piper/run01 checkpoint; train the two-phase policy first")
print(max(candidates,key=lambda d:int(d.name)))
PY
)
if [[ -n "$(ss -lntH 'sport = :8000')" ]]; then
  echo "Port 8000 is occupied; stop the existing policy service before starting full policy" >&2
  exit 1
fi
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
export XLA_PYTHON_CLIENT_PREALLOCATE=false
echo "Serving 14-D actions with predicted task phase; use the updated Full client." >&2
exec .venv/bin/python scripts/serve_policy.py --port 8000 policy:checkpoint --policy.config pi05_b2w_piper --policy.dir "$checkpoint"

"""Verify full-dataset coverage and the B2W-specific policy transforms."""
import copy
import json
from pathlib import Path

import numpy as np

from openpi import transforms as t
from openpi.policies import b2w_piper_full_policy as f
from openpi.policies import policy_config
from openpi.training import config
from openpi.training import data_loader

c = config.get_config("pi05_b2w_piper")
d = c.data.create(c.assets_dirs, c.model)
ds = data_loader.create_torch_dataset(d, c.model.action_horizon, c.model)
assert len(ds) == ds.meta.total_frames
assert set(ds.hf_dataset.unique("episode_index")) == set(range(ds.meta.total_episodes))
assert d.norm_stats is not None
samples = []
# Include previously held-out episodes and both ends of an action window.
first_end = int(ds.meta.episodes[0]["dataset_to_index"]) - 1
indices = sorted({0, first_end, int(ds.meta.episodes[1]["dataset_from_index"]),
                  int(ds.meta.episodes[8]["dataset_from_index"]), len(ds) // 2, len(ds) - 1})
for idx in indices:
    x = ds[idx]
    expected = ds.hf_dataset[idx]["action"].numpy()
    np.testing.assert_allclose(x["action"][0].numpy(), expected)
    raw = t.compose(d.repack_transforms.inputs)(x)
    raw = {k: (v.numpy() if hasattr(v, "numpy") else v) for k, v in raw.items()}
    y = t.compose(d.data_transforms.inputs)(copy.deepcopy(raw))
    assert y["prompt"] == f.DEFAULT_PROMPT
    y = t.Normalize(d.norm_stats, use_quantiles=True)(y)
    normalized_actions = y["actions"].copy()
    y = t.compose(d.model_transforms.inputs)(y)
    assert y["state"][13] == 2 * int(raw["observation/task_phase"]) - 1
    np.testing.assert_array_equal(y["actions"][:, :13], normalized_actions)
    y = t.Unnormalize(d.norm_stats, use_quantiles=True)(y)
    restored = t.compose(d.data_transforms.outputs)(y)
    np.testing.assert_allclose(restored["actions"][:, :13], raw["actions"], atol=2e-5, rtol=2e-5)
    np.testing.assert_allclose(restored["actions"][:, 13], raw["action_task_phase"])
    if idx in (first_end, len(ds) - 1):
        np.testing.assert_allclose(raw["actions"], np.tile(expected, (50, 1)))
    samples.append(raw)
    print("PASS real sample", idx, "episode", int(x["episode_index"]), flush=True)
p = policy_config.create_trained_policy(
    c, "/root/gpufree-data/cache/openpi/openpi-assets/checkpoints/pi05_base", norm_stats=d.norm_stats
)
for raw in samples[:2]:
    raw.pop("actions", None)
    out = p.infer(raw)["actions"]
    assert out.shape == (50, 14)
    assert np.isfinite(out).all()
    print("PASS base-weight forward", out.shape, flush=True)
report = {
    "config": c.name,
    "task_phase_rule": c.policy_metadata["task_phase_rule"],
    "episodes": ds.meta.total_episodes,
    "frames": len(ds),
    "sample_indices": indices,
    "normalization": "scripts/compute_norm_stats.py --max-frames 100000",
    "stats_anchor_frames": 100_000 // c.batch_size * c.batch_size,
    "stats_drop_last_frames": 100_000 % c.batch_size,
    "base_forward_shape": [50, 14],
}
Path("work/full_dataset_verification.json").write_text(json.dumps(report, indent=2))
print("PASS full dataset and policy adapters", report, flush=True)

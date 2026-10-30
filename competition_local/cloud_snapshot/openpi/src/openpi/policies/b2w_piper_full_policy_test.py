"""B2W full-demo contract tests; no model, tokenizer, videos, or dataset reads.

Run in the server's OpenPI environment after installing the full policy module:
    pytest -q /path/to/test_full_policy.py

The reference contracts are LiberoInputs/Outputs and ALOHA's paired
DeltaActions/AbsoluteActions (relative to the observation, not adjacent actions).
"""

import copy

import numpy as np
import pytest

from openpi import transforms
from openpi.models import model
from openpi.policies import b2w_piper_full_policy as full
from openpi.training import config

PROMPT = "Pick up both target objects and place them into the designated bin"
EXPECTED_MASK = np.array([False, False] + [True] * 6 + [False] * 5)


def observation():
    return {
        "observation/image": np.full((4, 5, 3), 11, dtype=np.uint8),
        "observation/wrist_image": np.full((4, 5, 3), 29, dtype=np.uint8),
        "observation/state": np.linspace(0.1, 1.3, 13, dtype=np.float32),
    }


def absolute_actions():
    # Each step differs: accidentally subtracting adjacent actions cannot pass.
    return (np.arange(50 * 13).reshape(50, 13) / 100).astype(np.float32)


@pytest.mark.parametrize("prompt", [None, "Climb the stairs.", "Grasp the object", PROMPT + "."])
def test_fixed_prompt_overrides_stage_language(prompt):
    raw = observation()
    if prompt is not None:
        raw["prompt"] = prompt
    result = full.B2wPiperFullInputs()(raw)
    assert full.DEFAULT_PROMPT == PROMPT
    assert str(result["prompt"]) == PROMPT
    assert not str(result["prompt"]).endswith(".")


def test_libero_image_mapping_and_complete_state_actions():
    raw = observation()
    raw["actions"] = absolute_actions()
    result = full.B2wPiperFullInputs()(copy.deepcopy(raw))
    np.testing.assert_array_equal(result["state"], raw["observation/state"])
    np.testing.assert_array_equal(result["actions"], raw["actions"])
    np.testing.assert_array_equal(result["image"]["base_0_rgb"], raw["observation/image"])
    np.testing.assert_array_equal(result["image"]["left_wrist_0_rgb"], raw["observation/wrist_image"])
    np.testing.assert_array_equal(result["image"]["right_wrist_0_rgb"], np.zeros((4, 5, 3), np.uint8))
    assert bool(result["image_mask"]["base_0_rgb"])
    assert bool(result["image_mask"]["left_wrist_0_rgb"])
    assert not bool(result["image_mask"]["right_wrist_0_rgb"])


@pytest.mark.parametrize("shape", [(12,), (14,), (1, 13)])
def test_reject_wrong_state_shape(shape):
    raw = observation()
    raw["observation/state"] = np.zeros(shape, dtype=np.float32)
    with pytest.raises(ValueError, match="Expected finite full"):
        full.B2wPiperFullInputs()(raw)


@pytest.mark.parametrize("shape", [(50, 12), (50, 14), (13,), (1, 50, 13)])
def test_reject_wrong_action_shape(shape):
    raw = observation()
    raw["actions"] = np.zeros(shape, dtype=np.float32)
    with pytest.raises(ValueError, match="Expected finite full"):
        full.B2wPiperFullInputs()(raw)


@pytest.mark.parametrize("value", [np.nan, np.inf, -np.inf])
@pytest.mark.parametrize("field", ["observation/state", "actions"])
def test_reject_nonfinite_state_and_actions(field, value):
    raw = observation()
    raw["actions"] = absolute_actions()
    raw[field].flat[0] = value
    with pytest.raises(ValueError, match="Expected finite full"):
        full.B2wPiperFullInputs()(raw)


def test_aloha_delta_roundtrip_changes_only_arm_joints():
    np.testing.assert_array_equal(full.JOINT_DELTA_MASK, EXPECTED_MASK)
    state = observation()["observation/state"]
    actions = absolute_actions()
    data = {"state": state.copy(), "actions": actions.copy()}
    delta = transforms.DeltaActions(full.JOINT_DELTA_MASK)(data)
    np.testing.assert_allclose(delta["actions"][:, 2:8], actions[:, 2:8] - state[None, 2:8])
    np.testing.assert_array_equal(delta["actions"][:, ~EXPECTED_MASK], actions[:, ~EXPECTED_MASK])
    np.testing.assert_array_equal(delta["state"], state)
    restored = transforms.AbsoluteActions(full.JOINT_DELTA_MASK)(delta)
    np.testing.assert_allclose(restored["actions"], actions, rtol=1e-6, atol=1e-6)


def test_outputs_crop_padding_and_preserve_all_13_channels():
    padded = np.arange(50 * 32, dtype=np.float32).reshape(50, 32)
    result = full.B2wPiperFullOutputs()({"actions": padded})
    assert result["actions"].shape == (50, 13)
    np.testing.assert_array_equal(result["actions"], padded[:, :13])


@pytest.fixture
def full_data_config(monkeypatch):
    # The real factory normally constructs a tokenizer; this test concerns only
    # B2W's data adapters and does not instantiate or download the tokenizer.
    monkeypatch.setattr(config.ModelTransformFactory, "__call__", lambda self, model_config: transforms.Group())
    monkeypatch.setattr(config.DataConfigFactory, "_load_norm_stats", lambda *args, **kwargs: None)
    train = config.get_config("pi05_b2w_piper")
    return train, train.data.create(train.assets_dirs, train.model)


def test_full_config_keeps_base_model_dimensions_and_fixed_language(full_data_config):
    train, data = full_data_config
    assert train.name == "pi05_b2w_piper"
    assert train.model.model_type == model.ModelType.PI05
    assert train.model.action_dim == 32
    assert train.model.action_horizon == 50
    assert data.prompt_from_task is False
    assert str(data.local_root).rstrip("/") == "/root/gpufree-data/lerobot_datasets/b2w_piper_c4_transport"
    assert any(isinstance(step, full.B2wPiperTwoPhaseInputs) for step in data.data_transforms.inputs)
    assert any(isinstance(step, transforms.DeltaActions) for step in data.data_transforms.inputs)
    assert any(isinstance(step, transforms.AbsoluteActions) for step in data.data_transforms.outputs)
    assert any(isinstance(step, full.B2wPiperTwoPhaseOutputs) for step in data.data_transforms.outputs)


def test_config_transform_order_restores_physical_actions_and_decodes_phase(full_data_config):
    _, data = full_data_config
    raw = {**observation(), "observation/task_phase": 0, "action_task_phase": np.ones(50)}
    raw["actions"] = absolute_actions()
    raw["prompt"] = "Approach the stairs and climb to the upper landing."
    transformed = copy.deepcopy(raw)
    for step in data.data_transforms.inputs:
        transformed = step(transformed)
    assert str(transformed["prompt"]) == PROMPT
    expected = raw["actions"].copy()
    expected[:, 2:8] -= raw["observation/state"][None, 2:8]
    np.testing.assert_allclose(transformed["actions"], expected)
    # Match the model's 32-D output after Unnormalize, including padding.
    transformed["actions"] = np.pad(transformed["actions"], ((0, 0), (0, 19)))
    transformed["state"] = np.pad(transformed["state"], (0, 19))
    transformed["actions"][:, 13] = 1  # Encoded next-step phase; unaffected by joint conversion.
    for step in data.data_transforms.outputs:
        transformed = step(transformed)
    assert transformed["actions"].shape == (50, 14)
    np.testing.assert_allclose(transformed["actions"][:, :13], raw["actions"], rtol=1e-6, atol=1e-6)
    np.testing.assert_array_equal(transformed["actions"][:, 13], np.ones(50))




@pytest.fixture
def phase_transform(tmp_path):
    import json
    path = tmp_path / "phase_boundaries.json"
    path.write_text(json.dumps({
        "version": 1, "repo_id": "test", "stage_id": 8, "offset_seconds": 2.0,
        "total_episodes": 2, "total_frames": 32,
        "episodes": [
            {"episode_index": 0, "length": 12, "phase1_start_frame": 5},
            {"episode_index": 1, "length": 20, "phase1_start_frame": 13},
        ],
    }))
    return full.PhaseFromFrame(str(path), "test")


@pytest.mark.parametrize("episode,boundary", [(0, 5), (1, 13)])
def test_phase_uses_current_local_frame_not_future_action_chunk(phase_transform, episode, boundary):
    for frame, phase in [(boundary - 1, 0), (boundary, 1), (boundary + 1, 1)]:
        sample = {"episode_index": np.int64(episode), "frame_index": np.int64(frame),
                  "actions": absolute_actions()}
        result = phase_transform(sample)
        assert result["observation/task_phase"] == phase
        assert "episode_index" not in result and "frame_index" not in result
        np.testing.assert_array_equal(result["actions"], sample["actions"])
        assert sample["frame_index"] == frame


@pytest.mark.parametrize("episode,frame", [(2, 0), (0, -1), (0, 12), (1, 20), (0, 4.5)])
def test_phase_rejects_unknown_episode_or_invalid_frame(phase_transform, episode, frame):
    with pytest.raises(ValueError):
        phase_transform({"episode_index": episode, "frame_index": frame})


@pytest.mark.parametrize("phase", [-1, 2, np.nan, [0, 1]])
def test_two_phase_rejects_invalid_phase(phase):
    with pytest.raises(ValueError, match="task_phase"):
        full.B2wPiperTwoPhaseInputs()({**observation(), "observation/task_phase": phase})


def test_two_phase_requires_explicit_phase():
    with pytest.raises(KeyError, match="task_phase"):
        full.B2wPiperTwoPhaseInputs()(observation())


@pytest.mark.parametrize("phase", [0, 1])
def test_two_phase_normalization_and_joint_delta_roundtrip(phase):
    raw = {**observation(), "observation/task_phase": phase, "actions": absolute_actions(),
           "action_task_phase": np.r_[np.zeros(25), np.ones(25)]}
    data = full.B2wPiperTwoPhaseInputs()(copy.deepcopy(raw))
    assert data["prompt"] == PROMPT
    assert data["state"].shape == (13,)  # Also the input seen by compute_norm_stats.
    data = transforms.DeltaActions(full.JOINT_DELTA_MASK)(data)
    stats = {key: transforms.NormStats(
        mean=np.zeros(13), std=np.ones(13), q01=np.full(13, -2.), q99=np.full(13, 10.),
    ) for key in ("state", "actions")}
    data = transforms.Normalize(stats, use_quantiles=True)(data)
    physical = data["state"].copy()
    normalized_actions = data["actions"].copy()
    augmented = full.AppendTaskPhase()(data)
    assert augmented["state"].shape == (14,)
    np.testing.assert_array_equal(augmented["state"][:13], physical)
    assert augmented["state"][13] == 2 * phase - 1
    assert data["state"].shape == (13,)  # Input was not mutated.
    np.testing.assert_array_equal(augmented["actions"][:, :13], normalized_actions)
    np.testing.assert_array_equal(augmented["actions"][:, 13], 2 * raw["action_task_phase"] - 1)
    padded = transforms.PadStatesAndActions(32)(augmented)
    restored = transforms.Unnormalize(stats, use_quantiles=True)(padded)
    restored = transforms.AbsoluteActions(full.JOINT_DELTA_MASK)(restored)
    output = full.B2wPiperTwoPhaseOutputs()(restored)
    assert output["actions"].shape == (50, 14)
    np.testing.assert_allclose(output["actions"][:, :13], raw["actions"], atol=2e-6)
    np.testing.assert_array_equal(output["actions"][:, 13], raw["action_task_phase"])


def test_only_one_b2w_config(monkeypatch, tmp_path):
    monkeypatch.setattr(config.ModelTransformFactory, "__call__", lambda self, model_config: transforms.Group())
    monkeypatch.setattr(config.DataConfigFactory, "_load_norm_stats", lambda *args, **kwargs: None)
    names = [item.name for item in config._CONFIGS if "b2w_piper" in item.name]
    assert names == ["pi05_b2w_piper"]
    for suffix in ("_full", "_two_phase"):
        with pytest.raises(ValueError, match="not found"):
            config.get_config("pi05_b2w_piper" + suffix)
    train = config.get_config("pi05_b2w_piper")
    data = train.data.create(tmp_path, train.model)
    assert train.policy_metadata["required_phase_key"] == "observation/task_phase"
    assert train.policy_metadata["action_schema"] != "b2w_piper_full"
    assert train.policy_metadata["action_dim"] == 14
    assert train.policy_metadata["phase_action_semantics"] == "next_step_latched_v1"
    assert train.model.action_dim == 32 and train.model.action_horizon == 50
    assert data.prompt_from_task is False
    assert isinstance(data.data_transforms.inputs[0], full.B2wPiperTwoPhaseInputs)
    assert isinstance(data.model_transforms.inputs[0], full.AppendTaskPhase)
    assert "frame_index" in data.repack_transforms.inputs[0].structure


@pytest.mark.parametrize("episode,boundary,length", [(0, 5, 12), (1, 13, 20)])
def test_phase_action_targets_next_observation_and_pads_within_episode(phase_transform, episode, boundary, length):
    for frame in (0, boundary - 2, boundary - 1, boundary, length - 1):
        sample = {"episode_index": episode, "frame_index": frame, "actions": absolute_actions()}
        result = phase_transform(sample)
        assert result["observation/task_phase"] == int(frame >= boundary)
        expected = [int(min(frame + j + 1, length - 1) >= boundary) for j in range(50)]
        np.testing.assert_array_equal(result["action_task_phase"], expected)
        if frame == boundary - 1:
            assert result["observation/task_phase"] == 0
            assert result["action_task_phase"][0] == 1
        if frame == boundary - 2:
            np.testing.assert_array_equal(result["action_task_phase"][:2], [0, 1])


@pytest.mark.parametrize("phases", [np.zeros(49), np.zeros((50, 1)), np.full(50, 2), np.full(50, np.nan)])
def test_reject_invalid_action_phase_targets(phases):
    data = {"state": np.zeros(13), "task_phase": 0,
            "actions": np.zeros((50, 13)), "action_task_phase": phases}
    with pytest.raises(ValueError, match="action_task_phase"):
        full.AppendTaskPhase()(data)


def test_action_phase_supervision_is_required_when_training():
    with pytest.raises(KeyError, match="action_task_phase"):
        full.B2wPiperTwoPhaseInputs()({**observation(), "observation/task_phase": 0,
                                     "actions": absolute_actions()})

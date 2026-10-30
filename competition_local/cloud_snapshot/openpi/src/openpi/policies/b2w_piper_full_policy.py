"""Full-demo policy: LIBERO camera adapter and ALOHA joint-delta transforms."""
import dataclasses
import functools
import json
from pathlib import Path

import numpy as np

from openpi import transforms
from openpi.models import model
from openpi.policies import libero_policy

DEFAULT_PROMPT = "Pick up both target objects and place them into the designated bin"
ACTION_DIM = 13  # Physical channels in the original dataset.
POLICY_ACTION_DIM = 14  # Physical controls plus next-step task phase.
JOINT_DELTA_MASK = transforms.make_bool_mask(-2, 6, -5)

@dataclasses.dataclass(frozen=True)
class B2wPiperFullInputs(transforms.DataTransformFn):
    def __call__(self, data: dict) -> dict:
        state = np.asarray(data["observation/state"], dtype=np.float32)
        if state.shape != (ACTION_DIM,) or not np.isfinite(state).all():
            raise ValueError("Expected finite full state[13]")
        inputs = {**data, "observation/state": state, "prompt": DEFAULT_PROMPT}
        if "actions" in inputs:
            actions = np.asarray(inputs["actions"], dtype=np.float32)
            if actions.ndim != 2 or actions.shape[-1] != ACTION_DIM or not np.isfinite(actions).all():
                raise ValueError("Expected finite full actions[T,13]")
            inputs["actions"] = actions
        # Override even an explicitly supplied legacy stage prompt.
        return libero_policy.LiberoInputs(model_type=model.ModelType.PI05)(inputs)

@dataclasses.dataclass(frozen=True)
class B2wPiperFullOutputs(transforms.DataTransformFn):
    def __call__(self, data: dict) -> dict:
        return {"actions": np.asarray(data["actions"])[..., :ACTION_DIM]}


@dataclasses.dataclass(frozen=True)
class PhaseFromFrame(transforms.DataTransformFn):
    """Current observation phase and next-step phase targets for each action."""

    boundaries_path: str
    repo_id: str

    @functools.cached_property
    def boundaries(self) -> dict[int, tuple[int, int]]:
        manifest = json.loads(Path(self.boundaries_path).read_text())
        if (manifest["version"] != 1 or manifest["repo_id"] != self.repo_id
                or manifest["stage_id"] != 8 or manifest["offset_seconds"] != 2.0):
            raise ValueError("Incompatible two-phase boundary manifest")
        result = {}
        for item in manifest["episodes"]:
            episode, length, start = item["episode_index"], item["length"], item["phase1_start_frame"]
            if episode in result or not 0 < start < length:
                raise ValueError(f"Invalid phase boundary for episode {episode}")
            result[episode] = (start, length)
        if (len(result) != manifest["total_episodes"]
                or sum(length for _, length in result.values()) != manifest["total_frames"]):
            raise ValueError("Incomplete two-phase boundary manifest")
        return result

    def __call__(self, data: dict) -> dict:
        result = dict(data)
        episode = np.asarray(result.pop("episode_index")).item()
        frame = np.asarray(result.pop("frame_index")).item()
        if episode not in self.boundaries:
            raise ValueError(f"No phase boundary for episode {episode}")
        start, length = self.boundaries[episode]
        if not np.isfinite(frame) or int(frame) != frame or not 0 <= frame < length:
            raise ValueError(f"Invalid frame {frame} for episode {episode}")
        result["observation/task_phase"] = np.int32(frame >= start)
        if "actions" in result:
            actions = np.asarray(result["actions"])
            if actions.ndim != 2 or actions.shape[-1] != ACTION_DIM:
                raise ValueError("Expected physical actions[T,13] for phase labels")
            # LeRobot queries actions at t + j / fps. Each phase command is the
            # phase AFTER that action; clip within this episode like action padding.
            next_frames = np.minimum(int(frame) + np.arange(len(actions)) + 1, length - 1)
            result["action_task_phase"] = (next_frames >= start).astype(np.int32)
        return result


def _task_phase(value) -> int:
    phase = np.asarray(value)
    if phase.size != 1 or phase.item() not in (0, 1):
        raise ValueError("task_phase must be a scalar 0 or 1")
    return int(phase.item())


@dataclasses.dataclass(frozen=True)
class B2wPiperTwoPhaseInputs(transforms.DataTransformFn):
    def __call__(self, data: dict) -> dict:
        # LiberoInputs returns only its known fields, so restore the phase explicitly.
        phase = _task_phase(data["observation/task_phase"])
        result = {**B2wPiperFullInputs()(data), "task_phase": np.int32(phase)}
        if "actions" in result:
            result["action_task_phase"] = np.asarray(data["action_task_phase"])
        return result


@dataclasses.dataclass(frozen=True)
class AppendTaskPhase(transforms.DataTransformFn):
    """Run after physical normalization and before pi05 state tokenization."""

    def __call__(self, data: dict) -> dict:
        result = dict(data)
        phase = _task_phase(result.pop("task_phase"))
        state = np.asarray(result["state"])
        if state.shape != (ACTION_DIM,):
            raise ValueError("Expected normalized physical state[13] before adding task_phase")
        result["state"] = np.concatenate([state, np.array([2 * phase - 1], dtype=state.dtype)])
        if "actions" in result:
            actions = np.asarray(result["actions"])
            phases = np.asarray(result.pop("action_task_phase"))
            if (actions.ndim != 2 or actions.shape[-1] != ACTION_DIM
                    or phases.shape != (len(actions),) or not np.isin(phases, (0, 1)).all()):
                raise ValueError("Expected actions[T,13] and binary action_task_phase[T]")
            encoded = (2 * phases - 1).astype(actions.dtype)
            result["actions"] = np.concatenate([actions, encoded[:, None]], axis=-1)
        return result


@dataclasses.dataclass(frozen=True)
class B2wPiperTwoPhaseOutputs(transforms.DataTransformFn):
    def __call__(self, data: dict) -> dict:
        actions = np.asarray(data["actions"])[..., :POLICY_ACTION_DIM].copy()
        # Official quantile Unnormalize leaves channels beyond the 13-D stats
        # unchanged. Decode this fixed -1/+1 channel to a 0/1 phase score.
        actions[..., ACTION_DIM] = (actions[..., ACTION_DIM] + 1) / 2
        return {"actions": actions}

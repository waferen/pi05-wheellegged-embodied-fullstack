"""Pure client contract checks: no simulator and no network connection."""
import sys
from types import SimpleNamespace as NS
from unittest.mock import patch
import numpy as np
import torch
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from full_client import FullClient, decode_action, PROMPT
class Fake:
    def __init__(self): self.observations=[]
    def infer(self,obs):
        self.observations.append(obs)
        a=np.zeros((50,14)); a[:,9]=1; a[:,0]=.2
        a[:,2]=100*len(self.observations)+np.arange(50)
        return {"actions":a}
    def reset(self): pass
    def close(self): pass
robot=NS(root_lin_vel_b=torch.zeros((1,3)),root_ang_vel_b=torch.zeros((1,3)),
         joint_pos=torch.tensor([[.1,.2,.3,.4,.5,.6,.023,-.023]]))
camera=NS(data=NS(output={"rgb":torch.zeros((1,48,64,3),dtype=torch.uint8)}))
env=NS(scene={"robot":NS(data=robot),"front_rgb":camera,"wrist_rgb":camera})
for requested in (None, 1, 5, 10, 25, 50):
    execute_steps = 25 if requested is None else requested
    kwargs = {} if requested is None else {"execute_steps": requested}
    f = Fake()
    with patch("full_client.RemoteFullPolicy", return_value=f):
        p = FullClient("unused", 0, list(range(8)), **kwargs)
    try:
        for i in range(2*execute_steps+1):
            robot.root_lin_vel_b[0,0]=i
            _, _, joints, _ = p.step(env,"walk")
            assert joints[0] == 100*(i//execute_steps+1)+i%execute_steps
        assert len(f.observations)==3
        assert [x["observation/state"][0] for x in f.observations]==[0,execute_steps,2*execute_steps]
        assert all(x["prompt"]==PROMPT for x in f.observations)
        np.testing.assert_allclose(f.observations[0]["observation/state"][2:9],[.1,.2,.3,.4,.5,.6,.5])
    finally:
        p.close()
for bad in (0, -1, 51, 1.5, "10", True, None):
    with patch("full_client.RemoteFullPolicy") as remote:
        try: FullClient("unused", 0, list(range(8)), execute_steps=bad)
        except ValueError: pass
        else: raise AssertionError("Invalid execute_steps accepted")
        remote.assert_not_called()
for i,mode in enumerate(("walk","stair","low_grasp","high_grasp")):
    a=np.zeros(14); a[:2]=[2,-2]; a[8]=2; a[9+i]=1
    cmd,m,j,g=decode_action(a)
    assert m==mode and g==1 and j.shape==(6,)
    assert cmd==((1.,0.,-1.) if i<2 else (0.,0.,0.))
for bad in (np.zeros(4),np.full(13,np.nan),np.full(13,np.inf)):
    try: decode_action(bad)
    except ValueError: pass
    else: raise AssertionError("Invalid action accepted")
print("PASS full 13D observation; gripper units; fixed prompt; configurable execution cadence (default 25; 1/5/10/25/50); invalid execution steps; four modes; limits; invalid actions")


class PhaseFake(Fake):
    def __init__(self, switch_index=2):
        super().__init__()
        self.switch_index = switch_index
        self.resets = 0
    def infer(self, obs):
        result = super().infer(obs)
        if len(self.observations) == 1:
            result["actions"][self.switch_index:, 13] = 1
        # Later predictions of 0 must not regress an already latched phase.
        return result
    def reset(self):
        self.resets += 1

f = PhaseFake()
with patch("full_client.RemoteFullPolicy", return_value=f):
    p = FullClient("unused", 0, list(range(8)), execute_steps=25)
try:
    for expected in (0, 0, 1, 1):
        p.step(env, "walk")
        assert p.task_phase == expected
    assert len(f.observations) == 2
    assert [int(x["observation/task_phase"]) for x in f.observations] == [0, 1]
    assert f.resets == 1
    p.reset()
    assert p.task_phase == 0
    p.step(env, "walk")
    assert int(f.observations[-1]["observation/task_phase"]) == 0
finally:
    p.close()

# A switch in a discarded part of the chunk must never update the phase.
f = PhaseFake(switch_index=30)
with patch("full_client.RemoteFullPolicy", return_value=f):
    p = FullClient("unused", 0, list(range(8)), execute_steps=25)
try:
    for _ in range(51):
        p.step(env, "walk")
        assert p.task_phase == 0
    assert f.resets == 0
finally:
    p.close()
print("PASS phase execution timing, feedback, monotonic latch, discarded predictions and reset")


CONTROL_MODES = ("walk", "stair", "low_grasp", "high_grasp")


class ControlFake(Fake):
    """Predict identifiable chunks and optionally switch ownership during infer."""

    def __init__(self, mode="walk", phase=0):
        super().__init__()
        self.mode = mode
        self.phase = phase
        self.resets = 0
        self.on_infer = None
        self.bad_index = None
        self.bad_mode = None

    def infer(self, obs):
        result = super().infer(obs)
        actions = result["actions"]
        actions[:, 9:13] = 0
        actions[:, 9 + CONTROL_MODES.index(self.mode)] = 1
        actions[:, 13] = self.phase
        if self.bad_index is not None:
            actions[self.bad_index, 9:13] = 0
            actions[self.bad_index, 9 + CONTROL_MODES.index(self.bad_mode)] = 1
            actions[self.bad_index, 13] = 1
        callback, self.on_infer = self.on_infer, None
        if callback is not None:
            callback()
        return result

    def reset(self):
        self.resets += 1


def control_client(fake, execute_steps=25):
    with patch("full_client.RemoteFullPolicy", return_value=fake):
        return FullClient("unused", 0, list(range(8)), execute_steps=execute_steps)


# Ownership changes preserve both phase values, discard queued commands, and
# resuming uses the observation at resume time rather than the pre-handoff one.
for initial_phase in (0, 1):
    f = ControlFake(phase=initial_phase)
    p = control_client(f)
    try:
        assert p.model_control is True
        p.step(env, "walk")
        p.step(env, "walk")  # Leave an unconsumed suffix even after a phase latch.
        assert p.task_phase == initial_phase
        calls_before = len(f.observations)

        p.set_model_control(False)
        assert p.model_control is False
        resets_after_handoff = f.resets
        p.set_model_control(False)
        assert f.resets == resets_after_handoff
        assert p.task_phase == initial_phase
        # No scene at all: manual control must not read state, cameras or infer.
        assert p.step(NS(), "walk") is None
        assert len(f.observations) == calls_before

        robot.root_lin_vel_b[0, 0] = 700 + initial_phase
        camera.data.output["rgb"].fill_(71 + initial_phase)
        p.set_model_control(True)
        assert p.model_control is True
        assert p.task_phase == initial_phase
        result = p.step(env, "walk")
        assert result is not None
        assert result[2][0] == 100 * (calls_before + 1)
        assert len(f.observations) == calls_before + 1
        latest = f.observations[-1]
        assert latest["observation/state"][0] == 700 + initial_phase
        assert int(latest["observation/task_phase"]) == initial_phase
        for key in ("observation/image", "observation/wrist_image"):
            assert np.all(latest[key] == 71 + initial_phase)

        # Repeated F5 is a no-op, including keeping the pending action suffix.
        resets_before_repeat = f.resets
        p.set_model_control(True)
        assert f.resets == resets_before_repeat
        assert p.step(env, "walk")[2][0] == 100 * (calls_before + 1) + 1
        assert len(f.observations) == calls_before + 1

        f.phase = 0
        p.reset()
        assert p.task_phase == 0
        p.step(env, "walk")
        assert int(f.observations[-1]["observation/task_phase"]) == 0
    finally:
        p.close()
        camera.data.output["rgb"].zero_()


# An in-flight result belongs to the old owner even if F5 restores model
# control before infer returns. It must neither execute nor latch its phase.
for resume_during_infer in (False, True):
    f = ControlFake(phase=1)
    p = control_client(f)
    try:
        robot.root_lin_vel_b[0, 0] = 800

        def switch_during_infer():
            p.set_model_control(False)
            robot.root_lin_vel_b[0, 0] = 801
            camera.data.output["rgb"].fill_(81)
            if resume_during_infer:
                p.set_model_control(True)

        f.on_infer = switch_during_infer
        assert p.step(env, "walk") is None
        assert p.task_phase == 0
        assert p.model_control is resume_during_infer
        assert len(f.observations) == 1
        if not resume_during_infer:
            assert p.step(NS(), "walk") is None
            p.set_model_control(True)

        result = p.step(env, "walk")
        assert result is not None and result[2][0] == 200
        assert len(f.observations) == 2
        latest = f.observations[-1]
        assert latest["observation/state"][0] == 801
        assert int(latest["observation/task_phase"]) == 0
        assert np.all(latest["observation/image"] == 81)
        assert p.task_phase == 1  # Only the fresh result may now advance it.
    finally:
        p.close()
        camera.data.output["rgb"].zero_()

# Repeating F5 during an inference does not create a new ownership generation.
f = ControlFake()
p = control_client(f)
try:
    f.on_infer = lambda: p.set_model_control(True)
    assert p.step(env, "walk")[2][0] == 100
    assert p.step(env, "walk")[2][0] == 101
    assert len(f.observations) == 1
    assert f.resets == 0
finally:
    p.close()

print("PASS model/manual handoff, idempotence, phase preservation, fresh observations and in-flight ownership generations")


# F2 corrections are manual-only, can move in either direction, and carry the
# selected phase into a fresh prediction after model control resumes.
for initial_phase in (0, 1):
    f = ControlFake(phase=initial_phase)
    p = control_client(f)
    try:
        p.task_phase = initial_phase
        assert p.step(env, "walk")[2][0] == 100
        generation, resets = p._control_generation, f.resets
        assert p.toggle_task_phase() is False
        assert p.task_phase == initial_phase
        assert p._control_generation == generation and f.resets == resets
        assert p.step(env, "walk")[2][0] == 101  # Rejected F2 preserves cache.
        assert len(f.observations) == 1

        p.set_model_control(False)
        for expected in (1 - initial_phase, initial_phase, 1 - initial_phase):
            generation, resets = p._control_generation, f.resets
            assert p.toggle_task_phase() is True
            assert p.task_phase == expected
            assert p._control_generation == generation + 1
            assert f.resets == resets + 1
            assert p.step(NS(), "walk") is None
            assert len(f.observations) == 1

        selected_phase = p.task_phase
        robot.root_lin_vel_b[0, 0] = 900 + initial_phase
        camera.data.output["rgb"].fill_(91 + initial_phase)
        f.phase = 0  # A lower prediction cannot undo a manually selected phase 2.
        p.set_model_control(True)
        assert p.task_phase == selected_phase
        assert p.step(env, "walk")[2][0] == 200
        latest = f.observations[-1]
        assert int(latest["observation/task_phase"]) == selected_phase
        assert latest["observation/state"][0] == 900 + initial_phase
        assert np.all(latest["observation/image"] == 91 + initial_phase)
        assert p.task_phase == selected_phase
    finally:
        p.close()
        camera.data.output["rgb"].zero_()

# Correcting phase 2 back to phase 1 does not disable subsequent automatic
# advancement from a newly inferred, accepted action.
f = ControlFake(phase=1)
p = control_client(f)
try:
    p.step(env, "walk")
    assert p.task_phase == 1
    p.set_model_control(False)
    assert p.toggle_task_phase() is True and p.task_phase == 0
    p.set_model_control(True)
    assert p.step(env, "walk")[2][0] == 200
    assert int(f.observations[-1]["observation/task_phase"]) == 0
    assert p.task_phase == 1
finally:
    p.close()

# A result requested before a manual correction must neither execute nor
# overwrite that correction, even when model control has already resumed.
f = ControlFake(phase=1)
p = control_client(f)
try:
    p.task_phase = 1

    def correct_phase_during_infer():
        p.set_model_control(False)
        assert p.toggle_task_phase() is True
        p.set_model_control(True)

    f.on_infer = correct_phase_during_infer
    assert p.step(env, "walk") is None
    assert p.model_control is True and p.task_phase == 0
    f.phase = 0
    assert p.step(env, "walk")[2][0] == 200
    assert [int(obs["observation/task_phase"]) for obs in f.observations] == [1, 0]
    assert p.task_phase == 0
finally:
    p.close()

print("PASS manual phase toggle, model-mode rejection, cache invalidation, fresh phase feedback and stale-result rejection")


# Cover every current/predicted mode pair, both latched phases, and values on
# either side of the phase threshold. Rejected rows must reject ALL channels.
for current_mode in CONTROL_MODES:
    for predicted_mode in CONTROL_MODES:
        forbidden = current_mode in ("low_grasp", "high_grasp") and predicted_mode not in ("walk", current_mode)
        for initial_phase in (0, 1):
            for predicted_phase in (0, 0.499, 0.5, 1):
                f = ControlFake(mode=predicted_mode, phase=predicted_phase)
                p = control_client(f)
                try:
                    p.task_phase = initial_phase
                    result = p.step(env, current_mode)
                    if forbidden:
                        assert result is None, (current_mode, predicted_mode)
                        assert p.task_phase == initial_phase
                        # A subsequent legal prediction must come from a new
                        # chunk, not another row of the rejected old chunk.
                        f.mode, f.phase = current_mode, 0
                        result = p.step(env, current_mode)
                        assert result is not None and result[2][0] == 200
                        assert len(f.observations) == 2
                        assert int(f.observations[-1]["observation/task_phase"]) == initial_phase
                        assert p.task_phase == initial_phase
                    else:
                        assert result is not None and result[1] == predicted_mode
                        assert result[2][0] == 100
                        assert p.task_phase == max(initial_phase, int(predicted_phase >= 0.5))
                finally:
                    p.close()


# Reject illegal rows even at the final executed step, but ignore illegal
# future predictions that fall outside the selected execution horizon.
for current_mode, forbidden_modes in (
    ("low_grasp", ("stair", "high_grasp")),
    ("high_grasp", ("stair", "low_grasp")),
):
    for forbidden_mode in forbidden_modes:
        for execute_steps in (1, 5, 25, 50):
            f = ControlFake(mode=current_mode)
            f.bad_index, f.bad_mode = execute_steps - 1, forbidden_mode
            p = control_client(f, execute_steps)
            try:
                for i in range(execute_steps - 1):
                    assert p.step(env, current_mode)[2][0] == 100 + i
                    assert p.task_phase == 0
                assert p.step(env, current_mode) is None
                assert p.task_phase == 0
                assert len(f.observations) == 1
                f.bad_index = None
                assert p.step(env, current_mode)[2][0] == 200
                assert len(f.observations) == 2
                assert p.task_phase == 0
            finally:
                p.close()

        for execute_steps in (1, 5, 25):
            f = ControlFake(mode=current_mode)
            f.bad_index, f.bad_mode = execute_steps, forbidden_mode
            p = control_client(f, execute_steps)
            try:
                for i in range(2 * execute_steps + 1):
                    result = p.step(env, current_mode)
                    assert result is not None
                    assert result[2][0] == 100 * (i // execute_steps + 1) + i % execute_steps
                    assert p.task_phase == 0
                assert len(f.observations) == 3
                assert f.resets == 0
            finally:
                p.close()

print("PASS all mode transitions, whole-row rejection, phase thresholds, chunk invalidation and execution-window boundaries")

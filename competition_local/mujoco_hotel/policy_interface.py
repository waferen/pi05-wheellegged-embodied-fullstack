"""Source FullClient contract with inference isolated from main-thread execution.

Only the remote call runs in a worker. Buffers, control generations and phase
feedback belong to the simulator thread; an abandoned response cannot latch phase.
"""
import sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
for path in ('pi05_client_deps', 'cloud_snapshot/openpi/packages/openpi-client/src',
             'cloud_snapshot/PPSUC2026/scripts/rsl_rl'):
    sys.path.insert(0, str(ROOT / path))
from full_client import RemoteFullPolicy, PROMPT, MODES, decode_action


class PolicyInterface:
    def __init__(self, host='127.0.0.1', port=8765, execute_steps=25, remote=None):
        if isinstance(execute_steps, bool) or not isinstance(execute_steps, int) or not 1 <= execute_steps <= 50:
            raise ValueError('execute_steps must be an integer from 1 to 50')
        self.remote = remote if remote is not None else RemoteFullPolicy(host, port)
        self.action_horizon = execute_steps
        self.task_phase = 0
        self.model_control = True
        self.generation = 0
        self.actions = []

    def invalidate(self):
        self.generation += 1
        self.actions = []

    def set_model_control(self, enabled):
        enabled = bool(enabled)
        if enabled != self.model_control:
            self.model_control = enabled
            self.invalidate()

    def toggle_task_phase(self):
        if not self.model_control:
            self.task_phase = 1 - self.task_phase
            self.invalidate()

    def reset(self):
        self.task_phase = 0
        self.invalidate()

    def observation(self, env, mode):
        d = env.scene['robot'].data
        q = d.joint_pos[0].detach().cpu().numpy()
        state = np.array([d.root_lin_vel_b[0, 0].item(), d.root_ang_vel_b[0, 2].item(),
                          *q[:6], np.clip((q[6] - .001) / .044, 0., 1.),
                          *[float(mode == m) for m in MODES]], dtype=np.float32)
        if mode not in MODES or state.shape != (13,) or not np.isfinite(state).all():
            raise ValueError('Invalid policy state or mode')
        obs = {'observation/state': state, 'observation/task_phase': np.int32(self.task_phase), 'prompt': PROMPT}
        for key, sensor in [('observation/image', 'front_rgb'), ('observation/wrist_image', 'wrist_rgb')]:
            rgb = env.scene[sensor].data.output['rgb'][0, :, :, :3].detach().cpu().numpy().copy()
            if rgb.shape != (480, 640, 3) or rgb.dtype != np.uint8:
                raise ValueError('Expected uint8 RGB camera[480,640,3]')
            obs[key] = rgb
        return obs

    def accept_prediction(self, generation, result, start_index=0):
        if not self.model_control or generation != self.generation:
            return False
        actions = np.asarray(result['actions'])
        if actions.shape != (50, 14) or not np.isfinite(actions).all():
            raise ValueError('Expected finite prediction[50,14]')
        if not isinstance(start_index,int) or not 0 <= start_index < len(actions):
            return False
        self.actions = list(actions[start_index:start_index+self.action_horizon].copy())
        return True

    def take_action(self, mode):
        if not self.model_control or not self.actions:
            return None
        action = self.actions.pop(0)
        decoded = decode_action(action)
        if mode in ('low_grasp', 'high_grasp') and decoded[1] not in (mode, 'walk'):
            self.invalidate()
            return None
        return decoded, bool(action[13] >= .5), self.generation

    def commit(self, token):
        """Call only after the simulator actually executes the accepted row."""
        _, advance, generation = token
        if self.model_control and generation == self.generation and advance and self.task_phase == 0:
            self.task_phase = 1
            self.invalidate()

    def close(self):
        self.set_model_control(False)
        self.remote.close()

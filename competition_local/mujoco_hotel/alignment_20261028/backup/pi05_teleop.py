"""MuJoCo observation/action adapter for the unchanged trained OpenPI policy."""
import concurrent.futures
import json
from pathlib import Path
import sys
import time
from types import SimpleNamespace as NS

import mujoco
import numpy as np
import torch

from original_teleop import OriginalTeleop, HERE, ARMS, LIMITS

ROOT = HERE.parent
sys.path.insert(0, str(ROOT / 'pi05_client_deps'))
sys.path.insert(0, str(ROOT / 'cloud_snapshot/openpi/packages/openpi-client/src'))
sys.path.insert(0, str(ROOT / 'cloud_snapshot/PPSUC2026/scripts/rsl_rl'))
from full_client import FullClient


class Pi05Teleop(OriginalTeleop):
    def __init__(self, host='127.0.0.1', port=8765, execute_steps=25):
        super().__init__()
        self.model_control = True
        self.client = FullClient(host, port, list(range(8)), execute_steps=execute_steps)
        self.executor = concurrent.futures.ThreadPoolExecutor(max_workers=1, thread_name_prefix='pi05')
        self.future = None
        self.context = None
        self.last_env = None
        self.inference_count = self.executed_steps = self.rejected_steps = 0
        self.last_infer_seconds = None
        self.last_error = ''
        self.run = HERE / 'runs/pi05_40000_demo'
        self.run.mkdir(parents=True, exist_ok=True)
        self.trace = (self.run / 'actions.jsonl').open('a', encoding='utf-8', buffering=1)
        self.sensor_scene = mujoco.MjvScene(self.model, maxgeom=12000)
        self.sensor_option = mujoco.MjvOption()
        self.sensor_option.geomgroup[3] = 0
        self.sensor_camera = mujoco.MjvCamera()
        self.sensor_camera.type = mujoco.mjtCamera.mjCAMERA_FIXED
        self.camera_ids = [next(i for i in range(self.model.ncam) if suffix in self.model.camera(i).name)
                           for suffix in ['HeadCamera_HeadCamera', 'GripperCamera_GripperCamera']]
        # Match the source runtime's initial standing warm-up before camera inference.
        for _ in range(50):
            super().control_frame(.02, command_override=np.zeros(3))
        self.message = 'PI0.5 ready; F4/Space manual, F5 model, F2 phase in manual'

    def attach_context(self, context):
        self.context = context
        mujoco.mjr_resizeOffscreen(640, 480, context)

    def policy_status(self):
        return {
            'pi05_inference': True, 'checkpoint_step': 40000,
            'control': 'model' if self.model_control else 'manual',
            'task_phase': self.client.task_phase, 'execute_steps': self.client.broker._action_horizon,
            'inference_count': self.inference_count, 'executed_policy_steps': self.executed_steps,
            'rejected_policy_steps': self.rejected_steps,
            'waiting_for_prediction': self.future is not None,
            'last_infer_seconds': self.last_infer_seconds, 'policy_error': self.last_error,
        }

    def set_mode(self, value):
        if self.mode in ('low_grasp', 'high_grasp') and value not in (self.mode, 'walk'):
            return
        # Repeated locomotion predictions must not cancel ongoing automatic homing.
        if value == self.mode:
            return
        super().set_mode(value)

    def set_model_control(self, enabled):
        self.model_control = bool(enabled)
        self.client.set_model_control(enabled)
        self.keys.clear()
        self.ik_target = None
        self.ik_velocity[:] = 0.
        self.command = np.zeros(3)
        self.message = 'PI0.5 MODEL CONTROL' if enabled else 'MANUAL CONTROL; F5 resumes PI0.5'
        print('[PI05 CONTROL]', self.message, flush=True)

    def key(self, key, pressed=True, repeat=False):
        if key in ('Q', 'ESCAPE'):
            return super().key(key, pressed, repeat)
        if key in ('F4', 'F5', 'F2', 'SPACE'):
            if pressed and not repeat:
                if key in ('F4', 'SPACE'):
                    self.set_model_control(False)
                elif key == 'F5':
                    self.set_model_control(True)
                elif not self.model_control:
                    self.client.toggle_task_phase()
            return
        if not self.model_control:
            return super().key(key, pressed, repeat)

    def _environment(self, capture):
        if capture:
            if self.context is None:
                raise RuntimeError('The visible MuJoCo renderer must be initialized first')
            images = []
            viewport = mujoco.MjrRect(0, 0, 640, 480)
            mujoco.mjr_setBuffer(mujoco.mjtFramebuffer.mjFB_OFFSCREEN, self.context)
            try:
                for camera_id in self.camera_ids:
                    self.sensor_camera.fixedcamid = camera_id
                    mujoco.mjv_updateScene(self.model, self.data, self.sensor_option, None,
                                          self.sensor_camera, mujoco.mjtCatBit.mjCAT_ALL, self.sensor_scene)
                    mujoco.mjr_render(viewport, self.sensor_scene, self.context)
                    rgb = np.empty((480, 640, 3), dtype=np.uint8)
                    mujoco.mjr_readPixels(rgb, None, viewport, self.context)
                    images.append(np.flipud(rgb).copy())
            finally:
                mujoco.mjr_setBuffer(mujoco.mjtFramebuffer.mjFB_WINDOW, self.context)
            self.images = images
            if self.inference_count == 0:
                from PIL import Image
                for name, image in zip(('front', 'wrist'), images):
                    Image.fromarray(image).save(self.run / f'first_observation_{name}.png')
        velocity = np.zeros(6)
        mujoco.mj_objectVelocity(self.model, self.data, mujoco.mjtObj.mjOBJ_BODY, self.base, velocity, 1)
        state = NS(root_lin_vel_b=torch.tensor([[velocity[3], velocity[4], velocity[5]]], dtype=torch.float32),
                   root_ang_vel_b=torch.tensor([[velocity[0], velocity[1], velocity[2]]], dtype=torch.float32),
                   joint_pos=torch.tensor(self.data.qpos[self.qadr[16:24]][None], dtype=torch.float32))
        env = NS(scene={'robot': NS(data=state)})
        for key, image in zip(('front_rgb', 'wrist_rgb'), self.images):
            env.scene[key] = NS(data=NS(output={'rgb': torch.from_numpy(image[None])}))
        return env

    def _predict(self, env, mode):
        started = time.monotonic()
        result = self.client.step(env, mode)
        return result, time.monotonic() - started

    def _take_prediction(self):
        if self.future is not None:
            if not self.future.done():
                self.message = 'PI0.5 predicting... physics paused; camera UI and F4 remain active'
                return False, None
            future, self.future = self.future, None
            action, self.last_infer_seconds = future.result()
            self.inference_count += 1
            self.message = f'PI0.5 prediction {self.inference_count}, {self.last_infer_seconds:.3f}s'
            return True, action
        if self.client.broker._last_results is None:
            env = self._environment(capture=True)
            self.future = self.executor.submit(self._predict, env, self.mode)
            self.message = 'PI0.5 predicting...'
            return False, None
        return True, self.client.step(self._environment(capture=False), self.mode)

    def control_frame(self, dt=.02):
        if not self.model_control:
            # Drain a stale request only after it has finished; never execute it.
            if self.future is not None and self.future.done():
                try:
                    self.future.result()
                except Exception:
                    pass
                self.future = None
            return super().control_frame(dt)
        try:
            ready, decoded = self._take_prediction()
        except Exception as exc:
            self.last_error = f'{type(exc).__name__}: {exc}'
            self.set_model_control(False)
            self.message = 'PI0.5 error; manual control: ' + self.last_error
            print('[PI05 ERROR]', self.last_error, flush=True)
            return False
        if not ready:
            return False
        command = np.zeros(3)
        if decoded is not None:
            command, next_mode, joints, gripper = decoded
            self.set_mode(next_mode)
            if self.arm_ready:
                self.home_motion = None
                self.ik_target = None
                self.ik_velocity[:] = 0.
                for name, value in zip(ARMS[:6], np.clip(joints, LIMITS[:, 0], LIMITS[:, 1])):
                    self.goals[name] = float(value)
                opening = .001 + float(gripper) * (.045 - .001)
                self.goals[ARMS[6]], self.goals[ARMS[7]] = opening, -opening
            self.executed_steps += 1
        else:
            self.rejected_steps += 1
        super().control_frame(dt, command_override=command)
        if self.executed_steps % 25 == 0:
            self.trace.write(json.dumps({'wall_time': time.time(), 'sim_time': float(self.data.time),
                'base': self.data.xpos[self.base].tolist(), 'command': list(command),
                'mode': self.mode, **self.policy_status()}) + '\n')
        return True

    def close(self):
        self.client.set_model_control(False)
        self.client.close()
        self.executor.shutdown(wait=False, cancel_futures=True)
        self.trace.close()

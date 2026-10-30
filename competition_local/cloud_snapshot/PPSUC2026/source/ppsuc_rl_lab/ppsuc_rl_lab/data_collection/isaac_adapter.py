"""Robot-specific camera setup and snapshots. Control remains in teleop."""
import json
import os
from pathlib import Path

import numpy as np
from .episode import EpisodeWriter, FPS, mode_vector
from .stages import STAGE_PROMPTS, STAGE_COUNT


def configure_cameras(env_cfg):
    from isaaclab.sensors import CameraCfg
    for name, path in (
        ("front_rgb", "base_link/HeadCamera/HeadCamera"),
        ("wrist_rgb", "gripper_base/GripperCamera/GripperCamera"),
    ):
        setattr(env_cfg.scene, name, CameraCfg(
            prim_path=f"{env_cfg.scene.robot.prim_path}/{path}",
            update_period=1/FPS, height=480, width=640,
            data_types=["rgb"], spawn=None))


class Collector:
    def __init__(self, env, joint_ids, directory, closed, opened, notify=None):
        if not np.isclose(env.step_dt, 1/FPS, atol=1e-9, rtol=0):
            raise ValueError(f"Expected env.step_dt=0.02, got {env.step_dt}; do not alter cerebellum cadence")
        if opened <= closed:
            raise ValueError("Invalid gripper limits")
        self.env, self.robot, self.ids = env, env.scene["robot"], joint_ids
        self.directory = directory
        self.closed, self.opened = closed, opened
        self.cameras = ["front_rgb", "wrist_rgb"]
        self.stage_id = 1
        self._stage_start = 0
        self._pending_stage = False
        self._submitted_stage = 1
        self.writer = None
        self.last_command = None
        self._observation = None
        self._pending_frame = None
        self.mode = None
        self.arm_ready = False
        self.completed = 0
        self._notify_callback = notify or (lambda message, error=False: None)
        self._status_path = os.environ.get("PPSUC_STAGE_STATUS")
        self.notify("等待录制：按 F1 开始")

    def notify(self, message, error=False):
        self._status_message, self._status_error = message, error
        self._publish_status()
        self._notify_callback(message, error=error)

    def _publish_status(self):
        # Best-effort UI status, independent of the dataset disk and frame writes.
        if self._status_path:
            try:
                path = Path(self._status_path)
                temporary = path.with_suffix(".tmp")
                temporary.write_text(json.dumps(dict(
                    message=self._status_message, error=self._status_error,
                    mode=self.mode, arm_ready=self.arm_ready, recording=self.writer is not None,
                    stage=self.stage_id, prompt=STAGE_PROMPTS[self.stage_id-1],
                ), ensure_ascii=False), encoding="utf-8")
                temporary.replace(path)
            except OSError:
                pass

    def ratio(self, value):
        return float(np.clip((value-self.closed)/(self.opened-self.closed), 0, 1))

    def start(self):
        if self.writer is None:
            try:
                self.writer = EpisodeWriter(self.directory)
            except Exception as exc:
                self.notify(f"Recording failed: {exc}", error=True)
                raise
            self._observation = self._pending_frame = None
            self.stage_id = self._submitted_stage = 1
            self._stage_start = 0
            self._pending_stage = False
            print(f"[RECORD] start {self.writer.path}", flush=True)
            self._show_stage()

    def _show_stage(self):
        message = f"[RECORD] 阶段 {self.stage_id}/{STAGE_COUNT}: {STAGE_PROMPTS[self.stage_id-1]} | F2 下一阶段 / F3 保存"
        print(message, flush=True)
        self.notify(message)

    def next_stage(self):
        if self.writer is None:
            return
        if self.stage_id == STAGE_COUNT:
            self.notify("已到最后阶段；完成放置后按 F3 保存")
        elif self.writer.count > self._stage_start:
            self._pending_stage = True

    def stop(self, publish=True):
        self._observation = self._pending_frame = None
        if self.writer is not None:
            writer = self.writer
            try:
                writer.close(publish)
            except Exception as exc:
                self.notify(f"Save failed: {exc}", error=True)
                raise
            if publish and writer.count:
                self.notify(f"Saved: {writer.path} | {writer.count} frames | {writer.count/FPS:.2f} s")
            elif not publish:
                self.notify(f"Recording interrupted: {writer.partial}", error=True)
            else:
                self.notify("Recording ended: no frames captured")
            self.writer = None
            self.completed += 1

    def command_submitted(self, command, arm_target, mode, *, arm_ready=False):
        # Call only after command setters have succeeded. Copy, never alias targets.
        if self.writer is not None and self._pending_stage:
            self.stage_id += 1
            self._stage_start = self.writer.count
            self._pending_stage = False
            self._show_stage()
        self._submitted_stage = self.stage_id
        arm = arm_target.detach().cpu().numpy()[0]
        self.last_command = np.array([command[0], command[2], *arm[:6], self.ratio(arm[6]), *mode_vector(mode)], np.float32)
        arm_ready = bool(arm_ready)
        if self.mode != mode or self.arm_ready != arm_ready:
            self.mode = mode
            self.arm_ready = arm_ready
            self._publish_status()
        if self.writer is not None and self._observation is not None:
            sim_time, state, images = self._observation
            self._pending_frame = (self.writer, sim_time, state,
                                   self.last_command.copy(), images, self._submitted_stage)
            self._observation = None

    def capture_observation(self, sim_time, current_mode):
        """Copy the observation before this control step changes the simulation."""
        self._observation = self._pending_frame = None
        if self.writer is None:
            return
        data = self.robot.data
        q = data.joint_pos[0, self.ids].detach().cpu().numpy()
        mode = mode_vector(current_mode)
        state = np.array([data.root_lin_vel_b[0, 0].item(), data.root_ang_vel_b[0, 2].item(),
                          *q[:6], self.ratio(q[6]), *mode], np.float32)
        images = {name.removesuffix("_rgb"): self.env.scene[name].data.output["rgb"][0, :, :, :3].detach().cpu().numpy().copy()
                  for name in self.cameras}
        self._observation = (sim_time, state, images)

    def sample(self):
        """Commit only the pre-step observation and command from this recording."""
        frame, self._pending_frame = self._pending_frame, None
        if frame is None or frame[0] is not self.writer:
            return
        _, sim_time, state, command, images, stage = frame
        try:
            self.writer.append(sim_time, state, command, images, stage)
        except Exception as exc:
            self.notify(f"Recording failed: {exc}", error=True)
            raise

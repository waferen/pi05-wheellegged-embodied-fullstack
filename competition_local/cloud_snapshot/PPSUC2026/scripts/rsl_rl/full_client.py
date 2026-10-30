"""Full-demo synchronous control, using ALOHA's ActionChunkBroker (50-step predictions)."""
import time
import numpy as np
from openpi_client import action_chunk_broker, websocket_client_policy

PROMPT = "Pick up both target objects and place them into the designated bin"
MODES = ("walk", "stair", "low_grasp", "high_grasp")

def decode_action(action, vx=1.0, wz=1.0):
    a = np.asarray(action)
    if a.shape != (14,) or not np.isfinite(a).all():
        raise ValueError("Expected finite action[14]")
    mode = MODES[int(np.argmax(a[9:13]))]
    command = (float(np.clip(a[0], -abs(vx), abs(vx))), 0.,
               float(np.clip(a[1], -abs(wz), abs(wz))))
    if mode in ("low_grasp", "high_grasp"):
        command = (0., 0., 0.)
    return command, mode, a[2:8].copy(), float(np.clip(a[8], 0., 1.))

class RemoteFullPolicy(websocket_client_policy.WebsocketClientPolicy):
    def __init__(self, host, port):
        super().__init__(host=host, port=port)
        meta = self.get_server_metadata()
        if (not isinstance(meta, dict) or meta.get("action_schema") != "b2w_piper_two_phase" or meta.get("action_dim") != 14
                or meta.get("default_prompt") != PROMPT
                or meta.get("phase_action_semantics") != "next_step_latched_v1"):
            self.close()
            raise ValueError(f"Not a compatible full policy: {meta}")
        print(f"[FULL POLICY] connected {meta}", flush=True)

    def infer(self, obs):
        start = time.monotonic()
        result = super().infer({**obs, "prompt": PROMPT})
        actions = np.asarray(result["actions"])
        if actions.shape != (50, 14) or not np.isfinite(actions).all():
            raise ValueError(f"Expected finite (50,14), got {actions.shape}")
        print(f"[FULL POLICY] infer_s={time.monotonic()-start:.3f} shape={actions.shape}", flush=True)
        return {**result, "actions": actions}

    def close(self):
        # The upstream client does not expose a public close method.
        self._ws.close()

class FullClient:
    def __init__(self, host, port, joint_ids, closed=.001, opened=.045, vx=1., wz=1., execute_steps=25):
        if opened <= closed:
            raise ValueError("Invalid gripper limits")
        if isinstance(execute_steps, bool) or not isinstance(execute_steps, int) or not 1 <= execute_steps <= 50:
            raise ValueError("execute_steps must be an integer from 1 to 50")
        self.remote = RemoteFullPolicy(host, port)
        self.broker = action_chunk_broker.ActionChunkBroker(self.remote, action_horizon=execute_steps)
        print(f"[FULL POLICY] prediction_steps=50 execute_steps={execute_steps}", flush=True)
        self.ids, self.closed, self.opened = joint_ids, closed, opened
        self.vx, self.wz = vx, wz
        self.task_phase = 0
        self.model_control = True
        self._control_generation = 0

    def set_model_control(self, enabled):
        enabled = bool(enabled)
        if enabled != self.model_control:
            self.model_control = enabled
            self._control_generation += 1
            self.broker.reset()  # Preserve task_phase across human takeover.

    def toggle_task_phase(self):
        """Allow an explicit phase correction only during human takeover."""
        if self.model_control:
            return False
        previous = self.task_phase
        self.task_phase = 1 - previous
        self._control_generation += 1
        self.broker.reset()
        print(f"[FULL POLICY] manual task phase {previous + 1} -> {self.task_phase + 1}", flush=True)
        return True

    def step(self, env, mode):
        if not self.model_control:
            return None
        if mode not in MODES:
            raise ValueError(f"Unknown mode {mode}")
        generation = self._control_generation
        d = env.scene["robot"].data
        q = d.joint_pos[0,self.ids].detach().cpu().numpy()
        state = np.array([d.root_lin_vel_b[0,0].item(), d.root_ang_vel_b[0,2].item(),
            *q[:6], np.clip((q[6]-self.closed)/(self.opened-self.closed),0.,1.),
            *[float(mode==m) for m in MODES]], dtype=np.float32)
        obs = {
            "observation/state": state,
            "observation/task_phase": np.int32(self.task_phase),
            "observation/image": env.scene["front_rgb"].data.output["rgb"][0,:,:,:3].detach().cpu().numpy().copy(),
            "observation/wrist_image": env.scene["wrist_rgb"].data.output["rgb"][0,:,:,:3].detach().cpu().numpy().copy(),
            "prompt": PROMPT,
        }
        action = self.broker.infer(obs)["actions"]
        if generation != self._control_generation or not self.model_control:
            self.broker.reset()
            return None
        decoded = decode_action(action, self.vx, self.wz)
        if mode in ("low_grasp", "high_grasp") and decoded[1] not in (mode, "walk"):
            # Reject the entire row, including arm, gripper and phase outputs.
            self.broker.reset()
            print(f"[FULL POLICY] rejected mode {mode} -> {decoded[1]}", flush=True)
            return None
        # Only the action consumed THIS step can advance the phase, not later
        # predictions in the chunk. Only manual correction or reset can regress it.
        if self.task_phase == 0 and action[13] >= 0.5:
            self.task_phase = 1
            self.broker.reset()  # Next prediction immediately sees the new phase.
            print("[FULL POLICY] task_phase 0 -> 1", flush=True)
        return decoded

    def reset(self):
        self.task_phase = 0
        self.broker.reset()

    def close(self):
        self.remote.close()

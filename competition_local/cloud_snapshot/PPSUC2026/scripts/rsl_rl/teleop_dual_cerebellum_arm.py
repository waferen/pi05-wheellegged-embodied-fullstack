"""Pure teleop for the B2W-Piper robot.

Three cerebellum policies are selected by mode and driven with a keyboard:
  - WALK (57-dim):          base forward/back + turn via WASD / arrow keys.
  - STAIR (57-dim):         same command space, tuned for rough terrain/stairs.
  - STABILIZE (75-dim):     base stops, body height controllable (Y/U), and the
                            Piper arm joints + gripper are driven directly.

Mode keys: 1=walk, 2=stair, 4=low grasp, 5=high grasp with simulation base lock.
Keys 3/J/Y/U/Z/X are disabled; grasp height is controlled internally.
"""

import argparse
import json
import math
import hashlib
import os
import sys
import time
from pathlib import Path

# Load the environment NumPy before Kit adds its incomplete GUI pip archive.
import numpy as np

# Isaac Sim enumerates each NVIDIA GPU twice on hosts where the same ICD manifest
# is installed in both standard Vulkan search directories.  Respect an explicit
# user override, otherwise select one manifest before AppLauncher initializes GPU.
if "VK_ICD_FILENAMES" not in os.environ:
    for _vulkan_icd in (
        "/etc/vulkan/icd.d/nvidia_icd.json",
        "/usr/share/vulkan/icd.d/nvidia_icd.json",
    ):
        if os.path.isfile(_vulkan_icd):
            os.environ["VK_ICD_FILENAMES"] = _vulkan_icd
            break

from isaaclab.app import AppLauncher

import cli_args  # isort: skip

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SCENE_USD = str(PROJECT_ROOT / "ppsuc_robot_model/scene/Competition_hotel/Collected_task03/hotel/task03.usd")
TASK03_USD_ROBOT_PRIM = "/World/task_scene/b2w_piper"
# fallback pose used only if task03.usd cannot be read
TASK03_USD_INIT_POS = (21.87995765019431, 7.159765281698144, 0.27594364462937915)
TASK03_USD_INIT_ROT = (0.0, 0.0, 0.0, 1.0)
DEFAULT_WALK_CHECKPOINT = str(PROJECT_ROOT / "ppsuc_robot_model/baseline/walk_impulse_recovery_cerebellum.pt")
DEFAULT_STABILIZER_CHECKPOINT = str(PROJECT_ROOT / "ppsuc_robot_model/baseline/height_stabilization_cerebellum.pt")
DEFAULT_STAIR_CHECKPOINT = str(PROJECT_ROOT / "ppsuc_robot_model/baseline/rough_terrain_stair_cerebellum.pt")

parser = argparse.ArgumentParser(description="Tri-cerebellum teleop for the B2W-Piper robot.")
parser.add_argument("--task", type=str, default="PPSUC-Isaac-Velocity-TaskScene2-NativeRobot-Cerebellum-Unitree-B2W-Piper-v0")
parser.add_argument("--agent", type=str, default="rsl_rl_cfg_entry_point")
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument("--scene_usd", type=str, default=DEFAULT_SCENE_USD)
parser.add_argument("--can03_position", type=int, choices=(1, 2, 3, 4), required=True,
                    help="Required trash_can03 placement index (1-4).")
parser.add_argument("--walk_checkpoint", type=str, default=DEFAULT_WALK_CHECKPOINT)
parser.add_argument("--stair_checkpoint", type=str, default=DEFAULT_STAIR_CHECKPOINT)
parser.add_argument("--stabilizer_checkpoint", type=str, default=DEFAULT_STABILIZER_CHECKPOINT)
parser.add_argument("--stand_height", type=float, default=0.59)
parser.add_argument("--min_height", type=float, default=0.236)
parser.add_argument("--arm_step", type=float, default=0.035)
parser.add_argument("--arm_home_duration", type=float, default=1.0)
parser.add_argument("--gripper_step", type=float, default=0.004)
parser.add_argument("--gripper_open", type=float, default=0.045)
parser.add_argument("--gripper_closed", type=float, default=0.001)
parser.add_argument("--vx", type=float, default=0.25)
parser.add_argument("--wz", type=float, default=0.60)
parser.add_argument("--real-time", action="store_true", default=True)
parser.add_argument("--no-real-time", dest="real_time", action="store_false")
parser.add_argument("--max_steps", type=int, default=0)
parser.add_argument("--record_dir", type=str, default=None)
parser.add_argument("--record_start", action="store_true", help="Start after camera warmup")
parser.add_argument("--record_warmup", type=int, default=50)
parser.add_argument("--record_frames", type=int, default=0, help="Stop after N recorded frames; 0 is unlimited")
parser.add_argument("--command_schedule", type=str, default=None, help="JSON scripted control for headless integration checks")
parser.add_argument("--policy_full", action="store_true")
parser.add_argument("--policy_host", default=None)
parser.add_argument("--policy_port", type=int, default=8000)
parser.add_argument("--execute_steps", type=int, default=25, metavar="N",
                    help="Execute N actions before a new full-policy prediction (1-50; default: 25).")
cli_args.add_rsl_rl_args(parser)
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
if not 1 <= args_cli.execute_steps <= 50:
    parser.error("--execute_steps must be an integer from 1 to 50")
if args_cli.policy_full and not args_cli.policy_host:
    parser.error("--policy_full requires --policy_host")
if args_cli.policy_host:
    args_cli.policy_full = True
    if args_cli.headless or args_cli.num_envs != 1:
        parser.error("Policy requires GUI and one environment")
    args_cli.enable_cameras = True
if args_cli.record_dir:
    if args_cli.record_warmup < 0 or args_cli.record_frames < 0:
        parser.error("Recording warmup and frame count must be nonnegative")
    args_cli.enable_cameras = True
if args_cli.command_schedule and not args_cli.headless:
    parser.error("--command_schedule is restricted to headless runs")
sys.argv = [sys.argv[0]] + hydra_args

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym  # noqa: E402
import carb  # noqa: E402
import torch  # noqa: E402
from pxr import Sdf, Usd, UsdGeom  # noqa: E402

from isaaclab.devices import Se2Keyboard, Se2KeyboardCfg  # noqa: E402
from isaaclab.envs import DirectMARLEnv, DirectMARLEnvCfg, DirectRLEnvCfg, ManagerBasedRLEnvCfg, multi_agent_to_single_agent  # noqa: E402
from isaaclab.utils.assets import retrieve_file_path
from isaaclab.utils.math import quat_mul, quat_conjugate, euler_xyz_from_quat  # noqa: E402
from isaaclab_rl.rsl_rl import RslRlBaseRunnerCfg, RslRlVecEnvWrapper  # noqa: E402
from isaaclab_tasks.utils.hydra import hydra_task_config  # noqa: E402

import ppsuc_rl_lab.train  # noqa: F401, E402
from ppsuc_rl_lab.teleop.ik import ArmIK, IK_KEYS, ALIASES
from ppsuc_rl_lab.teleop.base_lock import LowPoseLock

ARM_JOINT_NAMES = [f"arm_joint{i}" for i in range(1, 9)]
ARM_HOME = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.045, -0.045]
ARM_HARD_LIMITS = [
    (-2.618, 2.618),
    (0.0, 3.14),
    (-2.967, 0.0),
    (-1.745, 1.745),
    (-1.22, 1.22),
    (-2.0944, 2.0944),
]


CAN03_POSITIONS = {
    1: (7.95, 16.8, 0.05),
    2: (6.75, 18.45, 0.05),
    3: (6.45, 15.6, 0.05),
    4: (8.2, 11.55, 0.05),
}


def _set_can03_position(stage, index):
    """Set only the selected bin translation before physics initialization."""
    from pxr import Gf
    path = "/World/task_scene/Obstacle/trash_can03"
    prim = stage.GetPrimAtPath(path)
    if not prim.IsValid():
        raise RuntimeError(f"Required trash_can03 prim not found: {path}")
    ops = {op.GetOpName(): op for op in UsdGeom.Xformable(prim).GetOrderedXformOps()}
    translate = ops.get("xformOp:translate")
    if translate is None:
        raise RuntimeError(f"Missing xformOp:translate on {path}")
    point = CAN03_POSITIONS[index]
    with Usd.EditContext(stage, stage.GetSessionLayer()):
        translate.Set(Gf.Vec3d(*point))
    print(f"[SCENE] trash_can03 position={index} local_xyz={point}", flush=True)


def _ee_log_pose(position, orientation):
    """Arm-base XYZ in meters, followed by roll/pitch/yaw in degrees."""
    rpy = torch.stack(euler_xyz_from_quat(orientation.reshape(1, 4)), dim=-1).reshape(-1)
    rpy = torch.rad2deg((rpy + math.pi) % (2 * math.pi) - math.pi)
    xyz = [round(float(v), 3) for v in position.reshape(-1).tolist()]
    return xyz + [round(float(v), 1) for v in rpy.tolist()]


def _open_camera_views(env):
    """Bind two viewports to existing scene cameras; never create camera prims."""
    import asyncio
    import omni.ui as ui
    import omni.kit.app
    import omni.usd
    from isaacsim.core.utils.viewports import create_viewport_for_camera
    from omni.kit.viewport.utility import get_active_viewport_window

    stage = omni.usd.get_context().get_stage()
    paths = []
    for name in ("HeadCamera", "GripperCamera"):
        matches = [str(prim.GetPath()) for prim in stage.Traverse()
                   if prim.IsA(UsdGeom.Camera) and prim.GetName() == name]
        if len(matches) != 1:
            print(f"[VIEWPORT] Cannot uniquely locate existing {name}: {matches}; keeping current view.", flush=True)
            return None
        paths.append(matches[0])
    collection_count = sum(prim.IsA(UsdGeom.Camera) and prim.GetName().startswith("collection_")
                           for prim in stage.Traverse())
    print(f"[VIEWPORT] existing cameras selected; collection cameras={collection_count}", flush=True)
    head = get_active_viewport_window()
    if head is None:
        head = create_viewport_for_camera("Head Camera", paths[0], 640, 480)
    head.viewport_api.camera_path = paths[0]
    hand = create_viewport_for_camera("Hand Camera", paths[1], 640, 480)
    for window in (head, hand):
        window.viewport_api.set_texture_resolution((640, 480))

    async def arrange():
        await omni.kit.app.get_app().next_update_async()
        await omni.kit.app.get_app().next_update_async()
        left = ui.Workspace.get_window(head.title)
        right = ui.Workspace.get_window(hand.title)
        if left is None or right is None:
            raise RuntimeError("Camera viewport windows were not created")
        hidden = []
        for title in ("Stage", "Layer", "Layers", "Render Settings", "Property",
                      "Semantics Schema Editor", "IsaacLab", "Content", "Content Browser", "Console"):
            panel = ui.Workspace.get_window(title)
            if panel is not None:
                panel.visible = False
                hidden.append(title)
        right.dock_in(left, ui.DockPosition.RIGHT, 0.5)
        for _ in range(10):
            await omni.kit.app.get_app().next_update_async()
        for window in (head, hand):
            api = window.viewport_api
            widget = window.viewport_widget
            print(f"[VIEWPORT] {window.title} before normalization: "
                  f"render={api.resolution}, widget={widget.full_resolution}", flush=True)
            # The widget owns the saved resolution and can overwrite API-only
            # changes when docking/resizing. Configure both layers consistently.
            api.fill_frame = False
            widget.fill_frame = False
            widget.expand_viewport = False
            widget.set_resolution((640, 480))
            api.resolution_scale = 1.0
        for _ in range(10):
            await omni.kit.app.get_app().next_update_async()
        for window in (head, hand):
            print(f"[VIEWPORT] {window.title}: render={window.viewport_api.resolution}, "
                  f"window={window.width}x{window.height}", flush=True)
        print(f"[VIEWPORT] hidden panels={hidden}", flush=True)
        print(f"[VIEWPORT] left=head {paths[0]}; right=hand {paths[1]}; 640x480", flush=True)

    return head, hand, asyncio.ensure_future(arrange())


def _prepare_scene_usd(usd_path):
    """Resolve and force-reload the scene root layer before Isaac Lab references it."""
    path = Path(usd_path).expanduser().resolve(strict=True)
    stat = path.stat()
    digest = hashlib.sha256(path.read_bytes()).hexdigest()

    layer = Sdf.Layer.FindOrOpen(str(path))
    if layer is None:
        raise RuntimeError(f"Failed to open scene USD layer: {path}")
    if not layer.Reload(force=True):
        raise RuntimeError(f"Failed to reload scene USD layer: {path}")

    print(
        f"[INFO]: scene root layer -> {path} "
        f"size={stat.st_size} mtime_ns={stat.st_mtime_ns} sha256={digest}",
        flush=True,
    )
    return str(path)


def _read_task03_robot_pose_from_usd(usd_path):
    """Read the robot base pose authored in task03.usd."""
    try:
        stage = Usd.Stage.Open(str(usd_path))
        if stage is None:
            raise RuntimeError(f"failed to open {usd_path}")
        for prim_path in (
            "/World/task_scene/b2w_piper/base_link",
            "/World/task_scene/b2w_piper",
            "/World/b2w_piper/base_link",
            "/World/b2w_piper",
        ):
            prim = stage.GetPrimAtPath(prim_path)
            if not prim.IsValid():
                continue
            matrix = UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
            translation = matrix.ExtractTranslation()
            quat = matrix.ExtractRotationQuat()
            pos = (float(translation[0]), float(translation[1]), float(translation[2]))
            imag = quat.GetImaginary()
            rot = (float(quat.GetReal()), float(imag[0]), float(imag[1]), float(imag[2]))
            print(f"[INFO]: read task03 robot pose from {prim_path}", flush=True)
            return pos, rot
        raise RuntimeError("task03 robot prim not found")
    except Exception as exc:
        print(f"[WARN]: failed to read robot pose from {usd_path}: {exc}; using fallback constants", flush=True)
        return TASK03_USD_INIT_POS, TASK03_USD_INIT_ROT


def _load_actor_mlp(checkpoint, device):
    loaded = torch.load(checkpoint, map_location=device)
    state_dict = loaded.get("model_state_dict", loaded)
    actor_state = {
        key[len("actor.") :]: value
        for key, value in state_dict.items()
        if key.startswith("actor.")
    }
    if not actor_state:
        raise RuntimeError(f"No actor.* weights found in {checkpoint}")
    layers = []
    linear_ids = sorted(
        int(key.split(".", 1)[0])
        for key in actor_state
        if key.endswith(".weight")
    )
    for index, layer_id in enumerate(linear_ids):
        weight = actor_state[f"{layer_id}.weight"]
        layers.append(torch.nn.Linear(weight.shape[1], weight.shape[0]))
        if index < len(linear_ids) - 1:
            layers.append(torch.nn.ELU())
    actor = torch.nn.Sequential(*layers).to(device)
    actor.load_state_dict(actor_state)
    actor.eval()
    return actor


def _set_base_command(raw_env, command):
    term = raw_env.command_manager._terms["base_velocity"]
    cmd = torch.as_tensor(command, dtype=torch.float32, device=raw_env.device).view(1, 3)
    term.vel_command_b[:] = cmd.repeat(raw_env.num_envs, 1)
    if hasattr(term, "heading_target"):
        term.heading_target[:] = 0.0
    if hasattr(term, "is_heading_env"):
        term.is_heading_env[:] = False
    if hasattr(term, "is_standing_env"):
        term.is_standing_env[:] = bool(torch.all(cmd == 0.0).item())


def _set_height(raw_env, height):
    term = raw_env.command_manager._terms["body_height"]
    term.height_command[:, 0] = float(height)


def _policy_tensor(obs):
    if isinstance(obs, dict):
        return obs["policy"]
    if isinstance(obs, (tuple, list)):
        return _policy_tensor(obs[0])
    return obs


def _get_policy_obs(raw_env, wrapped_env):
    if hasattr(raw_env, "observation_manager"):
        obs = raw_env.observation_manager.compute_group("policy")
        return _policy_tensor(obs)
    return _policy_tensor(wrapped_env.get_observations())


def _slice_walk_obs(obs75):
    # 75-dim stabilizer observation:
    # base_ang_vel(3), gravity(3), vel_cmd(3), joint_pos(24), joint_vel(24), actions(16), height(1), z(1)
    # 57-dim legacy walking observation:
    # base_ang_vel(3), gravity(3), vel_cmd(3), leg/wheel joint_pos(16), leg/wheel joint_vel(16), actions(16)
    if obs75.shape[-1] == 57:
        return obs75
    if obs75.shape[-1] != 75:
        raise RuntimeError(f"Expected 75-dim task02 observation, got {obs75.shape}")
    return torch.cat((obs75[:, :9], obs75[:, 9:25], obs75[:, 33:49], obs75[:, 57:73]), dim=-1)


def _scale_walk_command(raw_command):
    if torch.is_tensor(raw_command):
        raw_command = raw_command.detach().cpu().numpy()
    raw = np.asarray(raw_command, dtype=np.float32)
    if raw.shape[0] < 3:
        raise RuntimeError(f"Expected 3-dim keyboard command, got {raw}")

    vx = float(np.clip(raw[0], -1.0, 1.0) * args_cli.vx)
    wz = float(np.clip(raw[2], -1.0, 1.0) * args_cli.wz)
    return (vx, 0.0, wz)


class StateBasedSe2Keyboard(Se2Keyboard):
    """SE(2) keyboard controller that derives commands from pressed-key state."""

    _MOVEMENT_KEYS = {
        "W": (1, 0),
        "S": (-1, 0),
        "A": (0, 1),
        "D": (0, -1),
    }

    def __init__(self, cfg):
        super().__init__(cfg)
        self._pressed_keys = set()
        self._record_keys = set()
        self.ik_keys = set()
        self.ik_pending = []
        self.ik_cancel = False
        self.joint_keys = set()
        self.arm_is_ready = lambda: False
        self.manual_control_enabled = lambda: True
        import omni.appwindow
        self._focus_sub = omni.appwindow.get_default_app_window().get_window_focus_event_stream().create_subscription_to_pop(
            lambda event: self.reset(), name="Teleop clear keys on focus change")

    def reset(self):
        self._pressed_keys.clear()
        self._record_keys.clear()
        self.ik_keys.clear()
        self.ik_pending.clear()
        self.ik_cancel = True
        self.joint_keys.clear()
        super().reset()

    def _refresh_command(self):
        x = 0.0
        wz = 0.0
        for key in self._pressed_keys:
            dx, dwz = self._MOVEMENT_KEYS[key]
            x += dx
            wz += dwz
        self._base_command[:] = np.asarray([np.clip(x, -1.0, 1.0), 0.0, np.clip(wz, -1.0, 1.0)])

    _KEY_NORMALIZE = {
        "1": "1", "ONE": "1", "KEY_1": "1", "NUM_1": "1", "NUMPAD_1": "1",
        "2": "2", "TWO": "2", "KEY_2": "2", "NUM_2": "2", "NUMPAD_2": "2",
        "3": "3", "THREE": "3", "KEY_3": "3", "NUM_3": "3", "NUMPAD_3": "3",
        "4": "4", "FOUR": "4", "NUM_4": "4", "NUMPAD_4": "4",
        "5": "5", "FIVE": "5", "NUM_5": "5", "NUMPAD_5": "5",
        "J": "J", "KEY_J": "J",
        "H": "H", "KEY_H": "H",
        "SPACE": "SPACE", "KEY_SPACE": "SPACE",
        "ESCAPE": "ESCAPE", "KEY_ESCAPE": "ESCAPE",
        "Q": "Q", "KEY_Q": "Q",
        "Y": "Y", "KEY_Y": "Y",
        "U": "U", "KEY_U": "U",
        "Z": "Z", "KEY_Z": "Z",
        "X": "X", "KEY_X": "X",
        "R": "R", "KEY_R": "R",
        "T": "T", "KEY_T": "T",
        "E": "E", "KEY_E": "E",
        "G": "G", "KEY_G": "G",
        "V": "V", "KEY_V": "V",
        "B": "B", "KEY_B": "B",
        "N": "N", "KEY_N": "N",
        "M": "M", "KEY_M": "M",
        "I": "I", "KEY_I": "I",
        "O": "O", "KEY_O": "O",
        "K": "K", "KEY_K": "K",
        "L": "L", "KEY_L": "L",
        "C": "C", "KEY_C": "C",
        "P": "P", "KEY_P": "P",
    }

    @staticmethod
    def _event_key(event_input):
        # Isaac/Carb may report keys as strings or KeyboardInput objects, and
        # some builds include prefixes. Keep movement keys and callbacks robust.
        key = event_input if isinstance(event_input, str) else event_input.name
        key = str(key).rsplit(".", 1)[-1].upper()
        if key.startswith("KEY_") and len(key) > 4:
            key = key[4:]
        return key

    def _on_keyboard_event(self, event, *args, **kwargs):
        raw_key = self._event_key(event.input)
        callback_key = self._KEY_NORMALIZE.get(raw_key, raw_key)
        ik_key = ALIASES.get(raw_key, raw_key)
        if (not self.manual_control_enabled()
                and callback_key not in ("F4", "F5", "Q", "ESCAPE")):
            return True
        if callback_key in ("F1", "F2", "F3", "F4", "F5"):
            if event.type == carb.input.KeyboardEventType.KEY_PRESS:
                if callback_key not in self._record_keys:
                    self._record_keys.add(callback_key)
                    callback = self._additional_callbacks.get(callback_key)
                    if callback:
                        callback()
            elif event.type == carb.input.KeyboardEventType.KEY_RELEASE:
                self._record_keys.discard(callback_key)
            return True
        # Grasp mode keys are separate from the rotation controls.
        if callback_key in ("4", "5"):
            if event.type == carb.input.KeyboardEventType.KEY_PRESS:
                self._additional_callbacks[callback_key]()
            return True
        if ik_key in IK_KEYS:
            if event.type == carb.input.KeyboardEventType.KEY_PRESS and self.arm_is_ready():
                if ik_key not in self.ik_keys:
                    if ik_key not in self.ik_pending:
                        self.ik_pending.append(ik_key)
                self.ik_keys.add(ik_key)
            elif event.type == carb.input.KeyboardEventType.KEY_RELEASE:
                self.ik_keys.discard(ik_key)
            return False
        if callback_key in ("R","T","E","G","V","B","N","M","I","O","K","L"):
            if event.type == carb.input.KeyboardEventType.KEY_PRESS:
                self.joint_keys.add(callback_key)
            elif event.type == carb.input.KeyboardEventType.KEY_RELEASE:
                self.joint_keys.discard(callback_key)
        if event.type == carb.input.KeyboardEventType.KEY_PRESS:
            if callback_key in ("1", "2", "3"):
                if callback_key in self._additional_callbacks:
                    self._additional_callbacks[callback_key]()
                return True
            if raw_key in self._MOVEMENT_KEYS:
                self._pressed_keys.add(raw_key)
                self._refresh_command()
            elif callback_key == "SPACE":
                self.reset()
            if callback_key in self._additional_callbacks:
                self._additional_callbacks[callback_key]()
        elif event.type == carb.input.KeyboardEventType.KEY_RELEASE:
            if raw_key in self._MOVEMENT_KEYS:
                self._pressed_keys.discard(raw_key)
                self._refresh_command()
        return True


class HeadlessSe2Input:
    """No-op SE(2) input used when no app window exists."""

    def reset(self):
        return None

    def advance(self):
        return (0.0, 0.0, 0.0)

    def add_callback(self, _key, _callback):
        return None


@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: RslRlBaseRunnerCfg):
    if args_cli.num_envs != 1:
        raise ValueError("Teleop supports --num_envs 1 only.")

    env_cfg.scene.num_envs = 1
    # Reduce shoulder/elbow static tracking error in teleop.
    # Keep the shared training configuration and drive effort limits unchanged.
    arm_drive = env_cfg.scene.robot.actuators["arms"]
    arm_drive.stiffness = {"arm_joint[23]": 800.0, "arm_joint[1456]": 80.0}
    arm_drive.damping = {"arm_joint[23]": 13.0, "arm_joint[1456]": 4.0}
    env_cfg.observations.policy.enable_corruption = False
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device
    task03_authored_pose = None
    if args_cli.scene_usd and hasattr(env_cfg.scene, "task_scene"):
        scene_usd = _prepare_scene_usd(args_cli.scene_usd)
        env_cfg.scene.task_scene.spawn.usd_path = scene_usd
        print(f"[INFO]: scene usd -> {scene_usd}", flush=True)
        if os.path.basename(scene_usd) == "task03.usd":
            env_cfg.scene.robot.prim_path = TASK03_USD_ROBOT_PRIM
            env_cfg.scene.robot.articulation_root_prim_path = "/base_link"
            task03_init_pos, task03_init_rot = _read_task03_robot_pose_from_usd(scene_usd)
            task03_authored_pose = (task03_init_pos, task03_init_rot)
            env_cfg.scene.robot.init_state.pos = task03_init_pos
            env_cfg.scene.robot.init_state.rot = task03_init_rot
            print(f"[INFO]: task03 authored robot pose pos={task03_init_pos} rot={task03_init_rot}", flush=True)
            # The teleop policy observation does not consume ray-caster data. In GUI mode,
            # these sensors can initialize while the large scene is still rendering, before
            # PhysX has created its articulation view.
            env_cfg.scene.height_scanner = None
            env_cfg.scene.height_scanner_base = None
            print(f"[INFO]: robot prim -> {TASK03_USD_ROBOT_PRIM}", flush=True)
    if hasattr(env_cfg.commands, "body_height"):
        env_cfg.commands.body_height.resampling_time_range = (1.0e9, 1.0e9)
        env_cfg.commands.body_height.discrete_values = [args_cli.stand_height]
    if hasattr(env_cfg, "episode_length_s"):
        env_cfg.episode_length_s = 1.0e9
    env_cfg.log_dir = os.path.dirname(args_cli.stabilizer_checkpoint)

    if args_cli.record_dir or args_cli.policy_host:
        from ppsuc_rl_lab.data_collection.isaac_adapter import configure_cameras
        configure_cameras(env_cfg)

    # Apply the placement as soon as the scene is spawned, before PhysX starts.
    scene_spawn = env_cfg.scene.task_scene.spawn.func

    def _spawn_scene_with_can03(*args, **kwargs):
        prim = scene_spawn(*args, **kwargs)
        _set_can03_position(prim.GetStage(), args_cli.can03_position)
        return prim

    env_cfg.scene.task_scene.spawn.func = _spawn_scene_with_can03

    env = gym.make(args_cli.task, cfg=env_cfg)
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)
    raw_env = env.unwrapped
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    camera_views = _open_camera_views(raw_env) if not args_cli.headless else None

    device = raw_env.device
    walk_checkpoint = retrieve_file_path(args_cli.walk_checkpoint)
    stair_checkpoint = retrieve_file_path(args_cli.stair_checkpoint)
    stabilizer_checkpoint = retrieve_file_path(args_cli.stabilizer_checkpoint)
    print(f"[INFO]: Loading walk checkpoint: {walk_checkpoint}", flush=True)
    walk_actor = _load_actor_mlp(walk_checkpoint, device)
    print(f"[INFO]: Loading stair checkpoint: {stair_checkpoint}", flush=True)
    stair_actor = _load_actor_mlp(stair_checkpoint, device)
    print(f"[INFO]: Loading stabilizer checkpoint: {stabilizer_checkpoint}", flush=True)
    stabilizer_actor = _load_actor_mlp(stabilizer_checkpoint, device)
    print(f"[INFO]: walk speed limits vx={args_cli.vx:.3f}, wz={args_cli.wz:.3f}", flush=True)

    robot = raw_env.scene["robot"]
    arm_joint_ids, arm_joint_names = robot.find_joints(ARM_JOINT_NAMES, preserve_order=True)
    if len(arm_joint_ids) != 8:
        raise RuntimeError(f"Expected 8 arm joints, got {len(arm_joint_ids)}: {arm_joint_names}")

    collector = None
    if args_cli.record_dir:
        from ppsuc_rl_lab.data_collection.isaac_adapter import Collector
        from ppsuc_rl_lab.data_collection.notifications import create_notifier
        notifier = None if args_cli.headless else create_notifier()
        collector = Collector(raw_env, arm_joint_ids, args_cli.record_dir,
                              args_cli.gripper_closed, args_cli.gripper_open, notify=notifier)

    schedule = []
    if args_cli.command_schedule:
        schedule = json.loads(Path(args_cli.command_schedule).read_text())
        for item in schedule:
            if item["step"] < 0 or item["mode"] not in ("walk", "stair", "low_grasp", "high_grasp"):
                raise ValueError("Invalid command schedule")
            if abs(item["vx"]) > args_cli.vx or abs(item["wz"]) > args_cli.wz:
                raise ValueError("Scheduled command exceeds speed limits")
        schedule.sort(key=lambda item: item["step"])
    scheduled_command = (0.0, 0.0, 0.0)
    mode = {"value": "walk"}
    control = {"model": bool(args_cli.policy_full)}
    policy_client = None
    low_lock = LowPoseLock(robot, arm_joint_ids)
    original_sim_step = raw_env.sim.step

    def _sim_step_with_lock(render=True):
        low_lock.enforce()
        return original_sim_step(render=render)

    raw_env.sim.step = _sim_step_with_lock
    target_height = {"value": float(args_cli.stand_height)}
    base_command = {"value": (0.0, 0.0, 0.0)}
    should_quit = {"value": False}

    arm_target = torch.tensor([ARM_HOME], dtype=torch.float32, device=device)
    gripper_value = {"value": float(args_cli.gripper_open)}
    ik_pause = {"until": 0}
    arm_home_motion = {"active": False, "start": None, "target": None, "start_step": 0, "duration": 0.0}
    full_status = {"path": os.environ.get("PPSUC_FULL_STATUS") if args_cli.policy_full else None,
                   "last": None, "status": "starting", "message": ""}

    def _print_mode():
        print(f"[MODE]: {mode['value']} height_cmd={target_height['value']:.3f}", flush=True)

    def _clamp_arm_target():
        limits = torch.tensor(ARM_HARD_LIMITS, dtype=torch.float32, device=device)
        arm_target[0, :6] = torch.max(torch.min(arm_target[0, :6], limits[:, 1]), limits[:, 0])
        gripper_value["value"] = max(args_cli.gripper_closed, min(args_cli.gripper_open, gripper_value["value"]))
        arm_target[0, 6] = gripper_value["value"]
        arm_target[0, 7] = -gripper_value["value"]

    def _cancel_arm_home_motion():
        arm_home_motion["active"] = False

    def _set_arm_home(smooth=True, *, mode_exit=False):
        if smooth and not mode_exit and not _arm_enabled():
            return
        target = torch.tensor([ARM_HOME[:6]], dtype=torch.float32, device=device)
        if smooth:
            arm_home_motion["active"] = True
            arm_home_motion["start"] = robot.data.joint_pos[0, arm_joint_ids[:6]].clone().unsqueeze(0)
            arm_home_motion["target"] = target
            arm_home_motion["start_step"] = step
            arm_home_motion["duration"] = max(float(args_cli.arm_home_duration), dt)
            print(f"[ARM]: homing over {arm_home_motion['duration']:.2f}s", flush=True)
            return
        _cancel_arm_home_motion()
        arm_target[:, :6] = target
        _clamp_arm_target()
        print("[ARM]: home", flush=True)

    def _write_arm_home_to_sim():
        target = torch.tensor([ARM_HOME], dtype=torch.float32, device=device)
        joint_pos = robot.data.joint_pos.clone()
        joint_vel = robot.data.joint_vel.clone()
        joint_pos[:, arm_joint_ids] = target
        joint_vel[:, arm_joint_ids] = 0.0
        robot.write_joint_state_to_sim(joint_pos, joint_vel)
        robot.set_joint_position_target(target[:, :6], joint_ids=arm_joint_ids[:6])
        robot.set_joint_position_target(target[:, 6:8], joint_ids=arm_joint_ids[6:8])
        arm_target[:] = target
        gripper_value["value"] = float(args_cli.gripper_open)
        _clamp_arm_target()
        print("[ARM]: initial joint state written to ARM_HOME", flush=True)

    def _update_arm_home_motion():
        if not arm_home_motion["active"]:
            return
        elapsed = max(0.0, (step - arm_home_motion["start_step"]) * dt)
        alpha = min(1.0, elapsed / arm_home_motion["duration"])
        alpha = alpha * alpha * (3.0 - 2.0 * alpha)
        arm_target[:, :6] = arm_home_motion["start"] * (1.0 - alpha) + arm_home_motion["target"] * alpha
        if alpha >= 1.0:
            _cancel_arm_home_motion()
            print("[ARM]: home", flush=True)

    def _arm_enabled():
        return (mode["value"] in ("low_grasp", "high_grasp") and low_lock.locked)

    def _arm_delta(index, delta):
        def _fn():
            if not _arm_enabled():
                return
            _cancel_arm_home_motion()
            ik_pause["until"] = step + 10
            arm_target[0, index] += float(delta)
            _clamp_arm_target()
            print(f"[ARM]: {ARM_JOINT_NAMES[index]}={float(arm_target[0, index].item()):.3f}", flush=True)
        return _fn

    def _gripper_delta(delta):
        def _fn():
            if not _arm_enabled():
                return
            _cancel_arm_home_motion()
            gripper_value["value"] += float(delta)
            _clamp_arm_target()
            print(f"[GRIPPER]: {gripper_value['value']:.3f}", flush=True)
        return _fn

    def _set_gripper(value, label):
        def _fn():
            if not _arm_enabled():
                return
            _cancel_arm_home_motion()
            gripper_value["value"] = float(value)
            _clamp_arm_target()
            print(f"[GRIPPER]: {label} {gripper_value['value']:.3f}", flush=True)
        return _fn

    def _set_mode(mode_name, *, home_on_exit=True):
        def _fn():
            if (mode["value"] in ("low_grasp", "high_grasp")
                    and mode_name not in (mode["value"], "walk")):
                return
            if (home_on_exit and mode_name == "walk"
                    and mode["value"] in ("low_grasp", "high_grasp")):
                # Start from measured joints, then enter walk immediately.
                _set_arm_home(mode_exit=True)
            low_lock.release()
            teleop.reset()
            target_height["value"] = args_cli.stand_height
            mode["value"] = mode_name
            base_command["value"] = (0.0, 0.0, 0.0)
            _print_mode()
            print(f"[MODE]: set -> {mode_name}", flush=True)
        return _fn

    def _grasp(mode_name, height):
        if mode["value"] in ("low_grasp", "high_grasp") and mode["value"] != mode_name:
            return
        if mode["value"] == mode_name and low_lock.phase in ("lowering", "locked"):
            return
        low_lock.release()
        teleop.reset()
        _cancel_arm_home_motion()
        mode["value"] = mode_name
        base_command["value"] = (0., 0., 0.)
        if mode_name == "high_grasp":
            low_lock.lock_current()
        else:
            target_height["value"] = height
            low_lock.request(target_height["value"], height_error=.005)
        _print_mode()

    def _publish_full_status(*, status=None, message=None):
        if not full_status["path"]:
            return
        if status is not None:
            full_status["status"] = status
        if message is not None:
            full_status["message"] = message
        data = dict(control="model" if control["model"] else "manual",
                    mode=mode["value"],
                    task_phase=policy_client.task_phase if policy_client is not None else None,
                    arm_ready=_arm_enabled(), homing=arm_home_motion["active"],
                    lock_phase=low_lock.phase, execute_steps=args_cli.execute_steps,
                    status=full_status["status"], message=full_status["message"])
        if data == full_status["last"]:
            return
        try:
            path = Path(full_status["path"])
            temporary = path.with_suffix(".tmp")
            temporary.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
            temporary.replace(path)
            full_status["last"] = data
        except OSError:
            pass  # The read-only status window must never interrupt control.

    def _toggle_task_phase():
        if (not control["model"] and policy_client is not None
                and policy_client.toggle_task_phase()):
            _publish_full_status()

    def _set_control(model_control):
        if control["model"] == model_control:
            return
        control["model"] = model_control
        teleop.reset()
        base_command["value"] = (0.0, 0.0, 0.0)
        if policy_client is not None:
            policy_client.set_model_control(model_control)
        _publish_full_status()
        print(f"[CONTROL]: {'model' if model_control else 'manual'}", flush=True)

    if args_cli.headless:
        teleop = HeadlessSe2Input()
        print("[INFO]: Headless mode: keyboard input disabled; base command remains zero.", flush=True)
    else:
        teleop = StateBasedSe2Keyboard(
            Se2KeyboardCfg(
                v_x_sensitivity=1.0,
                v_y_sensitivity=0.0,
                omega_z_sensitivity=1.0,
                sim_device=device,
            )
        )
    if not args_cli.headless:
        teleop.arm_is_ready = _arm_enabled
        teleop.manual_control_enabled = lambda: not control["model"]
    if args_cli.policy_full:
        teleop.add_callback("F4", lambda: _set_control(False))
        teleop.add_callback("F5", lambda: _set_control(True))
    teleop.add_callback("1", _set_mode("walk"))
    teleop.add_callback("2", _set_mode("stair"))
    teleop.add_callback("4", lambda: _grasp("low_grasp", args_cli.min_height))
    teleop.add_callback("5", lambda: _grasp("high_grasp", args_cli.stand_height))
    teleop.add_callback("SPACE", teleop.reset)
    teleop.add_callback("ESCAPE", lambda: should_quit.__setitem__("value", True))
    teleop.add_callback("Q", lambda: should_quit.__setitem__("value", True))


    teleop.add_callback("R", _arm_delta(0, -args_cli.arm_step))
    teleop.add_callback("T", _arm_delta(0, args_cli.arm_step))
    teleop.add_callback("E", _arm_delta(1, -args_cli.arm_step))
    teleop.add_callback("G", _arm_delta(1, args_cli.arm_step))
    teleop.add_callback("V", _arm_delta(2, -args_cli.arm_step))
    teleop.add_callback("B", _arm_delta(2, args_cli.arm_step))
    teleop.add_callback("N", _arm_delta(3, -args_cli.arm_step))
    teleop.add_callback("M", _arm_delta(3, args_cli.arm_step))
    teleop.add_callback("I", _arm_delta(4, -args_cli.arm_step))
    teleop.add_callback("O", _arm_delta(4, args_cli.arm_step))
    teleop.add_callback("K", _arm_delta(5, -args_cli.arm_step))
    teleop.add_callback("L", _arm_delta(5, args_cli.arm_step))
    teleop.add_callback("C", _set_gripper(args_cli.gripper_closed, "closed"))
    teleop.add_callback("P", _set_gripper(args_cli.gripper_open, "open"))
    teleop.add_callback("H", _set_arm_home)
    if collector:
        teleop.add_callback("F1", collector.start)
        teleop.add_callback("F2", collector.next_stage)
        teleop.add_callback("F3", collector.stop)
        print("[RECORD] F1 start / F2 next stage / F3 save. Latest state + last submitted command, 50 Hz.", flush=True)
    elif args_cli.policy_full:
        teleop.add_callback("F2", _toggle_task_phase)

    _publish_full_status()

    print("[INFO]: vel_cmd=[vx,wz]; ee/ee_cmd=[x,y,z,roll,pitch,yaw] in arm_base, meters/degrees; ee_cmd retains the latest requested pose; ik reports execution status.", flush=True)
    print("[INFO]: Keyboard movement is state-based; --vx and --wz are hard speed limits.", flush=True)
    print("WALK/STAIR: W/S forward/back, A/D turn, Space stop.", flush=True)
    print("MODE KEYS: 1=walk, 2=stair, 4=low grasp, 5=high grasp.", flush=True)
    print("MODES 4/5: exit through mode 1 only; direct 4<->5 or 4/5->2 is blocked.", flush=True)
    print("ARM after grasp mode is ready: R/T j1, E/G j2, V/B j3, N/M j4, I/O j5, K/L j6, C close, P open, H home 1s.", flush=True)
    if args_cli.policy_full:
        print("CONTROL: model by default; F4 manual, F5 model; F2 toggles phase 1/2 in manual control; Q/Esc quit.", flush=True)

    raw_env.reset()
    if task03_authored_pose is not None:
        task03_init_pos, task03_init_rot = task03_authored_pose
        root_pose = torch.tensor([[*task03_init_pos, *task03_init_rot]], dtype=torch.float32, device=device)
        robot.write_root_pose_to_sim(root_pose)
        robot.write_root_velocity_to_sim(torch.zeros((1, 6), dtype=torch.float32, device=device))
        raw_env.scene.write_data_to_sim()
        raw_env.sim.forward()
        print(
            f"[INFO]: robot reset to task03 authored pose pos={task03_init_pos} rot={task03_init_rot}",
            flush=True,
        )

    print("MODES 4/5: reach low/high height then lock base/legs/wheels; 1 releases and immediately walks. Previous IK keys 4/5 removed.", flush=True)
    _write_arm_home_to_sim()
    _set_height(raw_env, target_height["value"])
    _print_mode()

    if args_cli.policy_host:
        from full_client import FullClient
        _publish_full_status(status="connecting")
        try:
            policy_client = FullClient(args_cli.policy_host, args_cli.policy_port,
                                       arm_joint_ids, args_cli.gripper_closed, args_cli.gripper_open,
                                       args_cli.vx, args_cli.wz, execute_steps=args_cli.execute_steps)
        except Exception as exc:
            _publish_full_status(status="error", message=str(exc))
            raise
        policy_client.set_model_control(control["model"])
        _publish_full_status(status="running")
    step = 0
    dt = raw_env.step_dt
    ik = ArmIK(robot, arm_joint_ids, ARM_HARD_LIMITS)
    try:
        while simulation_app.is_running() and not should_quit["value"]:
            start = time.time()
            command = _scale_walk_command(teleop.advance())
            if args_cli.command_schedule:
                while schedule and step >= schedule[0]["step"]:
                    item = schedule.pop(0)
                    if item["mode"] in ("low_grasp", "high_grasp"):
                        _grasp(item["mode"], args_cli.min_height if item["mode"] == "low_grasp" else args_cli.stand_height)
                    else:
                        _set_mode(item["mode"])()
                    scheduled_command = (float(item["vx"]), 0.0, float(item["wz"]))
                command = scheduled_command
            full_arm_action = None
            if policy_client and control["model"]:
                command = (0.0, 0.0, 0.0)
                if step >= args_cli.record_warmup:
                    policy_action = policy_client.step(raw_env, mode["value"])
                    if policy_action is not None:
                        command, next_mode, joints, gripper = policy_action
                        full_arm_action = (joints, gripper)
                        if next_mode != mode["value"]:
                            if next_mode in ("low_grasp", "high_grasp"):
                                _grasp(next_mode, args_cli.min_height if next_mode == "low_grasp" else args_cli.stand_height)
                            else:
                                _set_mode(next_mode)()
            if collector and args_cli.record_start and step >= args_cli.record_warmup and collector.completed == 0:
                collector.start()
            if collector:
                collector.capture_observation(step * dt, mode["value"])
            if mode["value"] not in ("walk", "stair"):
                command = (0.0, 0.0, 0.0)
            base_command["value"] = command

            _set_base_command(raw_env, command if mode["value"] in ("walk", "stair") else (0.0, 0.0, 0.0))
            _set_height(raw_env, target_height["value"])
            if mode["value"] in ("low_grasp", "high_grasp") and not low_lock.locked:
                _cancel_arm_home_motion()
            _update_arm_home_motion()
            if getattr(teleop, "ik_cancel", False):
                if (ik.settling and not arm_home_motion["active"]
                        and not getattr(teleop, "joint_keys", set())):
                    arm_target[0,:6] = ik.stop()
                else:
                    ik.cancel()
                teleop.ik_cancel = False
            pending = getattr(teleop, "ik_pending", [])
            pressed = getattr(teleop, "ik_keys", set())
            if pending and _arm_enabled():
                _cancel_arm_home_motion()
            if (not control["model"] and _arm_enabled()
                    and not arm_home_motion["active"] and step >= ik_pause["until"]
                    and not getattr(teleop, "joint_keys", set())):
                arm_target[0,:6] = ik.step(pressed, dt, arm_target[0,:6], pending)
            else:
                ik.cancel()
            if hasattr(teleop, "ik_pending"):
                teleop.ik_pending.clear()
            if full_arm_action is not None:
                # Use the same arm/gripper gate as keyboard control. Walking and
                # stair predictions must not override or cancel automatic homing.
                if _arm_enabled():
                    _cancel_arm_home_motion()
                    joints, gripper = full_arm_action
                    arm_target[0, :6] = torch.as_tensor(joints, dtype=torch.float32, device=device)
                    gripper_value["value"] = args_cli.gripper_closed + gripper * (
                        args_cli.gripper_open - args_cli.gripper_closed)
            _clamp_arm_target()
            robot.set_joint_position_target(arm_target[:, :6], joint_ids=arm_joint_ids[:6])
            robot.set_joint_velocity_target(ik.velocity.unsqueeze(0), joint_ids=arm_joint_ids[:6])
            # Official PhysX gravity compensation. Floating-base tensors contain
            # six root coordinates before the articulation joint coordinates.
            arm_effort = torch.zeros_like(arm_target[:, :6])
            if _arm_enabled():
                gravity = robot.root_physx_view.get_gravity_compensation_forces()
                offset = 0 if robot.is_fixed_base else 6
                arm_effort = gravity[:, [j + offset for j in arm_joint_ids[:6]]]
            robot.set_joint_effort_target(arm_effort, joint_ids=arm_joint_ids[:6])
            robot.set_joint_position_target(arm_target[:, 6:8], joint_ids=arm_joint_ids[6:8])

            if collector:
                collector.command_submitted(command, arm_target, mode["value"], arm_ready=_arm_enabled())

            obs = _get_policy_obs(raw_env, env)
            with torch.inference_mode():
                if mode["value"] == "walk":
                    actions = walk_actor(_slice_walk_obs(obs))
                    _, _, dones, _ = env.step(actions)
                elif mode["value"] == "stair":
                    actions = stair_actor(_slice_walk_obs(obs))
                    _, _, dones, _ = env.step(actions)
                else:
                    actions = torch.zeros_like(raw_env.action_manager.action) if low_lock.locked else stabilizer_actor(obs)
                    _, _, dones, _ = env.step(actions)

            if mode["value"] in ("low_grasp", "high_grasp"):
                if bool(dones.any().item()):
                    low_lock.release()
                    _set_mode("walk", home_on_exit=False)()
                else:
                    low_lock.update(dt)

            _publish_full_status()
            if policy_client and bool(dones.any().item()):
                print("[POLICY] simulation reset; stopping", flush=True)
                break
            if collector:
                # env.step may auto-reset: never pair a reset state with the previous episode command.
                if bool(dones.any().item()):
                    collector.stop()
                    print("[RECORD] reset detected; excluded reset frame and stopped run", flush=True)
                    break
                collector.sample()
                if collector.writer and args_cli.record_frames > 0 and collector.writer.count >= args_cli.record_frames:
                    collector.stop()
                    break

            body_z = float(robot.data.root_pos_w[0, 2].item())
            if step % 50 == 0:
                try:
                    ground_z = low_lock._ground_height(robot.data.root_pos_w[0].tolist())
                    height = body_z - ground_z if ground_z is not None else float("nan")
                except Exception:
                    height = float("nan")
                quat = robot.data.root_quat_w[0].double()
                quat_norm = quat.norm().item()
                tilt = float("nan")
                if bool(torch.isfinite(quat).all()) and quat_norm > 1e-8:
                    quat = quat / quat_norm
                    tilt = math.degrees(math.acos(max(-1., min(1., 1 - 2 * (quat[1]**2 + quat[2]**2).item()))))
                ee_quat = quat_mul(
                    quat_conjugate(robot.data.body_quat_w[0, ik.base]),
                    robot.data.body_quat_w[0, ik.fingers[0]])
                ee = _ee_log_pose(ik.position(), ee_quat)
                ee_cmd = _ee_log_pose(*ik.target) if ik.target is not None else None
                print(
                    f"[STEP] step={step:05d} mode={mode['value']} vel_cmd=[{command[0]:.2f},{command[2]:.2f}] "
                    f"height_cmd={target_height['value']:.3f} height={height:.3f} tilt={tilt:.2f} lock={low_lock.phase} "
                    f"joint_drift={float((robot.data.joint_pos[:, low_lock.joints] - low_lock.positions).abs().max()) if low_lock.locked else 0.:.6f}rad "
                    f"drift_joint={robot.joint_names[low_lock.joints[int((robot.data.joint_pos[:, low_lock.joints] - low_lock.positions).abs().argmax())]] if low_lock.locked else 'none'} "
                    f"ee={ee} ee_cmd={ee_cmd} ik={ik.status} path={ik.path_kind} "
                    f"done={int(dones.sum().item())}",
                    flush=True,
                )

            step += 1
            if args_cli.max_steps > 0 and step >= args_cli.max_steps:
                break
            if args_cli.real_time:
                sleep_time = dt - (time.time() - start)
                if sleep_time > 0.0:
                    time.sleep(sleep_time)
    except Exception as exc:
        _publish_full_status(status="error", message=str(exc))
        if collector:
            collector.stop(publish=False)
        raise
    finally:
        if args_cli.policy_full and policy_client is not None:
            policy_client.close()
        low_lock.release()
        raw_env.sim.step = original_sim_step
        if collector:
            collector.stop()
        env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()

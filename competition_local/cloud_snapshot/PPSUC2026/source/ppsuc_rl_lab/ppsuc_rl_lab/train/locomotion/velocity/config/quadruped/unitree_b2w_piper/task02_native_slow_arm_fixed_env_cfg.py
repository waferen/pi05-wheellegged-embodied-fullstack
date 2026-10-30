 
"""Task02 (living room) teleop env cfg: native b2w_piper from task02.usd, slow arm-fixed.

The base class UnitreeB2WPiperTaskSceneNativeSlowArmFixedEnvCfg hardcodes the
Bomb-Disposal task.usd paths, so this subclass re-points the same scene asset
structure at the user's task02.usd living-room scene.
"""

import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg
from isaaclab.utils import configclass

from ppsuc_rl_lab.project_paths import ROBOT_MODEL_ROOT
from ppsuc_rl_lab.train.locomotion.velocity.config.quadruped.unitree_b2w_piper.cerebellum_env_cfg import (
    SQUAT_COMMAND_THRESHOLD,
    SQUAT_HEIGHT_FINAL,
    STAND_HEIGHT,
    UnitreeB2WPiperCerebellumFinalFlatEnvCfg,
)
from ppsuc_rl_lab.train.locomotion.velocity.config.quadruped.unitree_b2w_piper.task_scene_native_slow_arm_fixed_env_cfg import (
    UnitreeB2WPiperTaskSceneNativeSlowArmFixedEnvCfg,
)

TASK02_SCENE_USD = str(ROBOT_MODEL_ROOT / "scene/task02.usd")
# task02.usd already contains /World/b2w_piper; spawning it under /World/task_scene
# re-nests the prim to /World/task_scene/b2w_piper (same nesting as task.usd).
TASK02_ROBOT_PRIM = "/World/task_scene/b2w_piper"
TASK02_BASE_PRIM = TASK02_ROBOT_PRIM + "/base_link"
# Pose authored in task02.usd (base_link world position, verified collision-clear).
TASK02_INIT_POS = (4.28, 3.61, 0.75)


@configclass
class UnitreeB2WPiperTask02NativeSlowArmFixedEnvCfg(UnitreeB2WPiperTaskSceneNativeSlowArmFixedEnvCfg):
    def __post_init__(self):
        super().__post_init__()

        # Re-point the scene asset spawned by the base class at task02.usd.
        self.scene.task_scene.spawn.usd_path = TASK02_SCENE_USD

        # Native robot prim inside task02.usd, nested under the spawned scene root.
        self.scene.robot.prim_path = TASK02_ROBOT_PRIM
        self.scene.robot.articulation_root_prim_path = "/base_link"
        self.scene.robot.init_state.pos = TASK02_INIT_POS

        # Height scanners (if any were re-enabled upstream) track the native base.
        if self.scene.height_scanner is not None:
            self.scene.height_scanner.prim_path = TASK02_BASE_PRIM
            self.scene.height_scanner.mesh_prim_paths = ["/World/task_scene"]
        if self.scene.height_scanner_base is not None:
            self.scene.height_scanner_base.prim_path = TASK02_BASE_PRIM
            self.scene.height_scanner_base.mesh_prim_paths = ["/World/task_scene"]


@configclass
class UnitreeB2WPiperTask02NativeCerebellumEnvCfg(UnitreeB2WPiperCerebellumFinalFlatEnvCfg):
    """Task02 scene using the trained stand/squat cerebellum observation contract."""

    def __post_init__(self):
        super().__post_init__()

        self.scene.env_spacing = 0.0
        self.scene.terrain = None
        robot_cfg = self.scene.robot
        self.scene.__dict__.pop("robot", None)
        self.scene.task_scene = AssetBaseCfg(
            prim_path="/World/task_scene",
            spawn=sim_utils.UsdFileCfg(usd_path=TASK02_SCENE_USD),
            collision_group=-1,
        )
        self.scene.robot = robot_cfg

        self.scene.robot.prim_path = TASK02_ROBOT_PRIM
        self.scene.robot.spawn = None
        self.scene.robot.articulation_root_prim_path = "/base_link"
        self.scene.robot.init_state.pos = TASK02_INIT_POS
        self.scene.robot.init_state.rot = (0.0, 0.0, 0.0, 1.0)

        if self.scene.height_scanner is not None:
            self.scene.height_scanner.prim_path = TASK02_BASE_PRIM
            self.scene.height_scanner.mesh_prim_paths = ["/World/task_scene"]
        if self.scene.height_scanner_base is not None:
            self.scene.height_scanner_base.prim_path = TASK02_BASE_PRIM
            self.scene.height_scanner_base.mesh_prim_paths = ["/World/task_scene"]

        self.scene.contact_forces = None
        self.terminations.illegal_contact = None
        self.terminations.bad_orientation = None
        self.terminations.base_height_below_minimum = None
        self.rewards.undesired_contacts = None
        self.rewards.contact_forces = None
        self.rewards.wheel_vel_penalty = None
        self.rewards.feet_air_time = None
        self.rewards.feet_contact = None
        self.rewards.feet_contact_without_cmd = None
        self.rewards.feet_stumble = None
        self.rewards.feet_slide = None

        # Teleop owns the arm through IK; keep the trained policy's arm-joint
        # observations, but disable the training-only sine target writer.
        self.commands.arm_sine = None

        # Manual Z/X key commands should persist until changed by the operator.
        self.commands.body_height.resampling_time_range = (1.0e9, 1.0e9)
        self.commands.body_height.ranges.height = (SQUAT_HEIGHT_FINAL, STAND_HEIGHT)
        self.commands.body_height.discrete_values = [STAND_HEIGHT]
        self.commands.body_height.zero_base_below_height = SQUAT_COMMAND_THRESHOLD

        self.events.randomize_reset_base.params["pose_range"] = {
            "x": (0.0, 0.0), "y": (0.0, 0.0), "z": (0.0, 0.0),
            "roll": (0.0, 0.0), "pitch": (0.0, 0.0), "yaw": (0.0, 0.0),
        }
        self.events.randomize_reset_base.params["velocity_range"] = {
            "x": (0.0, 0.0), "y": (0.0, 0.0), "z": (0.0, 0.0),
            "roll": (0.0, 0.0), "pitch": (0.0, 0.0), "yaw": (0.0, 0.0),
        }
        self.events.randomize_push_robot = None
        self.events.randomize_apply_external_force_torque = None
        self.events.randomize_rigid_body_material = None
        self.events.randomize_rigid_body_mass_base = None
        self.events.randomize_rigid_body_mass_others = None
        self.events.randomize_com_positions = None
        self.events.randomize_actuator_gains = None

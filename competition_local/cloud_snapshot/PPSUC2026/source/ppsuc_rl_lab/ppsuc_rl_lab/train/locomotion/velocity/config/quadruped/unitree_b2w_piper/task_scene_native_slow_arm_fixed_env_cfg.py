import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg
from isaaclab.utils import configclass

from ppsuc_rl_lab.project_paths import ROBOT_MODEL_ROOT
from ppsuc_rl_lab.train.locomotion.velocity.config.quadruped.unitree_b2w_piper.flat_slow_arm_fixed_env_cfg import (
    UnitreeB2WPiperFlatSlowArmFixedEnvCfg,
)

TASK_SCENE_USD = str(ROBOT_MODEL_ROOT / "scene/task.usd")
TASK_SCENE_NATIVE_ROBOT_PRIM = "/World/task_scene/Bomb_Disposal_Task_v02/b2w_piper"
TASK_SCENE_NATIVE_BASE_PRIM = TASK_SCENE_NATIVE_ROBOT_PRIM + "/base_link"


@configclass
class UnitreeB2WPiperTaskSceneNativeSlowArmFixedEnvCfg(UnitreeB2WPiperFlatSlowArmFixedEnvCfg):
    def __post_init__(self):
        super().__post_init__()

        self.scene.env_spacing = 0.0
        self.scene.terrain = None
        robot_cfg = self.scene.robot
        self.scene.__dict__.pop("robot", None)
        self.scene.task_scene = AssetBaseCfg(
            prim_path="/World/task_scene",
            spawn=sim_utils.UsdFileCfg(usd_path=TASK_SCENE_USD),
            collision_group=-1,
        )
        self.scene.robot = robot_cfg

        self.scene.robot.prim_path = TASK_SCENE_NATIVE_ROBOT_PRIM
        self.scene.robot.spawn = None
        self.scene.robot.articulation_root_prim_path = "/base_link"
        self.scene.robot.init_state.pos = (12.190385, 6.832485, 0.894875)
        self.scene.robot.init_state.rot = (0.0, 0.0, 0.0, 1.0)

        if self.scene.height_scanner is not None:
            self.scene.height_scanner.prim_path = TASK_SCENE_NATIVE_BASE_PRIM
            self.scene.height_scanner.mesh_prim_paths = ["/World/task_scene"]
        if self.scene.height_scanner_base is not None:
            self.scene.height_scanner_base.prim_path = TASK_SCENE_NATIVE_BASE_PRIM
            self.scene.height_scanner_base.mesh_prim_paths = ["/World/task_scene"]

        self.scene.contact_forces = None
        self.terminations.illegal_contact = None
        self.rewards.undesired_contacts = None
        self.rewards.contact_forces = None
        self.rewards.wheel_vel_penalty = None
        self.rewards.feet_air_time = None
        self.rewards.feet_contact = None
        self.rewards.feet_contact_without_cmd = None
        self.rewards.feet_stumble = None
        self.rewards.feet_slide = None

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

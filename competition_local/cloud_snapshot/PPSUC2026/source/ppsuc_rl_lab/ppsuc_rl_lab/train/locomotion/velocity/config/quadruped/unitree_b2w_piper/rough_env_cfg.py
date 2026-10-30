from isaaclab.envs import mdp
from isaaclab.managers import SceneEntityCfg
from isaaclab.terrains import TerrainImporter
from isaaclab.utils import configclass

from ppsuc_rl_lab.assets.robots import UNITREE_B2W_PIPER_CFG
from ppsuc_rl_lab.train.locomotion.velocity.config.quadruped.unitree_b2.rough_env_cfg import UnitreeB2RoughEnvCfg


class UniqueTileTerrainImporter(TerrainImporter):
    pass


@configclass
class B2WPiperActionsCfg:
    joint_leg = mdp.JointPositionActionCfg(
        asset_name="robot",
        joint_names=[
            "FR_hip_joint", "FR_thigh_joint", "FR_calf_joint",
            "FL_hip_joint", "FL_thigh_joint", "FL_calf_joint",
            "RR_hip_joint", "RR_thigh_joint", "RR_calf_joint",
            "RL_hip_joint", "RL_thigh_joint", "RL_calf_joint",
        ],
        scale={".*_hip_joint": 0.125, "^(?!.*_hip_joint).*": 0.25},
        use_default_offset=True,
        clip={".*": (-100.0, 100.0)},
        preserve_order=True,
    )
    joint_wheel = mdp.JointVelocityActionCfg(
        asset_name="robot",
        joint_names=["FR_foot_joint", "FL_foot_joint", "RR_foot_joint", "RL_foot_joint"],
        scale=8.0,
        clip={".*": (-50.0, 50.0)},
        preserve_order=True,
    )


@configclass
class UnitreeB2WPiperRoughEnvCfg(UnitreeB2RoughEnvCfg):
    base_link_name = "base_link"
    foot_link_name = ".*_foot"
    leg_joint_names = [
        "FR_hip_joint", "FR_thigh_joint", "FR_calf_joint",
        "FL_hip_joint", "FL_thigh_joint", "FL_calf_joint",
        "RR_hip_joint", "RR_thigh_joint", "RR_calf_joint",
        "RL_hip_joint", "RL_thigh_joint", "RL_calf_joint",
    ]
    wheel_joint_names = ["FR_foot_joint", "FL_foot_joint", "RR_foot_joint", "RL_foot_joint"]
    arm_joint_names = [
        "arm_joint1", "arm_joint2", "arm_joint3", "arm_joint4",
        "arm_joint5", "arm_joint6", "arm_joint7", "arm_joint8",
    ]
    joint_names = leg_joint_names + wheel_joint_names

    def __post_init__(self):
        super().__post_init__()

        self.scene.robot = UNITREE_B2W_PIPER_CFG.replace(
            prim_path="{ENV_REGEX_NS}/Robot",
            init_state=UNITREE_B2W_PIPER_CFG.init_state.replace(pos=(0.0, 0.0, 0.78)),
        )
        self.scene.height_scanner.prim_path = "{ENV_REGEX_NS}/Robot/" + self.base_link_name
        self.scene.height_scanner_base.prim_path = "{ENV_REGEX_NS}/Robot/" + self.base_link_name
        self.scene.env_spacing = 4.0
        self.scene.terrain.env_spacing = 4.0
        self.scene.terrain.class_type = UniqueTileTerrainImporter
        self.scene.terrain.use_terrain_origins = True

        self.commands.base_velocity.debug_vis = False
        self.commands.base_velocity.heading_command = False
        self.commands.base_velocity.rel_heading_envs = 0.0
        self.commands.base_velocity.ranges.lin_vel_x = (0.2, 0.8)
        self.commands.base_velocity.ranges.lin_vel_y = (0.0, 0.0)
        self.commands.base_velocity.ranges.ang_vel_z = (-0.8, 0.8)
        self.commands.base_velocity.ranges.heading = (0.0, 0.0)

        self.actions = B2WPiperActionsCfg()
        self.observations.policy.joint_pos.params["asset_cfg"].joint_names = self.joint_names
        self.observations.policy.joint_vel.params["asset_cfg"].joint_names = self.joint_names
        self.observations.critic.joint_pos.params["asset_cfg"].joint_names = self.joint_names
        self.observations.critic.joint_vel.params["asset_cfg"].joint_names = self.joint_names
        self.observations.critic.height_scan = None

        self.rewards.base_height_l2.params["target_height"] = 0.72
        self.rewards.joint_torques_l2.params["asset_cfg"].joint_names = self.joint_names
        self.rewards.joint_acc_l2.params["asset_cfg"].joint_names = self.joint_names
        self.rewards.joint_pos_limits.params["asset_cfg"].joint_names = self.leg_joint_names
        self.rewards.joint_power.params["asset_cfg"].joint_names = self.joint_names
        self.rewards.stand_still.params["asset_cfg"].joint_names = self.leg_joint_names
        self.rewards.joint_pos_penalty.params["asset_cfg"].joint_names = self.leg_joint_names
        self.rewards.joint_mirror.params["mirror_joints"] = [
            ["FR_(hip|thigh|calf).*", "RL_(hip|thigh|calf).*"],
            ["FL_(hip|thigh|calf).*", "RR_(hip|thigh|calf).*"],
        ]
        self.rewards.wheel_vel_penalty.weight = -0.05
        self.rewards.wheel_vel_penalty.params["asset_cfg"].joint_names = self.wheel_joint_names
        self.rewards.wheel_vel_penalty.params["sensor_cfg"].body_names = [self.foot_link_name]

        self.events.randomize_reset_joints.params["asset_cfg"] = SceneEntityCfg("robot", joint_names=self.joint_names)
        if self.events.randomize_actuator_gains is not None:
            self.events.randomize_actuator_gains.params["asset_cfg"].joint_names = self.joint_names


        # ------------------------------Rewards (feet/contact body names) ------------------------------
        if getattr(self.rewards, "feet_air_time_variance", None) is not None:
            self.rewards.feet_air_time_variance.params["sensor_cfg"].body_names = [self.foot_link_name]
        if getattr(self.rewards, "feet_air_time", None) is not None:
            self.rewards.feet_air_time.params["sensor_cfg"].body_names = [self.foot_link_name]
        if getattr(self.rewards, "feet_contact", None) is not None:
            self.rewards.feet_contact.params["sensor_cfg"].body_names = [self.foot_link_name]
        if getattr(self.rewards, "feet_contact_without_cmd", None) is not None:
            self.rewards.feet_contact_without_cmd.params["sensor_cfg"].body_names = [self.foot_link_name]
        if getattr(self.rewards, "feet_stumble", None) is not None:
            self.rewards.feet_stumble.params["sensor_cfg"].body_names = [self.foot_link_name]
        if getattr(self.rewards, "feet_slide", None) is not None:
            self.rewards.feet_slide.params["sensor_cfg"].body_names = [self.foot_link_name]
            self.rewards.feet_slide.params["asset_cfg"].body_names = [self.foot_link_name]
        if getattr(self.rewards, "feet_height", None) is not None:
            self.rewards.feet_height.params["asset_cfg"].body_names = [self.foot_link_name]
        if getattr(self.rewards, "feet_height_body", None) is not None:
            self.rewards.feet_height_body.params["asset_cfg"].body_names = [self.foot_link_name]
        if getattr(self.rewards, "undesired_contacts", None) is not None:
            self.rewards.undesired_contacts.params["sensor_cfg"].body_names = [f"^(?!.*{self.foot_link_name}).*"]
        if getattr(self.rewards, "contact_forces", None) is not None:
            self.rewards.contact_forces.params["sensor_cfg"].body_names = [self.foot_link_name]


        # Disable zero-weight rewards inherited from B2 rough env (body names may not match B2W+Piper).
        self.disable_zero_weight_rewards()

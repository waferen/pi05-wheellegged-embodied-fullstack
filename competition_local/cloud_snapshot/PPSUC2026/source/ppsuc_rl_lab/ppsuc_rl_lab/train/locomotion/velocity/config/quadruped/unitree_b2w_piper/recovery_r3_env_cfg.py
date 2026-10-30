from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils import configclass

import ppsuc_rl_lab.train.locomotion.velocity.mdp as mdp
from ppsuc_rl_lab.train.locomotion.velocity.config.quadruped.unitree_b2w_piper.collision_resilience_d1_env_cfg import (
    bad_orientation_after_grace,
    root_height_below_minimum_after_grace,
)
from ppsuc_rl_lab.train.locomotion.velocity.config.quadruped.unitree_b2w_piper.flat_env_cfg import (
    UnitreeB2WPiperFlatEnvCfg,
)


def base_xy_out_of_recovery_radius(env, max_distance: float, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")):
    distance = mdp.root_xy_distance_from_env_origin(env, asset_cfg)
    return distance > max_distance


@configclass
class UnitreeB2WPiperFlatRecoveryR3EnvCfg(UnitreeB2WPiperFlatEnvCfg):
    """R3: stabilize first, then move on flat ground.

    The command is zero for the first 3 seconds after reset. When recovery ends, a fresh movement
    command is sampled. This teaches the policy to use the same controller for standing recovery
    and later commanded locomotion.
    """

    def __post_init__(self):
        super().__post_init__()

        self.scene.env_spacing = 4.0
        self.scene.terrain.env_spacing = 4.0

        self.commands.base_velocity.class_type = mdp.RecoveryThenForwardVelocityCommand
        self.commands.base_velocity.recovery_time_s = 3.0
        self.commands.base_velocity.debug_vis = False
        self.commands.base_velocity.heading_command = False
        self.commands.base_velocity.rel_heading_envs = 0.0
        self.commands.base_velocity.rel_standing_envs = 0.0
        self.commands.base_velocity.resampling_time_range = (3.0, 5.0)
        self.commands.base_velocity.ranges.lin_vel_x = (0.20, 0.75)
        self.commands.base_velocity.ranges.lin_vel_y = (0.0, 0.0)
        self.commands.base_velocity.ranges.ang_vel_z = (-0.45, 0.45)
        self.commands.base_velocity.ranges.heading = None

        self.events.randomize_apply_external_force_torque = None
        self.events.randomize_push_robot = None
        self.events.randomize_reset_base.params = {
            "pose_range": {
                "x": (-0.05, 0.05),
                "y": (-0.05, 0.05),
                "z": (0.0, 0.08),
                "roll": (-0.10, 0.10),
                "pitch": (-0.10, 0.10),
                "yaw": (-3.14, 3.14),
            },
            "velocity_range": {
                "x": (-0.15, 0.15),
                "y": (-0.15, 0.15),
                "z": (-0.08, 0.08),
                "roll": (-0.15, 0.15),
                "pitch": (-0.15, 0.15),
                "yaw": (-0.15, 0.15),
            },
        }

        self.terminations.illegal_contact = None
        self.terminations.bad_orientation = DoneTerm(
            func=bad_orientation_after_grace,
            params={"asset_cfg": SceneEntityCfg("robot"), "limit_angle": 1.05, "grace_time_s": 1.0},
        )
        self.terminations.base_height_below_minimum = DoneTerm(
            func=root_height_below_minimum_after_grace,
            params={"asset_cfg": SceneEntityCfg("robot"), "minimum_height": 0.38, "grace_time_s": 1.0},
        )
        self.terminations.base_xy_out_of_recovery_radius = DoneTerm(
            func=base_xy_out_of_recovery_radius,
            params={"max_distance": 2.0, "asset_cfg": SceneEntityCfg("robot")},
        )

        self.rewards.track_lin_vel_xy_exp.weight = 4.0
        self.rewards.track_ang_vel_z_exp.weight = 1.2
        self.rewards.upward.weight = 4.5
        self.rewards.ang_vel_xy_l2.weight = -0.10
        self.rewards.lin_vel_z_l2.weight = -2.0
        self.rewards.action_rate_l2.weight = -0.015
        self.rewards.joint_torques_l2.weight = -1.0e-5
        self.rewards.joint_acc_l2.weight = -1.0e-7
        self.rewards.joint_power.weight = -1.0e-5
        self.rewards.joint_pos_limits.weight = -5.0
        self.rewards.stand_still.weight = -1.0
        self.rewards.joint_pos_penalty.weight = -1.0
        self.rewards.wheel_vel_penalty.weight = -0.05
        self.rewards.undesired_contacts.weight = -0.5
        self.rewards.contact_forces.weight = -1.0e-4
        self.rewards.is_terminated.weight = -25.0

        self.rewards.recovery_xy_radius = RewTerm(
            func=mdp.root_xy_distance_l2_from_env_origin,
            weight=-6.0,
            params={"asset_cfg": SceneEntityCfg("robot")},
        )
        self.rewards.recovery_upright = RewTerm(
            func=mdp.recovery_upright_reward,
            weight=3.0,
            params={"recovery_time_s": 3.0, "asset_cfg": SceneEntityCfg("robot")},
        )
        self.rewards.recovery_still = RewTerm(
            func=mdp.recovery_still_reward,
            weight=4.0,
            params={
                "recovery_time_s": 3.0,
                "lin_vel_std": 0.25,
                "ang_vel_std": 0.8,
                "asset_cfg": SceneEntityCfg("robot"),
            },
        )
        self.rewards.recovery_unstable_after_deadline = RewTerm(
            func=mdp.recovery_unstable_after_deadline,
            weight=-6.0,
            params={
                "recovery_time_s": 3.0,
                "max_tilt_angle": 0.35,
                "min_height": 0.45,
                "max_lin_vel_xy": 0.30,
                "max_ang_vel_xy": 0.9,
                "asset_cfg": SceneEntityCfg("robot"),
            },
        )

        self.disable_zero_weight_rewards()

 
"""Unified low-level controller training for B2W+Piper.

This task trains one locomotion policy to track velocity commands, track a
body-height command, and reject manipulator motion disturbances. The arm is not
part of the action space; it is driven by a command term during training.
"""

from isaaclab.envs import mdp
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils import configclass

import ppsuc_rl_lab.train.locomotion.velocity.mdp as ppsuc_mdp
from ppsuc_rl_lab.train.locomotion.velocity.config.quadruped.unitree_b2w_piper.arm_stability_env_cfg import (
    ARM_JOINTS,
    UnitreeB2WPiperArmStabilityEnvCfg,
)
from ppsuc_rl_lab.train.locomotion.velocity.config.quadruped.unitree_b2w_piper.impulse_recovery_c1_env_cfg import (
    bad_orientation_persistent_after_grace,
    root_height_persistent_below_minimum_after_grace,
)

STAND_HEIGHT = 0.59
STAND_COMMAND_THRESHOLD = 0.55
SQUAT_COMMAND_THRESHOLD = 0.40
SQUAT_HEIGHT_STAGE1 = 0.50
SQUAT_HEIGHT_STAGE2 = 0.42
SQUAT_HEIGHT_STAGE3 = 0.34
SQUAT_HEIGHT_STAGE4 = 0.28
SQUAT_HEIGHT_FINAL = STAND_HEIGHT * 0.40


@configclass
class UnitreeB2WPiperCerebellumFlatEnvCfg(UnitreeB2WPiperArmStabilityEnvCfg):
    """Stage 1: learn height tracking and arm-disturbance rejection."""

    squat_height: float = SQUAT_HEIGHT_STAGE1
    arm_sine_amplitude: float = 0.22
    arm_sine_frequency_hz: float = 0.45

    def __post_init__(self):
        super().__post_init__()

        self.commands.base_velocity.heading_command = False
        self.commands.base_velocity.rel_heading_envs = 0.0
        self.commands.base_velocity.rel_standing_envs = 0.35
        self.commands.base_velocity.resampling_time_range = (3.0, 5.0)
        self.commands.base_velocity.ranges.lin_vel_x = (-0.45, 0.75)
        self.commands.base_velocity.ranges.lin_vel_y = (0.0, 0.0)
        self.commands.base_velocity.ranges.ang_vel_z = (-0.60, 0.60)
        self.commands.base_velocity.ranges.heading = None

        self.commands.body_height = ppsuc_mdp.BodyHeightCommandCfg(
            resampling_time_range=(3.0, 5.0),
            debug_vis=False,
            ranges=ppsuc_mdp.BodyHeightCommandCfg.Ranges(height=(self.squat_height, STAND_HEIGHT)),
            discrete_values=[self.squat_height, self.squat_height, STAND_HEIGHT, STAND_HEIGHT],
            zero_base_below_height=SQUAT_COMMAND_THRESHOLD,
        )

        self.commands.arm_sine.frequency_hz = self.arm_sine_frequency_hz
        self.commands.arm_sine.amplitude = self.arm_sine_amplitude

        # Policy input: 73 arm-stability dims + target body height + current base z.
        self.observations.policy.body_height_command = ObsTerm(
            func=mdp.generated_commands,
            params={"command_name": "body_height"},
            clip=(-10.0, 10.0),
            scale=1.0,
        )
        self.observations.policy.base_pos_z = ObsTerm(
            func=ppsuc_mdp.base_pos_z,
            params={"asset_cfg": SceneEntityCfg("robot")},
            clip=(-10.0, 10.0),
            scale=1.0,
        )
        if hasattr(self.observations, "critic"):
            self.observations.critic.body_height_command = ObsTerm(
                func=mdp.generated_commands,
                params={"command_name": "body_height"},
                clip=(-10.0, 10.0),
                scale=1.0,
            )
            self.observations.critic.base_pos_z = ObsTerm(
                func=ppsuc_mdp.base_pos_z,
                params={"asset_cfg": SceneEntityCfg("robot")},
                clip=(-10.0, 10.0),
                scale=1.0,
            )

        full_joint_names = self.joint_names + ARM_JOINTS
        self.observations.policy.joint_pos.params["asset_cfg"].joint_names = full_joint_names
        self.observations.policy.joint_vel.params["asset_cfg"].joint_names = full_joint_names
        self.observations.critic.joint_pos.params["asset_cfg"].joint_names = full_joint_names
        self.observations.critic.joint_vel.params["asset_cfg"].joint_names = full_joint_names

        self.rewards.track_body_height = RewTerm(
            func=ppsuc_mdp.track_body_height_exp,
            weight=22.0,
            params={"std": 0.090, "command_name": "body_height"},
        )
        self.rewards.body_height_abs_error = RewTerm(
            func=ppsuc_mdp.body_height_abs_error,
            weight=-18.0,
            params={"command_name": "body_height"},
        )
        self.rewards.stand_command_height_below = RewTerm(
            func=ppsuc_mdp.stand_command_height_below_l2,
            weight=-180.0,
            params={"command_name": "body_height", "stand_threshold": STAND_COMMAND_THRESHOLD},
        )
        self.rewards.stand_command_joint_deviation = RewTerm(
            func=ppsuc_mdp.stand_command_joint_deviation_l1,
            weight=-1.6,
            params={
                "height_command_name": "body_height",
                "stand_threshold": STAND_COMMAND_THRESHOLD,
                "asset_cfg": SceneEntityCfg("robot", joint_names=self.joint_names, preserve_order=True),
            },
        )
        self.rewards.low_height_or_zero_command_lin_vel = RewTerm(
            func=ppsuc_mdp.low_height_or_zero_command_lin_vel_xy_l2,
            weight=-12.0,
            params={
                "velocity_command_name": "base_velocity",
                "height_command_name": "body_height",
                "height_threshold": SQUAT_COMMAND_THRESHOLD,
                "command_threshold": 0.06,
            },
        )
        self.rewards.z_excess_penalty = RewTerm(
            func=ppsuc_mdp.z_excess_penalty,
            weight=-3.0,
            params={"penalty": 1.0, "threshold": 0.55},
        )

        if self.rewards.track_lin_vel_xy_exp is not None:
            self.rewards.track_lin_vel_xy_exp.weight = 8.0
        if self.rewards.track_ang_vel_z_exp is not None:
            self.rewards.track_ang_vel_z_exp.weight = 2.0
        if self.rewards.upward is not None:
            self.rewards.upward.weight = 6.0
        if self.rewards.flat_orientation_l2 is not None:
            self.rewards.flat_orientation_l2.weight = -2.5
        if self.rewards.lin_vel_z_l2 is not None:
            self.rewards.lin_vel_z_l2.weight = -1.2
        if self.rewards.ang_vel_xy_l2 is not None:
            self.rewards.ang_vel_xy_l2.weight = -0.12
        if self.rewards.action_rate_l2 is not None:
            self.rewards.action_rate_l2.weight = -0.014
        if self.rewards.stand_still is not None:
            self.rewards.stand_still.weight = -1.4
        if self.rewards.joint_pos_penalty is not None:
            self.rewards.joint_pos_penalty.weight = -0.65

        self.terminations.bad_orientation = DoneTerm(
            func=bad_orientation_persistent_after_grace,
            params={
                "asset_cfg": SceneEntityCfg("robot"),
                "limit_angle": 1.05,
                "grace_time_s": 1.5,
                "persistence_s": 0.45,
            },
        )
        self.terminations.base_height_below_minimum = DoneTerm(
            func=root_height_persistent_below_minimum_after_grace,
            params={
                "asset_cfg": SceneEntityCfg("robot"),
                "minimum_height": 0.20,
                "grace_time_s": 1.5,
                "persistence_s": 0.35,
            },
        )
        self.terminations.illegal_contact = None
        self.scene.robot.init_state.pos = (0.0, 0.0, STAND_HEIGHT)
        self.episode_length_s = 20.0


@configclass
class UnitreeB2WPiperCerebellumStage2FlatEnvCfg(UnitreeB2WPiperCerebellumFlatEnvCfg):
    """Stage 2: lower squat target and stronger arm disturbance."""

    squat_height: float = SQUAT_HEIGHT_STAGE2
    arm_sine_amplitude: float = 0.24
    arm_sine_frequency_hz: float = 0.50


@configclass
class UnitreeB2WPiperCerebellumStage3FlatEnvCfg(UnitreeB2WPiperCerebellumFlatEnvCfg):
    """Stage 3: reach the previous stable lower bound."""

    squat_height: float = SQUAT_HEIGHT_STAGE3
    arm_sine_amplitude: float = 0.26
    arm_sine_frequency_hz: float = 0.50


@configclass
class UnitreeB2WPiperCerebellumStage4FlatEnvCfg(UnitreeB2WPiperCerebellumFlatEnvCfg):
    """Stage 4: approach the final 40 percent squat."""

    squat_height: float = SQUAT_HEIGHT_STAGE4
    arm_sine_amplitude: float = 0.28
    arm_sine_frequency_hz: float = 0.52


@configclass
class UnitreeB2WPiperCerebellumFinalFlatEnvCfg(UnitreeB2WPiperCerebellumFlatEnvCfg):
    """Final stage: user's requested 40 percent squat height."""

    squat_height: float = SQUAT_HEIGHT_FINAL
    arm_sine_amplitude: float = 0.30
    arm_sine_frequency_hz: float = 0.55


@configclass
class UnitreeB2WPiperHeightStabilizerFlatEnvCfg(UnitreeB2WPiperCerebellumFlatEnvCfg):
    """Static height stabilizer used after switching away from teleop walking."""

    squat_height: float = SQUAT_HEIGHT_FINAL
    arm_sine_amplitude: float = 0.30
    arm_sine_frequency_hz: float = 0.50

    def __post_init__(self):
        super().__post_init__()

        # This policy is not a locomotion policy. It only holds zero base velocity
        # while tracking an operator-selected body height.
        self.commands.base_velocity.heading_command = False
        self.commands.base_velocity.rel_heading_envs = 0.0
        self.commands.base_velocity.rel_standing_envs = 1.0
        self.commands.base_velocity.resampling_time_range = (1.0e9, 1.0e9)
        self.commands.base_velocity.ranges.lin_vel_x = (0.0, 0.0)
        self.commands.base_velocity.ranges.lin_vel_y = (0.0, 0.0)
        self.commands.base_velocity.ranges.ang_vel_z = (0.0, 0.0)
        self.commands.base_velocity.ranges.heading = None

        self.commands.body_height.resampling_time_range = (1.2, 2.4)
        self.commands.body_height.ranges.height = (SQUAT_HEIGHT_FINAL, STAND_HEIGHT)
        self.commands.body_height.discrete_values = [
            STAND_HEIGHT,
            0.50,
            0.45,
            0.40,
            0.34,
            0.28,
            SQUAT_HEIGHT_FINAL,
        ]
        self.commands.body_height.zero_base_below_height = STAND_HEIGHT + 1.0e-3

        self.rewards.track_body_height.weight = 36.0
        self.rewards.track_body_height.params["std"] = 0.070
        self.rewards.body_height_abs_error.weight = -30.0
        self.rewards.stand_command_height_below.weight = -260.0
        self.rewards.stand_command_joint_deviation.weight = -0.8
        self.rewards.low_height_or_zero_command_lin_vel.weight = -28.0
        self.rewards.low_height_or_zero_command_lin_vel.params["height_threshold"] = STAND_HEIGHT + 1.0e-3
        self.rewards.z_excess_penalty.weight = -2.0

        self.rewards.root_xy_drift = RewTerm(
            func=ppsuc_mdp.root_xy_distance_l2_from_env_origin,
            weight=-10.0,
            params={"asset_cfg": SceneEntityCfg("robot")},
        )
        if self.rewards.track_lin_vel_xy_exp is not None:
            self.rewards.track_lin_vel_xy_exp.weight = 5.0
            self.rewards.track_lin_vel_xy_exp.params["std"] = 0.18
        if self.rewards.track_ang_vel_z_exp is not None:
            self.rewards.track_ang_vel_z_exp.weight = 3.0
            self.rewards.track_ang_vel_z_exp.params["std"] = 0.18
        if self.rewards.upward is not None:
            self.rewards.upward.weight = 8.0
        if self.rewards.flat_orientation_l2 is not None:
            self.rewards.flat_orientation_l2.weight = -4.0
        if self.rewards.lin_vel_z_l2 is not None:
            self.rewards.lin_vel_z_l2.weight = -2.0
        if self.rewards.ang_vel_xy_l2 is not None:
            self.rewards.ang_vel_xy_l2.weight = -0.35
        if self.rewards.action_rate_l2 is not None:
            self.rewards.action_rate_l2.weight = -0.020
        if self.rewards.stand_still is not None:
            self.rewards.stand_still.weight = -0.25
        if self.rewards.joint_pos_penalty is not None:
            self.rewards.joint_pos_penalty.weight = -0.35

        self.terminations.base_height_below_minimum.params["minimum_height"] = 0.18
        self.scene.robot.init_state.pos = (0.0, 0.0, STAND_HEIGHT)
        self.episode_length_s = 12.0

"""Arm-stability env cfg for b2w_piper.

The robot must stay perfectly still (feet planted, body stable) while its
manipulator arm swings on a sinusoidal trajectory. The arm is driven directly
by an ArmSineCommand (not part of the policy action space); the leg/wheel policy
(warm-started from the c2 locomotion policy) learns to reject the arm disturbance
without losing the stable standing behavior.

Key design:
  - velocity command = 0 (stay put), strongly tracked so the dog does not walk
  - observations extended to include the 8 arm joints (57 -> 73 dims) so the
    policy can anticipate the arm motion
  - arm driven by a per-step ArmSineCommand
  - strong balance/stability rewards preserved from c2
"""

import torch

from isaaclab.envs import mdp
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils import configclass

import ppsuc_rl_lab.train.locomotion.velocity.mdp as ppsuc_mdp
from ppsuc_rl_lab.train.locomotion.velocity.config.quadruped.unitree_b2w_piper.flat_slow_arm_fixed_env_cfg import (
    UnitreeB2WPiperFlatSlowArmFixedEnvCfg,
)

ARM_JOINTS = [
    "arm_joint1", "arm_joint2", "arm_joint3", "arm_joint4",
    "arm_joint5", "arm_joint6", "arm_joint7", "arm_joint8",
]


@configclass
class UnitreeB2WPiperArmStabilityEnvCfg(UnitreeB2WPiperFlatSlowArmFixedEnvCfg):
    def __post_init__(self):
        super().__post_init__()

        # ------------------------------------------------------------------
        # Movable velocity commands: W/A/S/D drive the dog, the arm swings
        # independently. The policy must stay stable while moving AND while
        # the arm disturbs it.
        # ------------------------------------------------------------------
        self.commands.base_velocity.heading_command = False
        self.commands.base_velocity.ranges.lin_vel_x = (-0.5, 0.5)
        self.commands.base_velocity.ranges.lin_vel_y = (0.0, 0.0)
        self.commands.base_velocity.ranges.ang_vel_z = (-0.5, 0.5)
        self.commands.base_velocity.ranges.heading = None
        self.commands.base_velocity.resampling_time_range = (4.0, 6.0)

        # ------------------------------------------------------------------
        # Arm sine command: drives the manipulator on a fixed sinusoidal swing.
        # ------------------------------------------------------------------
        self.commands.arm_sine = ppsuc_mdp.ArmSineCommandCfg(
            asset_name="robot",
            arm_joint_names=ARM_JOINTS,
            frequency_hz=0.5,
            amplitude=0.3,
            phase=[0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5],
        )

        # ------------------------------------------------------------------
        # Observations: include the arm joints so the policy can anticipate
        # the disturbance (joint_pos/joint_vel now cover 24 joints).
        # ------------------------------------------------------------------
        full_joint_names = self.joint_names + ARM_JOINTS  # 16 + 8 = 24
        self.observations.policy.joint_pos.params["asset_cfg"].joint_names = full_joint_names
        self.observations.policy.joint_vel.params["asset_cfg"].joint_names = full_joint_names
        self.observations.critic.joint_pos.params["asset_cfg"].joint_names = full_joint_names
        self.observations.critic.joint_vel.params["asset_cfg"].joint_names = full_joint_names

        # ------------------------------------------------------------------
        # Rewards: track velocity commands (move when commanded) + stay stable
        # under the arm disturbance.
        # ------------------------------------------------------------------
        # Track the commanded velocity normally (W/A/S/D move the dog).
        if self.rewards.track_lin_vel_xy_exp is not None:
            self.rewards.track_lin_vel_xy_exp.weight = 3.0
        if self.rewards.track_ang_vel_z_exp is not None:
            self.rewards.track_ang_vel_z_exp.weight = 1.5
        # Keep the body upright and at nominal height.
        if self.rewards.upward is not None:
            self.rewards.upward.weight = 5.0
        if self.rewards.flat_orientation_l2 is not None:
            self.rewards.flat_orientation_l2.weight = -2.0
        if self.rewards.lin_vel_z_l2 is not None:
            self.rewards.lin_vel_z_l2.weight = -1.0
        # Keep base_height_l2 so the dog does not crouch.
        if self.rewards.base_height_l2 is not None:
            self.rewards.base_height_l2.params["target_height"] = 0.72

        # ------------------------------------------------------------------
        # Terminations: fall detection.
        # ------------------------------------------------------------------
        from ppsuc_rl_lab.train.locomotion.velocity.config.quadruped.unitree_b2w_piper.impulse_recovery_c1_env_cfg import (
            bad_orientation_persistent_after_grace,
            root_height_persistent_below_minimum_after_grace,
        )
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
                "minimum_height": 0.38,
                "grace_time_s": 1.5,
                "persistence_s": 0.35,
            },
        )
        self.terminations.illegal_contact = None
        # is_terminated may be None after disable_zero_weight_rewards() in the parent.
        from isaaclab.managers import RewardTermCfg as RewTerm
        from isaaclab.envs import mdp as _mdp
        self.rewards.is_terminated = RewTerm(func=_mdp.is_terminated, weight=-20.0)

        # ------------------------------------------------------------------
        # Stable upright reset near the ground.
        # ------------------------------------------------------------------
        self.events.randomize_reset_base.params = {
            "pose_range": {
                "x": (-0.03, 0.03),
                "y": (-0.03, 0.03),
                "z": (0.0, 0.0),
                "roll": (0.0, 0.0),
                "pitch": (0.0, 0.0),
                "yaw": (-3.14, 3.14),
            },
            "velocity_range": {
                "x": (0.0, 0.0),
                "y": (0.0, 0.0),
                "z": (0.0, 0.0),
                "roll": (0.0, 0.0),
                "pitch": (0.0, 0.0),
                "yaw": (0.0, 0.0),
            },
        }
        self.scene.robot.init_state.pos = (0.0, 0.0, 0.72)


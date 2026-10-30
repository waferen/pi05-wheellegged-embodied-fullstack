"""Squat/stand training env cfg for b2w_piper.

The robot learns to track a dynamic body-height command (stand 0.72m <-> squat 0.45m)
on a flat ground. Inherits the flat slow arm-fixed configuration and adds:
  - a BodyHeightCommand that resamples a target height on a schedule
  - a base_pos_z observation so the policy can perceive its own body height
  - an exponential body-height tracking reward
"""

from isaaclab.envs import mdp
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass

import ppsuc_rl_lab.train.locomotion.velocity.mdp as ppsuc_mdp
from ppsuc_rl_lab.train.locomotion.velocity.config.quadruped.unitree_b2w_piper.flat_slow_arm_fixed_env_cfg import (
    UnitreeB2WPiperFlatSlowArmFixedEnvCfg,
)

# Stand height used by the base config (b2w rough sets target_height = 0.72).
STAND_HEIGHT = 0.72
SQUAT_HEIGHT = 0.45


@configclass
class UnitreeB2WPiperSquatFlatEnvCfg(UnitreeB2WPiperFlatSlowArmFixedEnvCfg):
    def __post_init__(self):
        super().__post_init__()

        # Keep the velocity command minimal: the robot should mostly stay put and squat.
        self.commands.base_velocity.ranges.lin_vel_x = (0.0, 0.0)
        self.commands.base_velocity.ranges.lin_vel_y = (0.0, 0.0)
        self.commands.base_velocity.ranges.ang_vel_z = (0.0, 0.0)
        self.commands.base_velocity.ranges.heading = (0.0, 0.0)
        self.commands.base_velocity.resampling_time_range = (10.0, 10.0)

        # Add the body-height command (squat <-> stand).
        self.commands.body_height = ppsuc_mdp.BodyHeightCommandCfg(
            resampling_time_range=(4.0, 6.0),
            debug_vis=False,
            ranges=ppsuc_mdp.BodyHeightCommandCfg.Ranges(height=(SQUAT_HEIGHT, STAND_HEIGHT)),
        )

        # Give the policy the target height command so it knows how low to squat.
        self.observations.policy.body_height_command = ObsTerm(
            func=mdp.generated_commands,
            params={"command_name": "body_height"},
            clip=(-10.0, 10.0),
            scale=1.0,
        )
        # Give the policy direct access to its own body height.
        self.observations.policy.base_pos_z = ObsTerm(
            func=ppsuc_mdp.base_pos_z,
            params={"asset_cfg": SceneEntityCfg("robot")},
            clip=(-10.0, 10.0),
            scale=1.0,
        )
        if hasattr(self.observations, "critic"):
            self.observations.critic.base_pos_z = ObsTerm(
                func=ppsuc_mdp.base_pos_z,
                params={"asset_cfg": SceneEntityCfg("robot")},
                clip=(-10.0, 10.0),
                scale=1.0,
            )

        # Height-tracking reward drives the squat/stand behavior.
        self.rewards.track_body_height = RewTerm(
            func=ppsuc_mdp.track_body_height_exp,
            weight=12.0,
            params={"std": 0.04, "command_name": "body_height"},
        )
        # Anti-jump: penalize excessive vertical velocity so the robot squats
        # smoothly instead of hopping (does not affect normal standing).
        self.rewards.z_excess_penalty = RewTerm(
            func=ppsuc_mdp.z_excess_penalty,
            weight=-8.0,
            params={"penalty": 1.0, "threshold": 0.6},
        )
        # Keep the base-height regularizer as a soft prior centered at the stand height.
        # (It may be None after disable_zero_weight_rewards() runs in the parent.)
        if self.rewards.base_height_l2 is not None:
            self.rewards.base_height_l2.params["target_height"] = STAND_HEIGHT

        # Strong upright/orientation regularization so squatting does not turn into falling.
        if self.rewards.upward is not None:
            self.rewards.upward.weight = 5.0
        if self.rewards.flat_orientation_l2 is not None:
            self.rewards.flat_orientation_l2.weight = -1.0

        # Stay-put: velocity command is zero, so strongly tracking lin_vel_xy
        # keeps the robot stationary while it squats/stands (no wandering).
        if self.rewards.track_lin_vel_xy_exp is not None:
            self.rewards.track_lin_vel_xy_exp.weight = 8.0
        if self.rewards.track_ang_vel_z_exp is not None:
            self.rewards.track_ang_vel_z_exp.weight = 3.0

        # Longer episode so the robot experiences several squat/stand transitions.
        if hasattr(self, "episode_length_s"):
            self.episode_length_s = 20.0


        # Spawn near the ground so reset does not drop the robot from mid-air.
        # B2W wheel-ground base height is ~0.72 (base_height_l2 target).
        self.scene.robot.init_state.pos = (0.0, 0.0, STAND_HEIGHT)

        # Stable ground reset: spawn upright, near-ground, no random velocity.
        # (Base flat/rough configs inject up to 0.2m height + random roll/pitch, which
        #  makes the robot drop from mid-air and land unstably.)
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

from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass

import ppsuc_rl_lab.train.locomotion.velocity.mdp as mdp
from ppsuc_rl_lab.train.locomotion.velocity.config.quadruped.unitree_b2w_piper.collision_resilience_d2_env_cfg import (
    UnitreeB2WPiperFlatCollisionResilienceD2EnvCfg,
)


@configclass
class UnitreeB2WPiperFlatCollisionResilienceD3BEnvCfg(UnitreeB2WPiperFlatCollisionResilienceD2EnvCfg):
    """D3B: stabilize in place, then drive forward into the D2 obstacle corridor.

    The first 3 seconds command zero velocity for in-place stabilization. After that a fresh
    forward-only command is sampled so the robot learns to move straight through natural obstacle
    contacts without artificial pushes or yaw commands.
    """

    def __post_init__(self):
        super().__post_init__()

        self.events.randomize_apply_external_force_torque = None
        self.events.randomize_push_robot = None

        self.commands.base_velocity.class_type = mdp.RecoveryThenForwardVelocityCommand
        self.commands.base_velocity.recovery_time_s = 3.0
        self.commands.base_velocity.resampling_time_range = (3.0, 5.0)
        self.commands.base_velocity.ranges.lin_vel_x = (0.35, 0.95)
        self.commands.base_velocity.ranges.lin_vel_y = (0.0, 0.0)
        self.commands.base_velocity.ranges.ang_vel_z = (0.0, 0.0)

        self.rewards.track_lin_vel_xy_exp.weight = 5.5
        self.rewards.track_ang_vel_z_exp.weight = 1.0
        self.rewards.upward.weight = 4.5
        self.rewards.stand_still.weight = -2.0
        self.rewards.undesired_contacts.weight = -0.7
        self.rewards.contact_forces.weight = -1.5e-4
        self.rewards.is_terminated.weight = -35.0

        self.rewards.recovery_upright = RewTerm(
            func=mdp.recovery_upright_reward,
            weight=3.0,
            params={"recovery_time_s": 3.0, "asset_cfg": SceneEntityCfg("robot")},
        )
        self.rewards.recovery_still = RewTerm(
            func=mdp.recovery_still_reward,
            weight=2.0,
            params={
                "recovery_time_s": 3.0,
                "lin_vel_std": 0.25,
                "ang_vel_std": 0.8,
                "asset_cfg": SceneEntityCfg("robot"),
            },
        )
        self.rewards.recovery_unstable_after_deadline = RewTerm(
            func=mdp.recovery_unstable_after_deadline,
            weight=-8.0,
            params={
                "recovery_time_s": 3.0,
                "max_tilt_angle": 0.35,
                "min_height": 0.45,
                "max_lin_vel_xy": 0.25,
                "max_ang_vel_xy": 0.8,
                "asset_cfg": SceneEntityCfg("robot"),
            },
        )

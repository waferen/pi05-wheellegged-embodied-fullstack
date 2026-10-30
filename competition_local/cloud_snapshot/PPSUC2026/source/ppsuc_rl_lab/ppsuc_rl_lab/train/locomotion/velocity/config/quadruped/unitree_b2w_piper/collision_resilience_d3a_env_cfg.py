from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils import configclass

import ppsuc_rl_lab.train.locomotion.velocity.mdp as mdp
from ppsuc_rl_lab.train.locomotion.velocity.config.quadruped.unitree_b2w_piper.collision_resilience_d2_env_cfg import (
    UnitreeB2WPiperFlatCollisionResilienceD2EnvCfg,
)


def base_xy_out_of_recovery_radius(env, max_distance: float, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")):
    asset = env.scene[asset_cfg.name]
    distance = mdp.root_xy_distance_from_env_origin(env, asset_cfg)
    return distance > max_distance


@configclass
class UnitreeB2WPiperFlatCollisionResilienceD3AEnvCfg(UnitreeB2WPiperFlatCollisionResilienceD2EnvCfg):
    """D3A: reset-to-stable posture recovery in place.

    Keeps the D2 corridor scene, removes artificial pushes, and commands zero velocity. The policy
    is trained to stabilize quickly after reset and remain still/upright before learning to move
    into obstacles in D3B.
    """

    def __post_init__(self):
        super().__post_init__()

        self.events.randomize_apply_external_force_torque = None
        self.events.randomize_push_robot = None
        self.events.randomize_reset_base.params["pose_range"].update({"x": (-0.05, 0.05), "y": (-0.05, 0.05)})

        self.commands.base_velocity.resampling_time_range = (3.0, 5.0)
        self.commands.base_velocity.ranges.lin_vel_x = (0.0, 0.0)
        self.commands.base_velocity.ranges.lin_vel_y = (0.0, 0.0)
        self.commands.base_velocity.ranges.ang_vel_z = (0.0, 0.0)

        self.rewards.track_lin_vel_xy_exp.weight = 2.0
        self.rewards.track_ang_vel_z_exp.weight = 1.0
        self.rewards.upward.weight = 5.0
        self.rewards.stand_still.weight = -1.5
        self.rewards.undesired_contacts.weight = -0.5
        self.rewards.contact_forces.weight = -1.0e-4
        self.rewards.is_terminated.weight = -25.0
        self.rewards.recovery_xy_radius = RewTerm(
            func=mdp.root_xy_distance_l2_from_env_origin,
            weight=-10.0,
            params={"asset_cfg": SceneEntityCfg("robot")},
        )

        self.terminations.base_xy_out_of_recovery_radius = DoneTerm(
            func=base_xy_out_of_recovery_radius,
            params={"max_distance": 2.0, "asset_cfg": SceneEntityCfg("robot")},
        )

        self.rewards.recovery_upright = RewTerm(
            func=mdp.recovery_upright_reward,
            weight=4.0,
            params={"recovery_time_s": 3.0, "asset_cfg": SceneEntityCfg("robot")},
        )
        self.rewards.recovery_still = RewTerm(
            func=mdp.recovery_still_reward,
            weight=6.0,
            params={
                "recovery_time_s": 3.0,
                "lin_vel_std": 0.20,
                "ang_vel_std": 0.6,
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

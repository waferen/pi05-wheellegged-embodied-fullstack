from isaaclab.utils import configclass

import ppsuc_rl_lab.train.locomotion.velocity.mdp as mdp

from ppsuc_rl_lab.train.locomotion.velocity.config.quadruped.unitree_b2w_piper.collision_resilience_d2_env_cfg import (
    UnitreeB2WPiperFlatCollisionResilienceD2EnvCfg,
)


@configclass
class UnitreeB2WPiperFlatCollisionResilienceD3EnvCfg(UnitreeB2WPiperFlatCollisionResilienceD2EnvCfg):
    """D3: forward collision recovery without artificial pushes.

    Keeps D2's narrow corridor and obstacle contacts, but removes manually injected external
    disturbances. The policy is rewarded for stabilizing quickly, moving forward through the
    obstacle field, and recovering after natural contacts without falling or stopping.
    """

    def __post_init__(self):
        super().__post_init__()

        # Keep D2's corridor/obstacle difficulty, but remove artificial force/torque and push events.
        self.events.randomize_apply_external_force_torque = None
        self.events.randomize_push_robot = None

        self.commands.base_velocity.class_type = mdp.RecoveryThenForwardVelocityCommand
        self.commands.base_velocity.recovery_time_s = 1.0

        # Focus this stage on straight-line recovery through obstacles. Yaw command stays zero so the policy
        # learns to stabilize in place first, then continue forward without turning.
        self.commands.base_velocity.resampling_time_range = (3.0, 5.0)
        self.commands.base_velocity.ranges.lin_vel_x = (0.35, 0.95)
        self.commands.base_velocity.ranges.lin_vel_y = (0.0, 0.0)
        self.commands.base_velocity.ranges.ang_vel_z = (0.0, 0.0)

        # Encourage fast stabilization and forward progress. Contacts are penalized, but not so strongly
        # that the policy learns to stop before the corridor.
        self.rewards.track_lin_vel_xy_exp.weight = 5.5
        self.rewards.track_ang_vel_z_exp.weight = 1.2
        self.rewards.upward.weight = 4.2
        self.rewards.stand_still.weight = -3.0
        self.rewards.undesired_contacts.weight = -0.7
        self.rewards.contact_forces.weight = -1.5e-4
        self.rewards.is_terminated.weight = -35.0

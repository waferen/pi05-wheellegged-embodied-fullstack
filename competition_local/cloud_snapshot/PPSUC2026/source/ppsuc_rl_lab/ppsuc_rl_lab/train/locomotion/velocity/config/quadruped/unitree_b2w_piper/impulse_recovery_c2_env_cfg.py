from isaaclab.utils import configclass

from ppsuc_rl_lab.train.locomotion.velocity.config.quadruped.unitree_b2w_piper.impulse_recovery_c1_env_cfg import (
    UnitreeB2WPiperFlatImpulseRecoveryC1EnvCfg,
)


@configclass
class UnitreeB2WPiperFlatImpulseRecoveryC2EnvCfg(UnitreeB2WPiperFlatImpulseRecoveryC1EnvCfg):
    """C2: stronger flat-ground interval impulse recovery.

    This keeps C1's stable flat reset and no continuous external wrench, but increases push strength and
    frequency. It should be resumed from the latest C1 checkpoint, not from the original model_41600.pt.
    """

    def __post_init__(self):
        super().__post_init__()

        # Slightly wider locomotion range so the policy stays useful for teleop after harder impacts.
        self.commands.base_velocity.ranges.lin_vel_x = (0.30, 1.25)
        self.commands.base_velocity.ranges.lin_vel_y = (0.0, 0.0)
        self.commands.base_velocity.ranges.ang_vel_z = (-0.8, 0.8)

        # Stronger and more frequent interval pushes than C1. Still no z/roll/pitch injection.
        self.events.randomize_push_robot.interval_range_s = (0.8, 2.2)
        self.events.randomize_push_robot.params["velocity_range"] = {
            "x": (-1.90, 1.90),
            "y": (-1.90, 1.90),
            "z": (0.0, 0.0),
            "roll": (0.0, 0.0),
            "pitch": (0.0, 0.0),
            "yaw": (-0.90, 0.90),
        }

        # Keep forward tracking important while giving extra tolerance to stronger transient impacts.
        self.rewards.track_lin_vel_xy_exp.weight = 4.2
        self.rewards.track_ang_vel_z_exp.weight = 1.6
        self.rewards.upward.weight = 4.4
        self.rewards.ang_vel_xy_l2.weight = -0.09
        self.rewards.action_rate_l2.weight = -0.014
        self.rewards.contact_forces.weight = -1.2e-4
        self.rewards.undesired_contacts.weight = -1.3

from isaaclab.utils import configclass

from ppsuc_rl_lab.train.locomotion.velocity.config.quadruped.unitree_b2w_piper.flat_env_cfg import (
    UnitreeB2WPiperFlatEnvCfg,
)


@configclass
class UnitreeB2WPiperFlatSlowArmFixedEnvCfg(UnitreeB2WPiperFlatEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.commands.base_velocity.heading_command = False
        self.commands.base_velocity.rel_heading_envs = 0.0
        self.commands.base_velocity.rel_standing_envs = 0.0
        self.commands.base_velocity.resampling_time_range = (4.0, 6.0)
        self.commands.base_velocity.ranges.lin_vel_x = (-0.45, 0.45)
        self.commands.base_velocity.ranges.lin_vel_y = (0.0, 0.0)
        self.commands.base_velocity.ranges.ang_vel_z = (-0.60, 0.60)
        self.commands.base_velocity.ranges.heading = None
        self.events.randomize_push_robot = None
        self.events.randomize_apply_external_force_torque = None
        self.disable_zero_weight_rewards()

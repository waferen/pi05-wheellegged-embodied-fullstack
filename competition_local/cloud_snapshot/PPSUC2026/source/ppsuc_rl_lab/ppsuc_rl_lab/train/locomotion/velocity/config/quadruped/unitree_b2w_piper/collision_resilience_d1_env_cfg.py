import torch

from isaaclab.envs import mdp
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils import configclass

from ppsuc_rl_lab.train.locomotion.velocity.config.quadruped.unitree_b2w_piper.flat_env_cfg import (
    UnitreeB2WPiperFlatEnvCfg,
)


def _past_grace_time(env, grace_time_s: float, key: str):
    current_len = env.episode_length_buf.detach().clone()
    state = getattr(_past_grace_time, "_state", {})
    ages, prev_len = state.get(key, (None, None))
    if ages is None or ages.shape != current_len.shape or ages.device != current_len.device:
        ages = torch.zeros(current_len.shape, dtype=torch.float32, device=current_len.device)
        state[key] = (ages, current_len)
        _past_grace_time._state = state
        return torch.zeros(current_len.shape, dtype=torch.bool, device=current_len.device)

    reset = current_len < prev_len
    advanced = current_len > prev_len
    ages[reset] = 0.0
    ages[advanced] += (current_len[advanced] - prev_len[advanced]).to(dtype=torch.float32) * env.step_dt
    state[key] = (ages, current_len)
    _past_grace_time._state = state
    return ages > grace_time_s


def bad_orientation_after_grace(env, limit_angle: float, grace_time_s: float, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")):
    asset = env.scene[asset_cfg.name]
    gravity_z = torch.clamp(-asset.data.projected_gravity_b[:, 2], -1.0, 1.0)
    fallen = torch.acos(gravity_z).abs() > limit_angle
    return fallen & _past_grace_time(env, grace_time_s, "bad_orientation")


def root_height_below_minimum_after_grace(
    env, minimum_height: float, grace_time_s: float, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
):
    asset = env.scene[asset_cfg.name]
    fallen = asset.data.root_pos_w[:, 2] < minimum_height
    return fallen & _past_grace_time(env, grace_time_s, "base_height_below_minimum")


@configclass
class UnitreeB2WPiperFlatCollisionResilienceD1EnvCfg(UnitreeB2WPiperFlatEnvCfg):
    """D1: flat-ground disturbance recovery for B2W+Piper.

    This stage trains recovery from pushes and external wrenches without adding walls or obstacles yet.
    The arm remains outside the action space, so the low-level policy still controls legs and wheels only.
    """

    def __post_init__(self):
        super().__post_init__()

        # Commands: keep D1 focused on stable forward/turning recovery.
        self.commands.base_velocity.debug_vis = False
        self.commands.base_velocity.heading_command = False
        self.commands.base_velocity.rel_heading_envs = 0.0
        self.commands.base_velocity.rel_standing_envs = 0.0
        self.commands.base_velocity.resampling_time_range = (3.0, 5.0)
        self.commands.base_velocity.ranges.lin_vel_x = (0.3, 1.0)
        self.commands.base_velocity.ranges.lin_vel_y = (0.0, 0.0)
        self.commands.base_velocity.ranges.ang_vel_z = (-0.8, 0.8)
        self.commands.base_velocity.ranges.heading = None

        # Reset: slight pose/velocity errors, not full random roll/pitch. Full random orientation is too hard for D1.
        self.events.randomize_reset_base.params = {
            "pose_range": {
                "x": (-0.25, 0.25),
                "y": (-0.25, 0.25),
                "z": (0.0, 0.08),
                "roll": (-0.18, 0.18),
                "pitch": (-0.18, 0.18),
                "yaw": (-3.14, 3.14),
            },
            "velocity_range": {
                "x": (-0.45, 0.45),
                "y": (-0.45, 0.45),
                "z": (-0.20, 0.20),
                "roll": (-0.35, 0.35),
                "pitch": (-0.35, 0.35),
                "yaw": (-0.45, 0.45),
            },
        }

        # Continuous reset wrench: stronger than the base B2 config, but still D1-safe.
        self.events.randomize_apply_external_force_torque.params["asset_cfg"].body_names = [self.base_link_name]
        self.events.randomize_apply_external_force_torque.params["force_range"] = (-120.0, 120.0)
        self.events.randomize_apply_external_force_torque.params["torque_range"] = (-35.0, 35.0)

        # Interval impulses: model bumps and sudden lateral disturbances.
        self.events.randomize_push_robot.interval_range_s = (2.0, 4.0)
        self.events.randomize_push_robot.params["velocity_range"] = {
            "x": (-1.0, 1.0),
            "y": (-1.0, 1.0),
            "z": (0.0, 0.0),
            "roll": (-0.25, 0.25),
            "pitch": (-0.25, 0.25),
            "yaw": (-0.8, 0.8),
        }

        # Keep contact terms active. For D1 this means: non-foot contacts are bad, foot impact forces are moderated.
        self.rewards.undesired_contacts.weight = -1.5
        self.rewards.undesired_contacts.params["sensor_cfg"].body_names = [f"^(?!.*{self.foot_link_name}).*"]
        self.rewards.undesired_contacts.params["threshold"] = 1.0
        self.rewards.contact_forces.weight = -2.0e-4
        self.rewards.contact_forces.params["sensor_cfg"].body_names = [self.foot_link_name]
        self.rewards.contact_forces.params["threshold"] = 100.0
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


        # Recovery objective: preserve tracking and uprightness while being disturbed.
        self.rewards.track_lin_vel_xy_exp.weight = 3.5
        self.rewards.track_ang_vel_z_exp.weight = 1.8
        self.rewards.upward.weight = 3.5
        self.rewards.ang_vel_xy_l2.weight = -0.08
        self.rewards.lin_vel_z_l2.weight = -2.0
        self.rewards.action_rate_l2.weight = -0.015
        self.rewards.joint_torques_l2.weight = -1.0e-5
        self.rewards.joint_acc_l2.weight = -1.0e-7
        self.rewards.joint_power.weight = -1.0e-5
        self.rewards.joint_pos_limits.weight = -5.0

        # D1 failure means the robot fell, not merely that it brushed something.
        # Keep body contacts as penalties, but reset immediately on severe tilt or base collapse.
        self.terminations.illegal_contact = None
        self.terminations.bad_orientation = DoneTerm(
            func=bad_orientation_after_grace,
            params={"asset_cfg": SceneEntityCfg("robot"), "limit_angle": 1.05, "grace_time_s": 1.0},
        )
        self.terminations.base_height_below_minimum = DoneTerm(
            func=root_height_below_minimum_after_grace,
            params={"asset_cfg": SceneEntityCfg("robot"), "minimum_height": 0.38, "grace_time_s": 1.0},
        )
        self.rewards.is_terminated.weight = -20.0

        # Remove inherited zero-weight terms before managers resolve regex body names.
        self.disable_zero_weight_rewards()

import torch

from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils import configclass

from ppsuc_rl_lab.train.locomotion.velocity.config.quadruped.unitree_b2w_piper.flat_env_cfg import (
    UnitreeB2WPiperFlatEnvCfg,
)


def _persistent_condition_after_grace(env, condition: torch.Tensor, grace_time_s: float, persistence_s: float, key: str):
    current_len = env.episode_length_buf.detach().clone()
    state = getattr(_persistent_condition_after_grace, "_state", {})
    ages, durations, prev_len = state.get(key, (None, None, None))
    if ages is None or ages.shape != current_len.shape or ages.device != current_len.device:
        ages = torch.zeros(current_len.shape, dtype=torch.float32, device=current_len.device)
        durations = torch.zeros_like(ages)
        state[key] = (ages, durations, current_len)
        _persistent_condition_after_grace._state = state
        return torch.zeros(current_len.shape, dtype=torch.bool, device=current_len.device)

    reset = current_len < prev_len
    delta_steps = torch.clamp(current_len - prev_len, min=0).to(dtype=torch.float32)
    delta_t = delta_steps * env.step_dt

    ages[reset] = 0.0
    durations[reset] = 0.0
    ages += delta_t
    durations = torch.where(condition, durations + delta_t, torch.zeros_like(durations))

    state[key] = (ages, durations, current_len)
    _persistent_condition_after_grace._state = state
    return (ages > grace_time_s) & (durations > persistence_s)


def bad_orientation_persistent_after_grace(
    env,
    limit_angle: float,
    grace_time_s: float,
    persistence_s: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
):
    asset = env.scene[asset_cfg.name]
    gravity_z = torch.clamp(-asset.data.projected_gravity_b[:, 2], -1.0, 1.0)
    fallen = torch.acos(gravity_z).abs() > limit_angle
    return _persistent_condition_after_grace(env, fallen, grace_time_s, persistence_s, "impulse_bad_orientation")


def root_height_persistent_below_minimum_after_grace(
    env,
    minimum_height: float,
    grace_time_s: float,
    persistence_s: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
):
    asset = env.scene[asset_cfg.name]
    fallen = asset.data.root_pos_w[:, 2] < minimum_height
    return _persistent_condition_after_grace(env, fallen, grace_time_s, persistence_s, "impulse_base_height")


@configclass
class UnitreeB2WPiperFlatImpulseRecoveryC1EnvCfg(UnitreeB2WPiperFlatEnvCfg):
    """C1: flat locomotion with interval impulse recovery.

    The robot resets upright on the plane, walks normally, then receives short velocity pushes during
    the episode. This preserves the strong locomotion policy from model_41600.pt while adding
    disturbance recovery.
    """

    def __post_init__(self):
        super().__post_init__()

        self.commands.base_velocity.debug_vis = False
        self.commands.base_velocity.heading_command = False
        self.commands.base_velocity.rel_heading_envs = 0.0
        self.commands.base_velocity.rel_standing_envs = 0.0
        self.commands.base_velocity.resampling_time_range = (3.0, 5.0)
        self.commands.base_velocity.ranges.lin_vel_x = (0.35, 1.15)
        self.commands.base_velocity.ranges.lin_vel_y = (0.0, 0.0)
        self.commands.base_velocity.ranges.ang_vel_z = (-0.7, 0.7)
        self.commands.base_velocity.ranges.heading = None

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

        self.events.randomize_apply_external_force_torque = None
        self.events.randomize_push_robot.interval_range_s = (1.0, 2.6)
        self.events.randomize_push_robot.params["velocity_range"] = {
            "x": (-1.45, 1.45),
            "y": (-1.45, 1.45),
            "z": (0.0, 0.0),
            "roll": (0.0, 0.0),
            "pitch": (0.0, 0.0),
            "yaw": (-0.65, 0.65),
        }

        self.rewards.track_lin_vel_xy_exp.weight = 4.0
        self.rewards.track_ang_vel_z_exp.weight = 1.6
        self.rewards.upward.weight = 4.0
        self.rewards.lin_vel_z_l2.weight = -2.0
        self.rewards.ang_vel_xy_l2.weight = -0.07
        self.rewards.action_rate_l2.weight = -0.012
        self.rewards.joint_torques_l2.weight = -1.0e-5
        self.rewards.joint_acc_l2.weight = -1.0e-7
        self.rewards.joint_power.weight = -1.0e-5
        self.rewards.joint_pos_limits.weight = -5.0
        self.rewards.stand_still.weight = -1.0
        self.rewards.undesired_contacts.weight = -1.5
        self.rewards.undesired_contacts.params["sensor_cfg"].body_names = [f"^(?!.*{self.foot_link_name}).*"]
        self.rewards.undesired_contacts.params["threshold"] = 1.0
        self.rewards.contact_forces.weight = -1.5e-4
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

        self.terminations.illegal_contact = None
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
        self.rewards.is_terminated.weight = -20.0

        self.disable_zero_weight_rewards()

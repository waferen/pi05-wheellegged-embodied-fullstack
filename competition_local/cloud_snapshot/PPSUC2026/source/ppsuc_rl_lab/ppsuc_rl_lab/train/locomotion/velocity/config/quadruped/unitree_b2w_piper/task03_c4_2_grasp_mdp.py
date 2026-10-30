"""MDP terms for the from-scratch C4_2 grasping task.

The terms deliberately use the object height relative to its reset height.  The
asset is already above the ground on the table, so an absolute-height reward
would incorrectly mark every reset as a successful lift.
"""

from __future__ import annotations

import torch

from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import FrameTransformer
from isaaclab.utils.math import quat_apply, subtract_frame_transforms

# The two gripper fingers.  arm_joint7/8 are the prismatic finger joints.
FINGER_LINKS = ["arm_link7", "arm_link8"]


def _ee_world_position(env, ee_frame_cfg: SceneEntityCfg) -> torch.Tensor:
    frame: FrameTransformer = env.scene[ee_frame_cfg.name]
    return frame.data.target_pos_w[..., 0, :]


def _fingertip_world_position(env, robot_cfg: SceneEntityCfg) -> torch.Tensor:
    """(num_envs, 2, 3) world positions of the two fingertips."""
    robot: Articulation = env.scene[robot_cfg.name]
    body_indices = [robot.body_names.index(name) for name in FINGER_LINKS]
    return robot.data.body_pos_w[:, body_indices, :]


def _fingertip_contact_force(env, robot_cfg: SceneEntityCfg) -> torch.Tensor:
    """(num_envs, 2) geometric contact proxy: distance of each fingertip to the object.

    A real ContactSensor is deferred (b2w_piper.usda bodies lack the contact
    reporter API).  Until it is added, "contact" is approximated as the inverse
    distance from each fingertip to the object, so the value is large when the
    finger is touching the object and ~0 far away.
    """
    obj: RigidObject = env.scene["object"]
    finger_w = _fingertip_world_position(env, robot_cfg)  # (N, 2, 3)
    d = torch.linalg.norm(finger_w - obj.data.root_pos_w.unsqueeze(1), dim=-1)  # (N, 2)
    return torch.exp(-d / 0.03)  # ~1 when touching, ~0 beyond ~10cm


def ee_position_in_robot_root_frame(
    env,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
) -> torch.Tensor:
    robot: Articulation = env.scene[robot_cfg.name]
    ee_w = _ee_world_position(env, ee_frame_cfg)
    ee_b, _ = subtract_frame_transforms(robot.data.root_pos_w, robot.data.root_quat_w, ee_w)
    return ee_b


def object_height_delta(
    env,
    initial_height: float,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    obj: RigidObject = env.scene[object_cfg.name]
    # env origins are at z=0 for the default plane, but subtracting them keeps
    # this correct when a terrain importer or a non-zero origin is used later.
    local_z = obj.data.root_pos_w[:, 2] - env.scene.env_origins[:, 2]
    return local_z - initial_height


def object_height_delta_obs(
    env,
    initial_height: float,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    return object_height_delta(env, initial_height, object_cfg).unsqueeze(-1)


def reaching_object(
    env,
    std: float,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
) -> torch.Tensor:
    obj: RigidObject = env.scene[object_cfg.name]
    distance = torch.linalg.norm(obj.data.root_pos_w - _ee_world_position(env, ee_frame_cfg), dim=1)
    return torch.exp(-distance / std)


def lift_progress(
    env,
    initial_height: float,
    scale_height: float = 0.10,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    delta = object_height_delta(env, initial_height, object_cfg)
    return torch.clamp(delta / scale_height, 0.0, 1.0)


def object_lifted(
    env,
    initial_height: float,
    minimal_lift: float = 0.035,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    delta = object_height_delta(env, initial_height, object_cfg)
    return (delta > minimal_lift).to(torch.float32)


def lift_hold(
    env,
    initial_height: float,
    minimal_lift: float = 0.035,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    obj: RigidObject = env.scene[object_cfg.name]
    delta = object_height_delta(env, initial_height, object_cfg)
    lifted = delta > minimal_lift
    speed_penalty = torch.linalg.norm(obj.data.root_lin_vel_w, dim=1)
    return lifted.to(torch.float32) * torch.exp(-speed_penalty)


# -----------------------------------------------------------------------------
# Extra observations
# -----------------------------------------------------------------------------


def gripper_opening(
    env,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Gripper opening amount: positive finger extension minus negative one.

    arm_joint7 range 0..0.035 (opens +), arm_joint8 range -0.035..0 (opens -).
    Opening = joint_pos[7] + |joint_pos[8]|, i.e. 0 when fully closed.
    """
    robot: Articulation = env.scene[robot_cfg.name]
    j7 = robot.data.joint_pos[:, robot.joint_names.index("arm_joint7")]
    j8 = robot.data.joint_pos[:, robot.joint_names.index("arm_joint8")]
    return (j7 - j8).unsqueeze(-1)  # = j7 + |j8| since j8 <= 0


def object_relative_to_ee(
    env,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
) -> torch.Tensor:
    """C4_2 position in the gripper_base frame (target tracking frame)."""
    ee_w = _ee_world_position(env, ee_frame_cfg)
    # use the frame's quaternion for a proper relative transform
    frame: FrameTransformer = env.scene[ee_frame_cfg.name]
    ee_quat = frame.data.target_quat_w[..., 0, :]
    rel_pos, _ = subtract_frame_transforms(ee_w, ee_quat, env.scene[object_cfg.name].data.root_pos_w)
    return rel_pos


def fingertip_object_distance(
    env,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Distance from the NEAREST fingertip to the object center."""
    obj: RigidObject = env.scene[object_cfg.name]
    finger_w = _fingertip_world_position(env, robot_cfg)  # (N, 2, 3)
    d = torch.linalg.norm(finger_w - obj.data.root_pos_w.unsqueeze(1), dim=-1)  # (N, 2)
    return d.min(dim=-1).values.unsqueeze(-1)  # (N, 1)


def fingertip_contact_obs(
    env,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Normal force on each fingertip, capped at a reasonable scale."""
    f = _fingertip_contact_force(env, robot_cfg)
    return torch.clamp(f, 0.0, 10.0)


# -----------------------------------------------------------------------------
# Extra rewards
# -----------------------------------------------------------------------------


def gripper_alignment(
    env,
    std: float = 0.2,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
) -> torch.Tensor:
    """Reward for the gripper pointing toward the object.

    Uses the projection of the EE->object direction onto the EE z-axis (the
    grasping direction).  Perfect when the object lies along the EE z-axis.
    """
    obj: RigidObject = env.scene[object_cfg.name]
    frame: FrameTransformer = env.scene[ee_frame_cfg.name]
    ee_w = frame.data.target_pos_w[..., 0, :]
    ee_quat = frame.data.target_quat_w[..., 0, :]  # wxyz, world frame
    # gripper grasping direction is the EE frame's z-axis rotated into the world
    z_local = torch.zeros(ee_quat.shape[0], 3, device=env.device)
    z_local[:, 2] = 1.0
    ee_z_world = quat_apply(ee_quat, z_local)
    to_obj = obj.data.root_pos_w - ee_w
    cos_angle = (to_obj * ee_z_world).sum(dim=-1) / (torch.linalg.norm(to_obj, dim=-1) + 1e-6)
    return torch.clamp(cos_angle, 0.0, 1.0)


def fingertips_around_object(
    env,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Reward when the two fingertips are on opposite sides of the object.

    The object is between the fingers when the dot of (f1-obj) and (f2-obj) is
    negative, i.e. the two fingertips are separated by the object.
    """
    obj: RigidObject = env.scene[object_cfg.name]
    finger_w = _fingertip_world_position(env, robot_cfg)  # (N, 2, 3)
    f1 = finger_w[:, 0, :] - obj.data.root_pos_w
    f2 = finger_w[:, 1, :] - obj.data.root_pos_w
    dot = (f1 * f2).sum(dim=-1)
    return (dot < 0.0).to(torch.float32)


def gripper_closed_with_contact(
    env,
    contact_threshold: float = 0.5,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Reward when the gripper is closed AND both fingertips contact the object."""
    opening = gripper_opening(env, robot_cfg).squeeze(-1)
    closed = opening < 0.02
    force = _fingertip_contact_force(env, robot_cfg)
    both_contact = (force.min(dim=-1).values > contact_threshold).to(torch.float32)
    return closed.to(torch.float32) * both_contact


def object_dropped(
    env,
    initial_height: float,
    tolerance: float = 0.12,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    return object_height_delta(env, initial_height, object_cfg) < -tolerance


def robot_fallen(
    env,
    minimum_height: float = 0.30,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    robot: Articulation = env.scene[robot_cfg.name]
    return robot.data.root_pos_w[:, 2] < minimum_height

"""B2W-Piper + C4_2 grasping environment, laid out for from-scratch training.

Unlike the full task03 stage (robot ~1.25 m from the table), this environment
places the robot next to the table so that, in the default standing pose, the
gripper fingertips are just above the C4_2 object.  A random policy only needs
to learn "reach down, close, lift" instead of first walking to the table.
"""

from __future__ import annotations

import os

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg, RigidObjectCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.envs import mdp as base_mdp
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import FrameTransformerCfg
from isaaclab.sim.spawners.from_files.from_files_cfg import GroundPlaneCfg, UsdFileCfg
from isaaclab.utils import configclass
from isaaclab_tasks.manager_based.manipulation.lift import mdp as lift_mdp

from ppsuc_rl_lab.assets import PPSUC_ASSETS_MODEL_DIR
from ppsuc_rl_lab.assets.robots import UNITREE_B2W_PIPER_CFG

from . import task03_c4_2_grasp_mdp as grasp_mdp


B2W_PIPER_USD = os.path.join(PPSUC_ASSETS_MODEL_DIR, "robot/b2w/b2w_piper.usda")
DANNY_USD = os.path.join(
    PPSUC_ASSETS_MODEL_DIR,
    "scene/Last02/Collected_Bomb_Disposal_Task_v03/SubUSDs/Danny.usd",
)
C4_2_USD = os.path.join(PPSUC_ASSETS_MODEL_DIR, "scene/c4_2_rigid_wrapper.usd")

# -----------------------------------------------------------------------------
# Layout (world coords, ground plane at z=0).  Verified with the robot at the
# default standing pose (base z=0.80): body reaches x to +0.39 (fingertips) and
# z to 0.11 (feet).  The Danny table bbox (scale 0.01) spans ~[-0.26, 1.16] in
# x when its root is at x=0.45 -- that overlaps the robot's front legs/feet.
#
# To avoid any initial collision we push the table so its left edge clears the
# robot's front (table root x >= 0.391 + half-width).  The object then sits on
# the table's near edge, a short reach (arm ~0.2) in front of the fingertips.
#   robot base    -> (0.00, 0.00, 0.81)
#   Danny table   -> root (0.90, 0.00, -0.004), scale 0.01
#                     left edge ~0.19 (>= robot body to x 0.391 in body-only:
#                     front legs to 0.334; fingertips at 0.391 are the arm)
#   C4_2          -> near table edge, mesh center ~(0.50, 0.045, 1.07)
# -----------------------------------------------------------------------------
# Table root x = robot-front(0.334) + table half-width(0.71) => left edge 0.39,
# which clears the robot's front legs/feet (to 0.334).  Fingertips are at
# x=0.391, so the object placed at the table's near edge sits ~0.06 m in front
# of the fingertips -- a short reach, no walking needed.
ROBOT_POS = (0.0, 0.0, 0.81)
TABLE_POS = (1.10, 0.0, -0.004)
# C4_2 wrapper mesh is offset +x 0.056 / +y 0.011 from its root.  Place the
# object just in front of the fingertips, above the tabletop so it settles.
OBJECT_POS = (0.45, 0.045, 1.07)
OBJECT_INITIAL_HEIGHT = OBJECT_POS[2]
OBJECT_ROT = (1.0, 0.0, 0.0, 0.0)

LEG_JOINTS = [
    "FR_hip_joint", "FR_thigh_joint", "FR_calf_joint",
    "FL_hip_joint", "FL_thigh_joint", "FL_calf_joint",
    "RR_hip_joint", "RR_thigh_joint", "RR_calf_joint",
    "RL_hip_joint", "RL_thigh_joint", "RL_calf_joint",
]
WHEEL_JOINTS = ["FR_foot_joint", "FL_foot_joint", "RR_foot_joint", "RL_foot_joint"]
# arm_joint1..6 are the arm DoF; arm_joint7/8 are the prismatic gripper fingers.
ARM_JOINTS = [f"arm_joint{i}" for i in range(1, 7)]
GRIPPER_JOINTS = ["arm_joint7", "arm_joint8"]
ALL_JOINTS = LEG_JOINTS + WHEEL_JOINTS + ARM_JOINTS + GRIPPER_JOINTS


@configclass
class C4_2SceneCfg(InteractiveSceneCfg):
    """Robot beside the table, object under the fingertips."""

    robot: ArticulationCfg = UNITREE_B2W_PIPER_CFG.replace(
        prim_path="{ENV_REGEX_NS}/Robot",
        spawn=UsdFileCfg(usd_path=B2W_PIPER_USD),
        init_state=UNITREE_B2W_PIPER_CFG.init_state.replace(pos=ROBOT_POS),
    )
    # Danny table (cm-scale asset -> 0.01).  Tabletop lands at ~1.01 m.
    table = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Table",
        init_state=AssetBaseCfg.InitialStateCfg(pos=TABLE_POS),
        spawn=UsdFileCfg(usd_path=DANNY_USD, scale=(0.01, 0.01, 0.01)),
    )
    object: RigidObjectCfg = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/C4_2",
        init_state=RigidObjectCfg.InitialStateCfg(pos=OBJECT_POS, rot=OBJECT_ROT),
        spawn=UsdFileCfg(
            usd_path=C4_2_USD,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                rigid_body_enabled=True,
                kinematic_enabled=False,
                disable_gravity=False,
                linear_damping=0.15,
                angular_damping=0.15,
                max_depenetration_velocity=2.0,
            ),
            mass_props=sim_utils.MassPropertiesCfg(mass=0.35),
            collision_props=sim_utils.CollisionPropertiesCfg(contact_offset=0.005, rest_offset=0.0),
        ),
    )
    ee_frame = FrameTransformerCfg(
        prim_path="{ENV_REGEX_NS}/Robot/base_link",
        target_frames=[
            FrameTransformerCfg.FrameCfg(
                prim_path="{ENV_REGEX_NS}/Robot/gripper_base",
                name="end_effector",
            )
        ],
    )
    # Track the two fingertip links so the policy can observe their positions.
    fingertip_frame = FrameTransformerCfg(
        prim_path="{ENV_REGEX_NS}/Robot/base_link",
        target_frames=[
            FrameTransformerCfg.FrameCfg(prim_path="{ENV_REGEX_NS}/Robot/arm_link7", name="finger_left"),
            FrameTransformerCfg.FrameCfg(prim_path="{ENV_REGEX_NS}/Robot/arm_link8", name="finger_right"),
        ],
    )
    ground = AssetBaseCfg(
        prim_path="/World/GroundPlane",
        spawn=GroundPlaneCfg(),
    )
    light = AssetBaseCfg(
        prim_path="/World/Light",
        spawn=sim_utils.DomeLightCfg(intensity=1800.0, color=(0.75, 0.75, 0.75)),
    )


@configclass
class C4_2ActionsCfg:
    joint_leg = base_mdp.JointPositionActionCfg(
        asset_name="robot",
        joint_names=LEG_JOINTS,
        scale={".*_hip_joint": 0.10, ".*_thigh_joint": 0.18, ".*_calf_joint": 0.18},
        use_default_offset=True,
        clip={".*": (-100.0, 100.0)},
        preserve_order=True,
    )
    joint_wheel = base_mdp.JointVelocityActionCfg(
        asset_name="robot", joint_names=WHEEL_JOINTS, scale=3.0, clip={".*": (-20.0, 20.0)}, preserve_order=True
    )
    joint_arm = base_mdp.JointPositionActionCfg(
        asset_name="robot",
        joint_names=ARM_JOINTS,
        scale=0.22,
        use_default_offset=True,
        clip={".*": (-100.0, 100.0)},
        preserve_order=True,
    )
    # Gripper fingers are prismatic (arm_joint7: 0..0.035, arm_joint8: -0.035..0).
    joint_gripper = base_mdp.JointPositionActionCfg(
        asset_name="robot",
        joint_names=GRIPPER_JOINTS,
        scale={"arm_joint7": 0.04, "arm_joint8": 0.04},
        use_default_offset=True,
        clip={".*": (-100.0, 100.0)},
        preserve_order=True,
    )


@configclass
class C4_2ObservationsCfg:
    @configclass
    class PolicyCfg(ObsGroup):
        base_lin_vel = ObsTerm(func=base_mdp.base_lin_vel, scale=1.0, clip=(-100.0, 100.0))
        base_ang_vel = ObsTerm(func=base_mdp.base_ang_vel, scale=0.25, clip=(-100.0, 100.0))
        projected_gravity = ObsTerm(func=base_mdp.projected_gravity, scale=1.0, clip=(-100.0, 100.0))
        joint_pos = ObsTerm(
            func=base_mdp.joint_pos_rel,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=ALL_JOINTS, preserve_order=True)},
            scale=1.0,
            clip=(-100.0, 100.0),
        )
        joint_vel = ObsTerm(
            func=base_mdp.joint_vel_rel,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=ALL_JOINTS, preserve_order=True)},
            scale=0.05,
            clip=(-100.0, 100.0),
        )
        object_position = ObsTerm(
            func=lift_mdp.object_position_in_robot_root_frame,
            params={"robot_cfg": SceneEntityCfg("robot"), "object_cfg": SceneEntityCfg("object")},
            scale=1.0,
            clip=(-100.0, 100.0),
        )
        ee_position = ObsTerm(
            func=grasp_mdp.ee_position_in_robot_root_frame,
            params={"robot_cfg": SceneEntityCfg("robot"), "ee_frame_cfg": SceneEntityCfg("ee_frame")},
            scale=1.0,
            clip=(-100.0, 100.0),
        )
        object_height_delta = ObsTerm(
            func=grasp_mdp.object_height_delta_obs,
            params={"initial_height": OBJECT_INITIAL_HEIGHT},
            scale=10.0,
            clip=(-1.0, 1.0),
        )
        object_rel_ee = ObsTerm(
            func=grasp_mdp.object_relative_to_ee,
            params={
                "object_cfg": SceneEntityCfg("object"),
                "ee_frame_cfg": SceneEntityCfg("ee_frame"),
            },
            scale=1.0,
            clip=(-10.0, 10.0),
        )
        gripper_opening = ObsTerm(
            func=grasp_mdp.gripper_opening,
            params={"robot_cfg": SceneEntityCfg("robot")},
            scale=1.0,
            clip=(0.0, 0.1),
        )
        fingertip_object_distance = ObsTerm(
            func=grasp_mdp.fingertip_object_distance,
            params={"object_cfg": SceneEntityCfg("object"), "robot_cfg": SceneEntityCfg("robot")},
            scale=1.0,
            clip=(0.0, 1.0),
        )
        fingertip_contact = ObsTerm(
            func=grasp_mdp.fingertip_contact_obs,
            params={"robot_cfg": SceneEntityCfg("robot")},
            scale=1.0,
            clip=(0.0, 1.0),
        )
        actions = ObsTerm(func=base_mdp.last_action, scale=1.0, clip=(-100.0, 100.0))

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()


@configclass
class C4_2EventCfg:
    reset_all = EventTerm(func=base_mdp.reset_scene_to_default, mode="reset")
    reset_object = EventTerm(
        func=base_mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {"x": (-0.04, 0.04), "y": (-0.04, 0.04), "z": (0.0, 0.0)},
            "velocity_range": {"x": (0.0, 0.0), "y": (0.0, 0.0), "z": (0.0, 0.0)},
            "asset_cfg": SceneEntityCfg("object"),
        },
    )


@configclass
class C4_2RewardsCfg:
    reaching = RewTerm(
        func=grasp_mdp.reaching_object,
        weight=2.5,
        params={"std": 0.18},
    )
    lift_progress = RewTerm(
        func=grasp_mdp.lift_progress,
        weight=12.0,
        params={"initial_height": OBJECT_INITIAL_HEIGHT, "scale_height": 0.10},
    )
    lifted = RewTerm(
        func=grasp_mdp.object_lifted,
        weight=35.0,
        params={"initial_height": OBJECT_INITIAL_HEIGHT, "minimal_lift": 0.035},
    )
    hold = RewTerm(
        func=grasp_mdp.lift_hold,
        weight=8.0,
        params={"initial_height": OBJECT_INITIAL_HEIGHT, "minimal_lift": 0.035},
    )
    alignment = RewTerm(
        func=grasp_mdp.gripper_alignment,
        weight=1.5,
        params={"object_cfg": SceneEntityCfg("object"), "ee_frame_cfg": SceneEntityCfg("ee_frame")},
    )
    fingertips_around = RewTerm(
        func=grasp_mdp.fingertips_around_object,
        weight=1.0,
        params={"object_cfg": SceneEntityCfg("object"), "robot_cfg": SceneEntityCfg("robot")},
    )
    gripper_contact = RewTerm(
        func=grasp_mdp.gripper_closed_with_contact,
        weight=3.0,
        params={"robot_cfg": SceneEntityCfg("robot"), "contact_threshold": 0.5},
    )
    action_rate = RewTerm(func=base_mdp.action_rate_l2, weight=-0.01)
    joint_velocity = RewTerm(
        func=base_mdp.joint_vel_l2,
        weight=-0.0001,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=ALL_JOINTS, preserve_order=True)},
    )


@configclass
class C4_2TerminationsCfg:
    time_out = DoneTerm(func=base_mdp.time_out, time_out=True)
    object_dropped = DoneTerm(
        func=grasp_mdp.object_dropped,
        params={"initial_height": OBJECT_INITIAL_HEIGHT, "tolerance": 0.12},
    )
    robot_fallen = DoneTerm(func=grasp_mdp.robot_fallen, params={"minimum_height": 0.30})


@configclass
class C4_2CurriculumCfg:
    pass


@configclass
class UnitreeB2WPiperC4_2GraspEnvCfg(ManagerBasedRLEnvCfg):
    scene: C4_2SceneCfg = C4_2SceneCfg(num_envs=256, env_spacing=3.0)
    observations: C4_2ObservationsCfg = C4_2ObservationsCfg()
    actions: C4_2ActionsCfg = C4_2ActionsCfg()
    rewards: C4_2RewardsCfg = C4_2RewardsCfg()
    terminations: C4_2TerminationsCfg = C4_2TerminationsCfg()
    events: C4_2EventCfg = C4_2EventCfg()
    curriculum: C4_2CurriculumCfg = C4_2CurriculumCfg()

    def __post_init__(self):
        self.decimation = 4
        self.episode_length_s = 8.0
        self.sim.dt = 0.005
        self.sim.render_interval = self.decimation
        self.sim.physx.bounce_threshold_velocity = 0.2
        self.sim.physx.friction_correlation_distance = 0.00625
        self.sim.physx.gpu_found_lost_aggregate_pairs_capacity = 1024 * 1024
        self.sim.physx.gpu_total_aggregate_pairs_capacity = 16384


@configclass
class UnitreeB2WPiperC4_2GraspEnvCfg_PLAY(UnitreeB2WPiperC4_2GraspEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 16
        self.observations.policy.enable_corruption = False

import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg
from isaaclab.utils import configclass

from ppsuc_rl_lab.train.locomotion.velocity.config.quadruped.unitree_b2w_piper.collision_resilience_d1_env_cfg import (
    UnitreeB2WPiperFlatCollisionResilienceD1EnvCfg,
)


def _static_obstacle_cfg(name: str, pos: tuple[float, float, float], size: tuple[float, float, float], color):
    return AssetBaseCfg(
        prim_path=f"{{ENV_REGEX_NS}}/{name}",
        spawn=sim_utils.CuboidCfg(
            size=size,
            collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=True),
            physics_material=sim_utils.RigidBodyMaterialCfg(
                static_friction=0.8,
                dynamic_friction=0.7,
                restitution=0.05,
            ),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=color),
        ),
        init_state=AssetBaseCfg.InitialStateCfg(pos=pos),
    )


@configclass
class UnitreeB2WPiperFlatCollisionResilienceD2EnvCfg(UnitreeB2WPiperFlatCollisionResilienceD1EnvCfg):
    """D2: low-risk flat-ground collision recovery.

    Adds simple static side obstacles that behave like soft doorway/corridor contacts. The task should still
    reward forward velocity tracking and recovery, while penalizing body contacts and falls.
    """

    def __post_init__(self):
        super().__post_init__()

        self.scene.env_spacing = 4.0
        self.scene.terrain.env_spacing = 4.0

        # Low-risk corridor bars. The gap is intentionally wider than the dog body,
        # so good walking passes through, while yaw drift creates side rubs.
        self.scene.left_collision_bar = _static_obstacle_cfg(
            "LeftCollisionBar",
            pos=(1.15, 0.78, 0.42),
            size=(2.8, 0.12, 0.84),
            color=(0.65, 0.10, 0.08),
        )
        self.scene.right_collision_bar = _static_obstacle_cfg(
            "RightCollisionBar",
            pos=(1.15, -0.78, 0.42),
            size=(2.8, 0.12, 0.84),
            color=(0.65, 0.10, 0.08),
        )
        self.scene.exit_rub_block_left = _static_obstacle_cfg(
            "ExitRubBlockLeft",
            pos=(2.75, 0.48, 0.30),
            size=(0.22, 0.28, 0.60),
            color=(0.80, 0.45, 0.05),
        )
        self.scene.exit_rub_block_right = _static_obstacle_cfg(
            "ExitRubBlockRight",
            pos=(2.75, -0.48, 0.30),
            size=(0.22, 0.28, 0.60),
            color=(0.80, 0.45, 0.05),
        )

        # D2 should begin roughly aligned with the corridor. Large yaw resets are for later stages.
        self.events.randomize_reset_base.params["pose_range"].update(
            {"x": (-0.20, 0.20), "y": (-0.15, 0.15), "roll": (-0.12, 0.12), "pitch": (-0.12, 0.12), "yaw": (-0.35, 0.35)}
        )
        self.events.randomize_reset_base.params["velocity_range"].update(
            {"x": (-0.35, 0.35), "y": (-0.35, 0.35), "roll": (-0.25, 0.25), "pitch": (-0.25, 0.25), "yaw": (-0.35, 0.35)}
        )

        # Keep this lower than D1 at first because obstacle contacts add difficulty.
        self.events.randomize_apply_external_force_torque.params["force_range"] = (-90.0, 90.0)
        self.events.randomize_apply_external_force_torque.params["torque_range"] = (-25.0, 25.0)
        self.events.randomize_push_robot.interval_range_s = (3.0, 5.0)
        self.events.randomize_push_robot.params["velocity_range"] = {
            "x": (-0.65, 0.65),
            "y": (-0.65, 0.65),
            "z": (0.0, 0.0),
            "roll": (-0.18, 0.18),
            "pitch": (-0.18, 0.18),
            "yaw": (-0.5, 0.5),
        }

        self.commands.base_velocity.resampling_time_range = (3.0, 5.0)
        self.commands.base_velocity.ranges.lin_vel_x = (0.25, 0.9)
        self.commands.base_velocity.ranges.lin_vel_y = (0.0, 0.0)
        self.commands.base_velocity.ranges.ang_vel_z = (-0.45, 0.45)

        # D2 accepts brief rubbing but should avoid sustained body contacts and hard foot impacts.
        self.rewards.undesired_contacts.weight = -1.0
        self.rewards.contact_forces.weight = -2.5e-4
        self.rewards.track_lin_vel_xy_exp.weight = 3.2
        self.rewards.track_ang_vel_z_exp.weight = 1.5
        self.rewards.upward.weight = 3.5
        self.rewards.is_terminated.weight = -25.0


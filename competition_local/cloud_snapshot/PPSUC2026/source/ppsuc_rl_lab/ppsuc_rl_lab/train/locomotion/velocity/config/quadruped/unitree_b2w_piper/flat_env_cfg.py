from isaaclab.utils import configclass

from ppsuc_rl_lab.project_paths import ROBOT_MODEL_ROOT
from ppsuc_rl_lab.train.locomotion.velocity.config.quadruped.unitree_b2w_piper.rough_env_cfg import (
    UnitreeB2WPiperRoughEnvCfg,
)


@configclass
class UnitreeB2WPiperFlatEnvCfg(UnitreeB2WPiperRoughEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        if self.rewards.base_height_l2 is not None:
            self.rewards.base_height_l2.params["sensor_cfg"] = None
        self.scene.env_spacing = 4.0
        self.scene.terrain.terrain_type = "usd"
        self.scene.terrain.usd_path = str(ROBOT_MODEL_ROOT / "scene/plane/default_environment.usd")
        self.scene.terrain.terrain_generator = None
        self.scene.height_scanner = None
        self.observations.policy.height_scan = None
        self.observations.critic.height_scan = None
        self.curriculum.terrain_levels = None
        self.terminations.terrain_out_of_bounds = None

"""Random-initialized PPO settings for C4_2 grasping."""

from isaaclab.utils import configclass

from ppsuc_rl_lab.train.locomotion.velocity.config.quadruped.unitree_b2.agents.rsl_rl_ppo_cfg import (
    UnitreeB2RoughPPORunnerCfg,
)


@configclass
class UnitreeB2WPiperC4_2GraspPPORunnerCfg(UnitreeB2RoughPPORunnerCfg):
    def __post_init__(self):
        super().__post_init__()
        self.experiment_name = "unitree_b2w_piper_c4_2_grasp_from_scratch"
        self.max_iterations = 10000
        self.save_interval = 250
        self.policy.init_noise_std = 0.8
        self.algorithm.entropy_coef = 0.003
        self.algorithm.learning_rate = 3.0e-4
        self.algorithm.desired_kl = 0.01

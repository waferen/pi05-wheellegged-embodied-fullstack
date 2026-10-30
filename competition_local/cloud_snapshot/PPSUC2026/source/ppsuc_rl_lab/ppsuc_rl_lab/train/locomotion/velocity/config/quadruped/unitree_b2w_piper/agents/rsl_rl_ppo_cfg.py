 
from isaaclab.utils import configclass

from ppsuc_rl_lab.train.locomotion.velocity.config.quadruped.unitree_b2.agents.rsl_rl_ppo_cfg import (
    UnitreeB2RoughPPORunnerCfg,
)


@configclass
class UnitreeB2WPiperFlatArmHighPPORunnerCfg(UnitreeB2RoughPPORunnerCfg):
    def __post_init__(self):
        super().__post_init__()
        self.max_iterations = 5000
        self.experiment_name = "unitree_b2w_piper_flat_arm_high"


@configclass
class UnitreeB2WPiperImpulseRecoveryC1PPORunnerCfg(UnitreeB2WPiperFlatArmHighPPORunnerCfg):
    def __post_init__(self):
        super().__post_init__()
        self.max_iterations = 3000
        self.save_interval = 100
        self.experiment_name = "unitree_b2w_piper_impulse_recovery_c1"


@configclass
class UnitreeB2WPiperImpulseRecoveryC2PPORunnerCfg(UnitreeB2WPiperImpulseRecoveryC1PPORunnerCfg):
    def __post_init__(self):
        super().__post_init__()
        self.max_iterations = 3000
        self.save_interval = 100
        self.experiment_name = "unitree_b2w_piper_impulse_recovery_c2"


@configclass
class UnitreeB2WPiperCollisionResilienceD1PPORunnerCfg(UnitreeB2WPiperFlatArmHighPPORunnerCfg):
    def __post_init__(self):
        super().__post_init__()
        self.max_iterations = 3000
        self.save_interval = 100
        self.experiment_name = "unitree_b2w_piper_collision_d1"


@configclass
class UnitreeB2WPiperCollisionResilienceD2PPORunnerCfg(UnitreeB2WPiperCollisionResilienceD1PPORunnerCfg):
    def __post_init__(self):
        super().__post_init__()
        self.max_iterations = 3000
        self.save_interval = 100
        self.experiment_name = "unitree_b2w_piper_collision_d2"


@configclass
class UnitreeB2WPiperCollisionResilienceD3PPORunnerCfg(UnitreeB2WPiperCollisionResilienceD2PPORunnerCfg):
    def __post_init__(self):
        super().__post_init__()
        self.max_iterations = 3000
        self.save_interval = 100
        self.experiment_name = "unitree_b2w_piper_collision_d3"


@configclass
class UnitreeB2WPiperCollisionResilienceD3APPORunnerCfg(UnitreeB2WPiperCollisionResilienceD2PPORunnerCfg):
    def __post_init__(self):
        super().__post_init__()
        self.max_iterations = 1500
        self.save_interval = 100
        self.experiment_name = "unitree_b2w_piper_collision_d3a"


@configclass
class UnitreeB2WPiperCollisionResilienceD3BPPORunnerCfg(UnitreeB2WPiperCollisionResilienceD3APPORunnerCfg):
    def __post_init__(self):
        super().__post_init__()
        self.max_iterations = 3000
        self.save_interval = 100
        self.experiment_name = "unitree_b2w_piper_collision_d3b"


@configclass
class UnitreeB2WPiperRecoveryR1PPORunnerCfg(UnitreeB2WPiperCollisionResilienceD2PPORunnerCfg):
    def __post_init__(self):
        super().__post_init__()
        self.max_iterations = 1500
        self.save_interval = 100
        self.experiment_name = "unitree_b2w_piper_recovery_r1"


@configclass
class UnitreeB2WPiperRecoveryR3PPORunnerCfg(UnitreeB2WPiperRecoveryR1PPORunnerCfg):
    def __post_init__(self):
        super().__post_init__()
        self.max_iterations = 2500
        self.save_interval = 100
        self.experiment_name = "unitree_b2w_piper_recovery_r3"


@configclass
class UnitreeB2WPiperCerebellumPPORunnerCfg(UnitreeB2WPiperFlatArmHighPPORunnerCfg):
    def __post_init__(self):
        super().__post_init__()
        self.max_iterations = 5000
        self.save_interval = 100
        self.experiment_name = "unitree_b2w_piper_cerebellum"
        self.policy.init_noise_std = 0.8
        self.algorithm.entropy_coef = 0.0005
        self.algorithm.learning_rate = 2.0e-4
        self.algorithm.desired_kl = 0.006


@configclass
class UnitreeB2WPiperHeightStabilizerPPORunnerCfg(UnitreeB2WPiperCerebellumPPORunnerCfg):
    def __post_init__(self):
        super().__post_init__()
        self.max_iterations = 3500
        self.save_interval = 100
        self.experiment_name = "unitree_b2w_piper_height_stabilizer"
        self.policy.init_noise_std = 0.55
        self.algorithm.entropy_coef = 0.00025
        self.algorithm.learning_rate = 1.5e-4
        self.algorithm.desired_kl = 0.004


@configclass
class UnitreeB2WPiperRoughPPORunnerCfg(UnitreeB2WPiperFlatArmHighPPORunnerCfg):
    """Rough terrain (stairs + rough) training for B2W+Piper."""
    def __post_init__(self):
        super().__post_init__()
        self.max_iterations = 20000
        self.experiment_name = "unitree_b2w_piper_rough"

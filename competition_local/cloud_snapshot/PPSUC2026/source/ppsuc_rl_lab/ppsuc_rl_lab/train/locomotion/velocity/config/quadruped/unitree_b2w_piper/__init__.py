 
import gymnasium as gym

from . import agents


gym.register(
    id="PPSUC-Isaac-Velocity-TaskScene-NativeRobot-Slow-ArmFixed-Unitree-B2W-Piper-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.task_scene_native_slow_arm_fixed_env_cfg:UnitreeB2WPiperTaskSceneNativeSlowArmFixedEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:UnitreeB2WPiperFlatArmHighPPORunnerCfg",
    },
)

gym.register(
    id="PPSUC-Isaac-Velocity-Flat-Slow-ArmFixed-Unitree-B2W-Piper-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.flat_slow_arm_fixed_env_cfg:UnitreeB2WPiperFlatSlowArmFixedEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:UnitreeB2WPiperFlatArmHighPPORunnerCfg",
    },
)


gym.register(
    id="PPSUC-Isaac-Velocity-Flat-ImpulseRecovery-C1-Unitree-B2W-Piper-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.impulse_recovery_c1_env_cfg:UnitreeB2WPiperFlatImpulseRecoveryC1EnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:UnitreeB2WPiperImpulseRecoveryC1PPORunnerCfg",
    },
)


gym.register(
    id="PPSUC-Isaac-Velocity-Flat-ImpulseRecovery-C2-Unitree-B2W-Piper-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.impulse_recovery_c2_env_cfg:UnitreeB2WPiperFlatImpulseRecoveryC2EnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:UnitreeB2WPiperImpulseRecoveryC2PPORunnerCfg",
    },
)


gym.register(
    id="PPSUC-Isaac-Velocity-Flat-CollisionResilience-D1-Unitree-B2W-Piper-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.collision_resilience_d1_env_cfg:UnitreeB2WPiperFlatCollisionResilienceD1EnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:UnitreeB2WPiperCollisionResilienceD1PPORunnerCfg",
    },
)


gym.register(
    id="PPSUC-Isaac-Velocity-Flat-CollisionResilience-D2-Unitree-B2W-Piper-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.collision_resilience_d2_env_cfg:UnitreeB2WPiperFlatCollisionResilienceD2EnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:UnitreeB2WPiperCollisionResilienceD2PPORunnerCfg",
    },
)


gym.register(
    id="PPSUC-Isaac-Velocity-Flat-CollisionResilience-D3-Unitree-B2W-Piper-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.collision_resilience_d3_env_cfg:UnitreeB2WPiperFlatCollisionResilienceD3EnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:UnitreeB2WPiperCollisionResilienceD3PPORunnerCfg",
    },
)


gym.register(
    id="PPSUC-Isaac-Velocity-Flat-CollisionResilience-D3A-Unitree-B2W-Piper-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.collision_resilience_d3a_env_cfg:UnitreeB2WPiperFlatCollisionResilienceD3AEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:UnitreeB2WPiperCollisionResilienceD3APPORunnerCfg",
    },
)


gym.register(
    id="PPSUC-Isaac-Velocity-Flat-CollisionResilience-D3B-Unitree-B2W-Piper-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.collision_resilience_d3b_env_cfg:UnitreeB2WPiperFlatCollisionResilienceD3BEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:UnitreeB2WPiperCollisionResilienceD3BPPORunnerCfg",
    },
)


gym.register(
    id="PPSUC-Isaac-Velocity-Flat-Recovery-R1-Unitree-B2W-Piper-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.recovery_r1_env_cfg:UnitreeB2WPiperFlatRecoveryR1EnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:UnitreeB2WPiperRecoveryR1PPORunnerCfg",
    },
)


gym.register(
    id="PPSUC-Isaac-Velocity-Flat-Recovery-R3-Unitree-B2W-Piper-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.recovery_r3_env_cfg:UnitreeB2WPiperFlatRecoveryR3EnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:UnitreeB2WPiperRecoveryR3PPORunnerCfg",
    },
)

gym.register(
    id="PPSUC-Isaac-Velocity-TaskScene2-NativeRobot-Slow-ArmFixed-Unitree-B2W-Piper-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.task02_native_slow_arm_fixed_env_cfg:UnitreeB2WPiperTask02NativeSlowArmFixedEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:UnitreeB2WPiperFlatArmHighPPORunnerCfg",
    },
)

gym.register(
    id="PPSUC-Isaac-Velocity-TaskScene2-NativeRobot-Cerebellum-Unitree-B2W-Piper-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.task02_native_slow_arm_fixed_env_cfg:UnitreeB2WPiperTask02NativeCerebellumEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:UnitreeB2WPiperCerebellumPPORunnerCfg",
    },
)

gym.register(
    id="PPSUC-Isaac-Velocity-Squat-Flat-Unitree-B2W-Piper-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.squat_env_cfg:UnitreeB2WPiperSquatFlatEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:UnitreeB2WPiperFlatArmHighPPORunnerCfg",
    },
)

gym.register(
    id="PPSUC-Isaac-Velocity-ArmStability-Flat-Unitree-B2W-Piper-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.arm_stability_env_cfg:UnitreeB2WPiperArmStabilityEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:UnitreeB2WPiperFlatArmHighPPORunnerCfg",
    },
)

gym.register(
    id="PPSUC-Isaac-Velocity-Cerebellum-Flat-Unitree-B2W-Piper-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.cerebellum_env_cfg:UnitreeB2WPiperCerebellumFlatEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:UnitreeB2WPiperCerebellumPPORunnerCfg",
    },
)

gym.register(
    id="PPSUC-Isaac-Velocity-CerebellumStage2-Flat-Unitree-B2W-Piper-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.cerebellum_env_cfg:UnitreeB2WPiperCerebellumStage2FlatEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:UnitreeB2WPiperCerebellumPPORunnerCfg",
    },
)

gym.register(
    id="PPSUC-Isaac-Velocity-CerebellumFinal-Flat-Unitree-B2W-Piper-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.cerebellum_env_cfg:UnitreeB2WPiperCerebellumFinalFlatEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:UnitreeB2WPiperCerebellumPPORunnerCfg",
    },
)

gym.register(
    id="PPSUC-Isaac-Velocity-HeightStabilizer-Flat-Unitree-B2W-Piper-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.cerebellum_env_cfg:UnitreeB2WPiperHeightStabilizerFlatEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:UnitreeB2WPiperHeightStabilizerPPORunnerCfg",
    },
)

gym.register(
    id="PPSUC-Isaac-Velocity-CerebellumStage3-Flat-Unitree-B2W-Piper-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.cerebellum_env_cfg:UnitreeB2WPiperCerebellumStage3FlatEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:UnitreeB2WPiperCerebellumPPORunnerCfg",
    },
)

gym.register(
    id="PPSUC-Isaac-Velocity-CerebellumStage4-Flat-Unitree-B2W-Piper-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.cerebellum_env_cfg:UnitreeB2WPiperCerebellumStage4FlatEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:UnitreeB2WPiperCerebellumPPORunnerCfg",
    },
)


gym.register(
    id="PPSUC-Isaac-Velocity-Rough-Unitree-B2W-Piper-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.rough_env_cfg:UnitreeB2WPiperRoughEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:UnitreeB2WPiperRoughPPORunnerCfg",
    },
)

gym.register(
    id="PPSUC-Isaac-Task03-C4-2-Grasp-B2W-Piper-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.task03_c4_2_grasp_env_cfg:UnitreeB2WPiperC4_2GraspEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.task03_c4_2_grasp_ppo_cfg:UnitreeB2WPiperC4_2GraspPPORunnerCfg",
    },
)

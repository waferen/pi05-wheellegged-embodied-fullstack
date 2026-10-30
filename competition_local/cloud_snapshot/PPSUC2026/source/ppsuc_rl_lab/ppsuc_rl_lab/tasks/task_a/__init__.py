import gymnasium as gym

gym.register(
    id = "PPSUC-TaskA-G1",
    entry_point="ppsuc_rl_lab.tasks.task_base.envs_base:BaseRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.env_cfg:TaskAEnvG1Cfg"
    },
)

gym.register(
    id = "PPSUC-TaskA-Tron1Piper",
    entry_point="ppsuc_rl_lab.tasks.task_base.envs_base:BaseRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.env_cfg:TaskAEnvTron1Cfg"
    },
)

gym.register(
    id = "PPSUC-TaskA-Tron2ALegged",
    entry_point="ppsuc_rl_lab.tasks.task_base.envs_base:BaseRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.env_cfg:TaskAEnvTron2ALeggedCfg"
    },
)

gym.register(
    id = "PPSUC-TaskA-Tron2AWheel",
    entry_point="ppsuc_rl_lab.tasks.task_base.envs_base:BaseRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.env_cfg:TaskAEnvTron2AWheelCfg"
    },
)

gym.register(
    id = "PPSUC-TaskA-B2Piper",
    entry_point="ppsuc_rl_lab.tasks.task_base.envs_base:BaseRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.env_cfg:TaskAEnvB2Cfg"
    },
)

gym.register(
    id = "PPSUC-TaskA-B2wPiper",
    entry_point="ppsuc_rl_lab.tasks.task_base.envs_base:BaseRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.env_cfg:TaskAEnvB2WCfg"
    },
)

from .env_cfg import TaskAEnvCfg, TaskAEnvB2Cfg, TaskAEnvTron2ALeggedCfg, TaskAEnvTron2AWheelCfg
__all__ = ['TaskAEnvCfg', 'TaskAEnvB2Cfg', 'TaskAEnvTron2ALeggedCfg', 'TaskAEnvTron2AWheelCfg']

import gymnasium as gym

from .terrain import TASK_D_TERRAIN_CFG
from .env_cfg import TaskDEnvCfg, TaskDEnvB2Cfg, TaskDEnvTron2ALeggedCfg, TaskDEnvTron2AWheelCfg


gym.register(
    id = "PPSUC-TaskD-G1",
    entry_point="ppsuc_rl_lab.tasks.task_base.envs_base:BaseRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.env_cfg:TaskDEnvG1Cfg"
    },
)

gym.register(
    id = "PPSUC-TaskD-Tron1Piper",
    entry_point="ppsuc_rl_lab.tasks.task_base.envs_base:BaseRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.env_cfg:TaskDEnvTron1Cfg"
    },
)

gym.register(
    id = "PPSUC-TaskD-Tron2ALegged",
    entry_point="ppsuc_rl_lab.tasks.task_base.envs_base:BaseRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.env_cfg:TaskDEnvTron2ALeggedCfg"
    },
)

gym.register(
    id = "PPSUC-TaskD-Tron2AWheel",
    entry_point="ppsuc_rl_lab.tasks.task_base.envs_base:BaseRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.env_cfg:TaskDEnvTron2AWheelCfg"
    },
)

gym.register(
    id = "PPSUC-TaskD-B2Piper",
    entry_point="ppsuc_rl_lab.tasks.task_base.envs_base:BaseRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.env_cfg:TaskDEnvB2Cfg"
    },
)

gym.register(
    id = "PPSUC-TaskD-B2wPiper",
    entry_point="ppsuc_rl_lab.tasks.task_base.envs_base:BaseRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.env_cfg:TaskDEnvB2WCfg"
    },
)

__all__ = ['TaskDEnvCfg', 'TaskDEnvB2Cfg', 'TaskDEnvTron2ALeggedCfg', 'TaskDEnvTron2AWheelCfg']

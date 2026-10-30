import gymnasium as gym

gym.register(
    id="PPSUC-TaskE-Piper",
    entry_point="ppsuc_rl_lab.tasks.task_base.envs_base:BaseRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.env_cfg:TaskEEnvPiperCfg",
        "rsl_rl_cfg_entry_point": "ppsuc_rl_lab.train.task_e.agents.rsl_rl_ppo_cfg:TaskESmokePPORunnerCfg",
    },
)

from .env_cfg import TaskEEnvPiperCfg

__all__ = ["TaskEEnvPiperCfg"]

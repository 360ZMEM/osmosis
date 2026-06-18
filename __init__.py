# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""EasyUUV STDW environment package.

@file __init__.py
@brief Gym registration entry for EasyUUV-Direct-v1 (4D baseline) and
       EasyUUV-Direct-Parametric-v1 (8D meta-control).
@details See docs/INDEX.md for the documentation tree, docs/guide/COMMAND_CONTRACT.md
         for runnable commands, and docs/guide/AGENT_HANDOFF.md for the AI continuation
         brief.
"""

try:
    import gymnasium as gym

    from . import agents
    from .easyuuv_env import EasyUUVEnv, EasyUUVEnvCfg, EasyUUVParametricEnvCfg
except ModuleNotFoundError as exc:
    # Keep easyuuv_stdw.eval importable on onboard computers that do not ship
    # Isaac Sim / omni.kit.  Environment registration is only needed inside
    # Isaac Lab workflows.
    if not str(exc).startswith("No module named 'omni"):
        raise
    gym = None
    agents = None
    EasyUUVEnv = EasyUUVEnvCfg = EasyUUVParametricEnvCfg = None

##
# Register Gym environments.
##

if gym is not None:
    gym.register(
        id="EasyUUV-Direct-v1",
        entry_point="omni.isaac.lab_tasks.direct.easyuuv_stdw:EasyUUVEnv",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": EasyUUVEnvCfg,
            "rsl_rl_cfg_entry_point": agents.rsl_rl_ppo_cfg.EasyUUVPPORunnerCfg,
        },
    )

# 8D meta-control parallel task (4 ctrl + 4 a_gain handled by ParametricGainTuner).
# Coexists with the 4D baseline; isolated via a separate cfg subclass.
    gym.register(
        id="EasyUUV-Direct-Parametric-v1",
        entry_point="omni.isaac.lab_tasks.direct.easyuuv_stdw:EasyUUVEnv",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": EasyUUVParametricEnvCfg,
            "rsl_rl_cfg_entry_point": agents.rsl_rl_ppo_cfg.EasyUUVParametricPPORunnerCfg,
        },
    )

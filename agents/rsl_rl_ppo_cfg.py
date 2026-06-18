# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from omni.isaac.lab.utils import configclass

from omni.isaac.lab_tasks.utils.wrappers.rsl_rl import (
    RslRlOnPolicyRunnerCfg,
    RslRlPpoActorCriticCfg,
    RslRlPpoAlgorithmCfg,
)


@configclass
class EasyUUVPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    num_steps_per_env = 24
    max_iterations = 800
    save_interval = 50
    experiment_name = "easyuuv_direct"
    empirical_normalization = False
    policy = RslRlPpoActorCriticCfg(
        init_noise_std=1.0,
        actor_hidden_dims=[64, 64],
        critic_hidden_dims=[64, 64],
        activation="elu",
    )
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.0,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=5.0e-4,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )


# =============================================================================
# 8 维元控制 PPO Runner cfg
# =============================================================================
# 在 4 维基线之上：
#   - experiment_name 切换到 ``easyuuv_parametric`` 以避免与 4 维 checkpoint 混淆。
#   - 网络容量从 [64, 64] 升到 [128, 128]，以承载新增的 4 维 a_gain 输出。
#   - init_noise_std 从 1.0 降到 0.8，避免初始 8 维高方差行为把仿真直接打飞。
#   - entropy_coef 从 0 升到 0.005，鼓励 a_gain 维度的早期探索。
#   - learning_rate 从 5e-4 略降到 3e-4，因为输出维度增加导致策略梯度方差略大。
@configclass
class EasyUUVParametricPPORunnerCfg(EasyUUVPPORunnerCfg):
    experiment_name = "easyuuv_parametric"
    policy = RslRlPpoActorCriticCfg(
        init_noise_std=0.8,
        actor_hidden_dims=[128, 128],
        critic_hidden_dims=[128, 128],
        activation="elu",
    )
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.005,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=3.0e-4,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )


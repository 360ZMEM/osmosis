"""@file eval/examples/thruster_io_demo.py
@brief 展示 4D 与 8D 动作布局，以及一个到物理推进器命令的参考映射。

动作契约：
  Baseline 策略 (`EasyUUV-Direct-v1`):       action ∈ R^4
      [u_surge, u_sway, u_heave, u_yaw]   每一维都在 [-1, 1]
  Parametric 策略 (`EasyUUV-Direct-Parametric-v1`): action ∈ R^8
      [u_surge, u_sway, u_heave, u_yaw,           # 控制意图（同 4D）
       a_gain_kp, a_gain_ki, a_gain_kd, a_gain_kff]  # 阻尼比调制
                                                    每一维都在 [-1, 1]（裁剪后）

4 个控制通道进入内环 PID；4 个 a_gain 通道通过 Bounded Safeguard 调制控制器阻尼比：
    ζ_i_eff = ζ_i_nom * (1 + β · a_gain_i)
默认 β = 0.25；该边界在保持闭环稳定的同时，允许策略在扰动下重整定增益。
"""

from __future__ import annotations

import argparse

import numpy as np

from easyuuv_stdw.eval import obs_from_state, Policy
from easyuuv_stdw.eval.deploy_config import DEFAULT_CONFIG_PATH, load_deploy_config
from easyuuv_stdw.eval.wrappers import ACT_DIM_BASELINE, ACT_DIM_PARAMETRIC


# 参考分配矩阵：由 [surge, sway, heave, yaw] 得到 4 个推进器力。
# TODO(deploy): 连接真实推进器前，必须替换为实物平台的 mixer 和符号约定。
THRUSTER_ALLOC = np.array([
    [ 1.0,  0.0, 0.0,  0.5],   # T0: 前右
    [ 1.0,  0.0, 0.0, -0.5],   # T1: 前左
    [ 0.0,  1.0, 1.0,  0.0],   # T2: 垂向 / 右侧横移
    [ 0.0, -1.0, 1.0,  0.0],   # T3: 垂向 / 左侧横移
], dtype=np.float32)


def to_thruster_cmds(action: np.ndarray) -> np.ndarray:
    """@brief 将策略的 4 通道控制意图映射为 4 个推进器力。

    @details 这里忽略 8D 参数化策略的 a_gain 通道，因为它们用于整定低层控制器，
    而不是直接进入推进器 mixer。
    """
    ctrl = np.clip(np.asarray(action, dtype=np.float32)[:4], -1.0, 1.0)   # 8D 中忽略 a_gain
    return THRUSTER_ALLOC @ ctrl


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default=str(DEFAULT_CONFIG_PATH), help="Deployment YAML config path")
    p.add_argument("--policy", default=None, help="Override policy path from deploy_config.yaml")
    p.add_argument("--obs_layout", default=None, choices=["a3_12d", "legacy_10d"],
                   help="Override obs_layout from deploy_config.yaml")
    args = p.parse_args()
    cfg = load_deploy_config(args.config)
    policy_path = args.policy or cfg.policy.model_path
    obs_layout = args.obs_layout or cfg.policy.obs_layout

    # TODO(deploy): 将合成 state 替换为由 IMU/深度/目标消息生成的硬件 bridge state。
    state = {
        "position":           np.zeros(3, dtype=np.float32),
        "orientation_quat":   np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32),
        "linear_velocity_b":  np.zeros(3, dtype=np.float32),
        "angular_velocity_b": np.zeros(3, dtype=np.float32),
        "goal_position":      np.array([0.5, 0.0, -1.0], dtype=np.float32),
        "goal_yaw":           0.0,
    }

    pol = Policy(policy_path, device=cfg.policy.device)
    action = pol.act(obs_from_state(state, layout=obs_layout))

    if action.shape[0] == ACT_DIM_BASELINE:
        print(f"baseline 4-D action: {action.tolist()}")
    elif action.shape[0] == ACT_DIM_PARAMETRIC:
        print(f"parametric 8-D action:")
        print(f"  control intent : {action[:4].tolist()}")
        print(f"  a_gain channels: {action[4:].tolist()}")
    else:
        raise SystemExit(f"unexpected action dim {action.shape[0]}")

    cmds = to_thruster_cmds(action)
    print(f"config={args.config}, obs_layout={obs_layout}, policy={policy_path}")
    print(f"thruster commands (4 thrusters, normalised): {cmds.tolist()}")


if __name__ == "__main__":
    main()

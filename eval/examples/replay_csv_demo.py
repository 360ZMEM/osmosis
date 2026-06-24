"""@file eval/examples/replay_csv_demo.py
@brief 最小端到端 demo：加载 Policy，从手写 state dict 构造一个观测，并打印动作。
"""

from __future__ import annotations

import argparse

import numpy as np

from easyuuv_stdw.eval import obs_from_state, Policy
from easyuuv_stdw.eval.deploy_config import DEFAULT_CONFIG_PATH, load_deploy_config


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

    # TODO(deploy): 将这里的 state 替换为真实 replay CSV 的一行，
    # 或来自在线硬件 bridge 的状态；字段必须符合 eval/wrappers.py 契约。
    state = {
        "position":           np.array([0.0, 0.0, -1.0], dtype=np.float32),
        "orientation_quat":   np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32),
        "linear_velocity_b":  np.zeros(3, dtype=np.float32),
        "angular_velocity_b": np.zeros(3, dtype=np.float32),
        "goal_position":      np.array([1.0, 0.0, -1.0], dtype=np.float32),
        "goal_yaw":           0.0,
    }
    obs = obs_from_state(state, layout=obs_layout)
    print(f"obs shape={obs.shape}, dtype={obs.dtype}")
    print(f"obs values={obs.tolist()}")

    pol = Policy(policy_path, device=cfg.policy.device)
    action = pol.act(obs)
    print(f"config={args.config}, obs_layout={obs_layout}, policy={policy_path}")
    print(f"backend={pol.backend}, action shape={action.shape}")
    print(f"action={action.tolist()}")


if __name__ == "__main__":
    main()

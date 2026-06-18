"""@file eval/examples/replay_csv_demo.py
@brief Minimal end-to-end demo: load a Policy, build one observation from a
       hand-crafted state dict, print the action.
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

    # TODO(deploy): Replace this state with one row from a real replay CSV or
    # from the live hardware bridge. The keys are the eval/wrappers.py contract.
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

"""@file eval/examples/thruster_io_demo.py
@brief Show the 4-D vs 8-D action layout and a reference mapping to physical
       thruster commands.

Action contract:
  Baseline policy (`EasyUUV-Direct-v1`):       action ∈ R^4
      [u_surge, u_sway, u_heave, u_yaw]   each in [-1, 1]
  Parametric policy (`EasyUUV-Direct-Parametric-v1`): action ∈ R^8
      [u_surge, u_sway, u_heave, u_yaw,           # control intent (same as 4-D)
       a_gain_kp, a_gain_ki, a_gain_kd, a_gain_kff]  # damping-ratio modulation
                                                    each in [-1, 1] (clipped)

The 4 control channels feed the inner PID; the 4 a_gain channels modulate the
controller damping ratio via a Bounded Safeguard:
    ζ_i_eff = ζ_i_nom * (1 + β · a_gain_i)
with β = 0.25 by default; this keeps the closed loop stable while letting the
policy retune gains under disturbance.
"""

from __future__ import annotations

import argparse

import numpy as np

from easyuuv_stdw.eval import obs_from_state, Policy
from easyuuv_stdw.eval.deploy_config import DEFAULT_CONFIG_PATH, load_deploy_config
from easyuuv_stdw.eval.wrappers import ACT_DIM_BASELINE, ACT_DIM_PARAMETRIC


# Reference allocation matrix: 4 thruster forces from [surge, sway, heave, yaw].
# TODO(deploy): Replace this with your real platform's mixer and sign
# conventions before connecting real thrusters.
THRUSTER_ALLOC = np.array([
    [ 1.0,  0.0, 0.0,  0.5],   # T0: front-right
    [ 1.0,  0.0, 0.0, -0.5],   # T1: front-left
    [ 0.0,  1.0, 1.0,  0.0],   # T2: vertical / sway right
    [ 0.0, -1.0, 1.0,  0.0],   # T3: vertical / sway left
], dtype=np.float32)


def to_thruster_cmds(action: np.ndarray) -> np.ndarray:
    """@brief Map the policy 4-channel control intent to 4 thruster forces.

    @details The a_gain channels from the 8-D parametric policy are ignored
    here because they tune the low-level controller, not the thruster mixer.
    """
    ctrl = np.clip(np.asarray(action, dtype=np.float32)[:4], -1.0, 1.0)   # ignore a_gain in 8-D
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

    # TODO(deploy): Replace this synthetic state with the hardware bridge state
    # produced from IMU/depth/goal messages.
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

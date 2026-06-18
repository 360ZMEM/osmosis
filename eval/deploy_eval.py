"""@file eval/deploy_eval.py
@brief CLI deployment-evaluation harness.

Replays a state log through a Policy and writes per-step obs/action/reward
to CSV for offline analysis. Isaac-independent.

Inputs:
  --policy <path>   .pt / .jit / .onnx checkpoint loadable by Policy.
  --replay <path>   CSV with columns required by obs_from_state:
                       t, px,py,pz, qw,qx,qy,qz, vx,vy,vz, wx,wy,wz, gx,gy,gz, gyaw
  --output <path>   Output CSV with action_*/reward columns appended.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np

from .policy_loader import Policy
from .wrappers import obs_from_state, reward_from_state, ACT_DIM_BASELINE, ACT_DIM_PARAMETRIC

REQUIRED_COLS = (
    "t", "px", "py", "pz", "qw", "qx", "qy", "qz",
    "vx", "vy", "vz", "wx", "wy", "wz",
    "gx", "gy", "gz", "gyaw",
)


def _row_to_state(row):
    return {
        "position":           np.array([row["px"], row["py"], row["pz"]], dtype=np.float32),
        "orientation_quat":   np.array([row["qw"], row["qx"], row["qy"], row["qz"]], dtype=np.float32),
        "linear_velocity_b":  np.array([row["vx"], row["vy"], row["vz"]], dtype=np.float32),
        "angular_velocity_b": np.array([row["wx"], row["wy"], row["wz"]], dtype=np.float32),
        "goal_position":      np.array([row["gx"], row["gy"], row["gz"]], dtype=np.float32),
        "goal_yaw":           float(row["gyaw"]),
    }


def run(policy_path: str, replay_path: str, output_path: str, *, obs_layout: str = "a3_12d") -> dict:
    pol = Policy(policy_path)

    rep = Path(replay_path)
    if not rep.is_file():
        raise FileNotFoundError(rep)

    with rep.open("r", newline="") as f:
        reader = csv.DictReader(f)
        missing = [c for c in REQUIRED_COLS if c not in reader.fieldnames]
        if missing:
            raise ValueError(f"replay CSV missing columns: {missing}")
        rows = [{k: float(v) for k, v in r.items()} for r in reader]

    if not rows:
        raise ValueError("replay CSV is empty")

    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    pos_err_sq_sum = 0.0
    reward_sum = 0.0
    n = 0

    first_action = pol.act(obs_from_state(_row_to_state(rows[0]), layout=obs_layout))
    act_dim = first_action.shape[0]
    if act_dim not in (ACT_DIM_BASELINE, ACT_DIM_PARAMETRIC):
        raise ValueError(f"unexpected action dim {act_dim} (expected 4 or 8)")

    fieldnames = list(REQUIRED_COLS) + [f"a{i}" for i in range(act_dim)] + ["reward", "pos_err_norm"]
    with out_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            state = _row_to_state(row)
            obs = obs_from_state(state, layout=obs_layout)
            action = pol.act(obs)
            r = reward_from_state(state, action)
            err = state["goal_position"] - state["position"]
            pe = float(np.linalg.norm(err))

            out_row = dict(row)
            for i in range(act_dim):
                out_row[f"a{i}"] = float(action[i])
            out_row["reward"] = r
            out_row["pos_err_norm"] = pe
            writer.writerow(out_row)

            pos_err_sq_sum += pe * pe
            reward_sum += r
            n += 1

    fmse = pos_err_sq_sum / n
    summary = {
        "policy": str(Path(policy_path).name),
        "backend": pol.backend,
        "n_steps": n,
        "fmse_pos_m2": fmse,
        "rmse_pos_m": float(np.sqrt(fmse)),
        "mean_reward": reward_sum / n,
        "output_csv": str(out_path),
        "action_dim": act_dim,
        "obs_layout": str(obs_layout),
    }
    print(summary)
    return summary


def _main():
    p = argparse.ArgumentParser(description="Replay-based policy evaluation.")
    p.add_argument("--policy", required=True)
    p.add_argument("--replay", required=True)
    p.add_argument("--output", default="./eval_out.csv")
    p.add_argument(
        "--obs_layout",
        default="a3_12d",
        choices=["a3_12d", "legacy_10d"],
        help="Observation contract. Use a3_12d for current parametric STDW deployment JIT.",
    )
    args = p.parse_args()
    run(args.policy, args.replay, args.output, obs_layout=args.obs_layout)


if __name__ == "__main__":
    _main()

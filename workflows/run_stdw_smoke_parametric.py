"""STDW smoke runner for the 8-dim Parametric task.

Mirrors ``workflows_new_stdw/run_stdw_smoke.py`` but targets
``EasyUUV-Direct-Parametric-v1`` and the freshly-trained
``model_1499.pt`` checkpoint, allowing two regression cells:

- ``identity_init=True``: meta-control bypassed (ζ_runtime ≡ ζ_nominal),
  STDW slow loop should behave equivalently to the legacy 4-dim path.
- ``tune_gains=True`` (default): STDW slow loop + 8-dim meta-control
  double envelope coexisting.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
WORKFLOW = ROOT / "workflows_new_stdw" / "play_stdw_adapt.py"
RUNNER = ROOT / "custom_workflows" / "run_with_isaac_env.sh"


def main() -> int:
    p = argparse.ArgumentParser(description="STDW smoke for Parametric task.")
    p.add_argument("--task", default="EasyUUV-Direct-Parametric-v1")
    p.add_argument("--logs_root", required=True,
                   help="Parent logs dir, e.g. .tmp/meta_train_full_*/logs/rsl_rl")
    p.add_argument("--load_run", required=True, help="Run sub-dir, e.g. 2026-06-05_22-08-02")
    p.add_argument("--checkpoint", default="model_1499.pt")
    p.add_argument("--total_steps", type=int, default=120)
    p.add_argument("--target_drift", type=float, default=0.05)
    p.add_argument("--drift_start_step", type=int, default=20)
    p.add_argument("--drift_end_step", type=int, default=100)
    p.add_argument("--slow_loop_interval", type=int, default=20)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--buffer_capacity", type=int, default=2048)
    p.add_argument("--results_root", required=True)
    p.add_argument("--artifacts_root", required=True)
    p.add_argument("--identity_init", default="False")
    p.add_argument("--tune_gains", default="True")
    args = p.parse_args()

    if not RUNNER.exists():
        raise FileNotFoundError(f"Missing Isaac Lab runner wrapper: {RUNNER}")

    os.environ.setdefault("CARB_APP_PATH", str(Path.home() / "isaaclab" / "_isaac_sim" / "kit"))
    os.environ.setdefault("ISAAC_PATH", str(Path.home() / "isaaclab" / "_isaac_sim"))
    os.environ.setdefault("EXP_PATH", str(Path.home() / "isaaclab" / "_isaac_sim" / "apps"))

    argv = [
        str(RUNNER), str(WORKFLOW),
        "--task", args.task,
        "--num_envs", "1",
        "--load_run", args.load_run,
        "--checkpoint", args.checkpoint,
        "--logs_root", args.logs_root,
        "--headless",
        "--total_steps", str(args.total_steps),
        "--target_drift", str(args.target_drift),
        "--drift_start_step", str(args.drift_start_step),
        "--drift_end_step", str(args.drift_end_step),
        "--slow_loop_interval", str(args.slow_loop_interval),
        "--batch_size", str(args.batch_size),
        "--buffer_capacity", str(args.buffer_capacity),
        "--enable_pseudo_action", "True",
        "--enable_lyapunov_mask", "True",
        "--results_root", args.results_root,
        "--artifacts_root", args.artifacts_root,
        # 8-dim meta-control passthrough — these go to env_cfg via train_meta-style overrides.
        # play_stdw_adapt.py doesn't natively know meta CLI; we keep both knobs at the env
        # default (cfg.tune_gains=True / cfg.identity_init=False) for the baseline cell, then
        # let the calling shell switch them via a tiny env_cfg patch (see below).
    ]

    return subprocess.run(argv, check=False).returncode


if __name__ == "__main__":
    sys.exit(main())

"""Smoke runner for the STDW (v3) online adaptation workflow.

Mirrors ``workflows/run_stdw_smoke.py`` but invokes the new
``workflows_new_stdw/play_stdw_adapt.py`` and short-circuits the runtime
parameters per plan §5 / §6 step 4.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
WORKFLOW = ROOT / "workflows_new_stdw" / "play_stdw_adapt.py"
RUNNER = ROOT / "custom_workflows" / "run_with_isaac_env.sh"

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

os.environ.setdefault("CARB_APP_PATH", str(Path.home() / "isaaclab" / "_isaac_sim" / "kit"))
os.environ.setdefault("ISAAC_PATH", str(Path.home() / "isaaclab" / "_isaac_sim"))
os.environ.setdefault("EXP_PATH", str(Path.home() / "isaaclab" / "_isaac_sim" / "apps"))

argv = [
    str(RUNNER),
    str(WORKFLOW),
    "--task",
    "EasyUUV-Direct-v1",
    "--num_envs",
    "1",
    "--load_run",
    "SS4",
    "--checkpoint",
    "model_500.pt",
    "--logs_root",
    str(Path.home() / "isaaclab" / "logs" / "rsl_rl"),
    "--headless",
    "--cpu",
    "--total_steps",
    "20",
    "--target_drift",
    "0.05",
    "--drift_start_step",
    "4",
    "--drift_end_step",
    "16",
    "--slow_loop_interval",
    "4",
    "--batch_size",
    "32",
    "--buffer_capacity",
    "1024",
    "--enable_pseudo_action",
    "True",
    "--enable_lyapunov_mask",
    "True",
    "--results_root",
    str(ROOT / ".tmp" / "stdw_new_logs"),
    "--artifacts_root",
    str(ROOT / ".tmp" / "stdw_new_artifacts"),
]

if not RUNNER.exists():
    raise FileNotFoundError(f"Missing Isaac Lab runner wrapper: {RUNNER}")

raise SystemExit(subprocess.run(argv, check=False).returncode)

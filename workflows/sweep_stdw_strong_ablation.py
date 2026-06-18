"""Default ablation diagnostics for asymmetric linear strong-validation failures."""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parent.parent
RUNNER = REPO_ROOT / "custom_workflows" / "run_with_isaac_env.sh"
PLAY = REPO_ROOT / "workflows" / "play_stdw_adapt.py"


def _bool(v: bool) -> str:
    return "True" if v else "False"


DISTURBANCES: list[dict[str, Any]] = [
    {
        "base_case": "cob_linear_asymmetric",
        "group": "cob",
        "scenario": "cob_shift_x",
        "extra": ["--target_drift", "0.05", "--drift_axes", "0"],
    },
    {
        "base_case": "thruster_single_ramp_to_target_asymmetric",
        "group": "thruster_single",
        "scenario": "thruster_single_fault",
        "extra": [
            "--fault_thrusters", "4",
            "--fault_profile", "ramp_to_target",
            "--fault_target_efficiency", "0.5",
            "--fault_ramp_duration_s", "5.0",
        ],
    },
    {
        "base_case": "density095_linear_asymmetric",
        "group": "density",
        "scenario": "density_095",
        "extra": ["--water_density_scale_target", "0.95"],
    },
    {
        "base_case": "torque_l0p5_asymmetric",
        "group": "torque_pulse",
        "scenario": "torque_pulse_medium",
        "extra": ["--torque_pulse_level", "0.5"],
    },
    {
        "base_case": "torque_l1p0_asymmetric",
        "group": "torque_pulse",
        "scenario": "torque_pulse_strong",
        "extra": ["--torque_pulse_level", "1.0"],
    },
]


ABLATIONS: list[dict[str, Any]] = [
    {"name": "no_slow_loop", "extra": ["--disable_slow_loop", "True"]},
    {"name": "no_lyapunov", "extra": ["--enable_lyapunov_mask", "False"]},
    {"name": "no_pseudo", "extra": ["--enable_pseudo_action", "False"]},
    {"name": "no_trigger_gate", "extra": ["--enable_trigger_gate", "False"]},
    {"name": "no_quantile_filter", "extra": ["--use_quantile_filter", "False"]},
]


def build_cases() -> list[dict[str, Any]]:
    cases: list[dict[str, Any]] = []
    for disturbance in DISTURBANCES:
        for ablation in ABLATIONS:
            name = f"{disturbance['base_case']}_{ablation['name']}"
            cases.append({
                "name": name,
                "base_case": disturbance["base_case"],
                "group": disturbance["group"],
                "scenario": disturbance["scenario"],
                "ramp_shape": "linear",
                "embodiment": "asymmetric",
                "use_stdw": True,
                "ablation": ablation["name"],
                "extra": [*disturbance["extra"], *ablation["extra"]],
            })
    return cases


def build_cmd(case: dict[str, Any], args: argparse.Namespace, case_dir: Path) -> list[str]:
    cmd = [
        "bash", str(RUNNER), str(PLAY),
        "--headless",
        "--task", "EasyUUV-Direct-Parametric-v1",
        "--num_envs", "1",
        "--experiment_name", "easyuuv_parametric",
        "--logs_root", "logs/rsl_rl",
        "--load_run", "2026-06-08_13-48-14_stage2",
        "--checkpoint", "model_2398.pt",
        "--workflow_config", "workflows/configs/matrix_wave_medium_full.yaml",
        "--total_steps", str(args.total_steps),
        "--seed", str(args.seed),
        "--scenario", str(case["scenario"]),
        "--embodiment", str(case["embodiment"]),
        "--use_stdw", _bool(bool(case["use_stdw"])),
        "--ramp_shape", str(case["ramp_shape"]),
        "--results_root", str(case_dir / "results"),
        "--artifacts_root", str(case_dir / "artifacts"),
    ]
    cmd.extend(case["extra"])
    return cmd


def latest_summary(case_dir: Path) -> tuple[str, dict[str, Any]]:
    summaries = sorted((case_dir / "results").glob("**/summary.json"), key=lambda p: p.stat().st_mtime)
    if not summaries:
        return "", {}
    path = summaries[-1]
    return str(path), json.loads(path.read_text(encoding="utf-8"))


def main() -> int:
    parser = argparse.ArgumentParser(description="Run default asymmetric-linear STDW ablation diagnostics.")
    parser.add_argument("--results_root", default=".results/exp_stdw_strong_ablation_20260613")
    parser.add_argument("--total_steps", type=int, default=1500)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--dry_run", action="store_true", default=False)
    parser.add_argument("--run", action="store_true", default=False)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    cases = build_cases()
    if args.limit is not None:
        cases = cases[: int(args.limit)]
    root = Path(args.results_root)
    root.mkdir(parents=True, exist_ok=True)

    rows = []
    for idx, case in enumerate(cases, 1):
        case_dir = root / case["name"]
        cmd = build_cmd(case, args, case_dir)
        print(f"[{idx:02d}/{len(cases):02d}] {case['name']}")
        print(" ".join(cmd))
        started = datetime.now().isoformat(timespec="seconds")
        rc = 0
        if args.run and not args.dry_run:
            case_dir.mkdir(parents=True, exist_ok=True)
            with (case_dir / "run.log").open("w", encoding="utf-8") as f:
                rc = subprocess.run(cmd, cwd=REPO_ROOT, stdout=f, stderr=subprocess.STDOUT).returncode
        ended = datetime.now().isoformat(timespec="seconds")
        summary_path, summary = latest_summary(case_dir)
        rows.append({
            "case": case["name"],
            "base_case": case["base_case"],
            "group": case["group"],
            "scenario": case["scenario"],
            "ramp_shape": case["ramp_shape"],
            "embodiment": case["embodiment"],
            "use_stdw": case["use_stdw"],
            "ablation": case["ablation"],
            "returncode": rc,
            "started_at": started,
            "ended_at": ended,
            "summary_path": summary_path,
            "final_mse": summary.get("final_mse"),
            "final_mse_after_drift": summary.get("final_mse_after_drift"),
            "slow_loop_triggers": summary.get("slow_loop_triggers"),
            "reset_count": summary.get("reset_count"),
            "nonfinite_guard_count": summary.get("nonfinite_guard_count"),
            "mean_total_mse": summary.get("mean_total_mse"),
            "mean_pitch_mse": summary.get("mean_pitch_mse"),
            "mean_depth_mse": summary.get("mean_depth_mse"),
        })
        if rc != 0:
            print(f"[ERROR] case failed: {case['name']} rc={rc}")
            break

    out = root / "strong_ablation_runs.csv"
    with out.open("w", newline="", encoding="utf-8") as f:
        fieldnames = list(rows[0].keys()) if rows else ["case"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"[INFO] wrote {out} rows={len(rows)}")
    return 0 if all(int(r["returncode"]) == 0 for r in rows) else 1


if __name__ == "__main__":
    sys.exit(main())

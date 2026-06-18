"""Dynamic parameter-family identification sweep.

Runs the approved 12-cell academic probe matrix:
density / thruster efficiency / thruster angle / torque negative control
times full / off / param_probe.
"""

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


def _bool(value: bool) -> str:
    return "True" if value else "False"


SCENARIOS: list[dict[str, Any]] = [
    {
        "family": "density",
        "scenario": "density_095",
        "extra": ["--water_density_scale_target", "0.95"],
    },
    {
        "family": "thruster_efficiency",
        "scenario": "thruster_single_fault",
        "extra": [
            "--fault_thrusters", "4",
            "--fault_profile", "ramp_to_target",
            "--fault_target_efficiency", "0.5",
            "--fault_ramp_duration_s", "5.0",
        ],
    },
    {
        "family": "thruster_angle",
        "scenario": "thruster_angle_yaw_p5deg",
        "extra": [],
    },
    {
        "family": "torque_negative_control",
        "scenario": "torque_pulse_medium",
        "extra": ["--torque_pulse_level", "0.5"],
    },
]


def build_cases() -> list[dict[str, Any]]:
    cases: list[dict[str, Any]] = []
    modes = [
        ("full", True, []),
        ("off", False, []),
        (
            "param_probe",
            True,
            [
                "--enable_param_probe", "True",
                "--param_probe_apply_result", "True",
                "--param_probe_start_step", "320",
                "--param_probe_window_steps", "240",
                "--param_probe_families", "density,thruster_efficiency,thruster_angle",
                "--param_probe_active_excitation", "True",
                "--param_probe_excitation_magnitude", "0.02",
                "--param_probe_excitation_channels", "0,1,2,3",
                "--param_probe_excitation_period_steps", "20",
            ],
        ),
    ]
    for spec in SCENARIOS:
        for mode, use_stdw, mode_extra in modes:
            cases.append({
                "name": f"{spec['family']}_linear_asymmetric_{mode}",
                "family": spec["family"],
                "scenario": spec["scenario"],
                "ramp_shape": "linear",
                "embodiment": "asymmetric",
                "mode": mode,
                "use_stdw": use_stdw,
                "extra": [*spec["extra"], *mode_extra],
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
        "--drift_start_step", "320",
        "--drift_end_step", "1200",
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
    parser = argparse.ArgumentParser(description="Run dynamic parameter-ID probe matrix.")
    parser.add_argument("--profile", default="small", choices=["small"])
    parser.add_argument("--results_root", default=".results/exp_dynamic_param_id_20260613")
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

    rows: list[dict[str, Any]] = []
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
            "family": case["family"],
            "scenario": case["scenario"],
            "ramp_shape": case["ramp_shape"],
            "embodiment": case["embodiment"],
            "mode": case["mode"],
            "use_stdw": case["use_stdw"],
            "returncode": rc,
            "started_at": started,
            "ended_at": ended,
            "summary_path": summary_path,
            "final_mse": summary.get("final_mse"),
            "final_mse_after_drift": summary.get("final_mse_after_drift"),
            "slow_loop_triggers": summary.get("slow_loop_triggers"),
            "gate_silenced_count": summary.get("gate_silenced_count"),
            "param_probe_selected_family": summary.get("param_probe_selected_family"),
            "param_probe_active_excitation": summary.get("param_probe_active_excitation"),
            "param_probe_scores": json.dumps(summary.get("param_probe_scores", {}), ensure_ascii=False),
            "param_probe_features": json.dumps(summary.get("param_probe_features", {}), ensure_ascii=False),
            "param_probe_reason": summary.get("param_probe_reason"),
            "param_probe_gate_slow_loop": summary.get("param_probe_gate_slow_loop"),
            "param_probe_gate_step": summary.get("param_probe_gate_step"),
        })
        if rc != 0:
            print(f"[ERROR] case failed: {case['name']} rc={rc}")
            break

    out = root / "dynamic_param_id_runs.csv"
    with out.open("w", newline="", encoding="utf-8") as f:
        fieldnames = list(rows[0].keys()) if rows else ["case"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"[INFO] wrote {out} rows={len(rows)}")
    return 0 if all(int(row["returncode"]) == 0 for row in rows) else 1


if __name__ == "__main__":
    sys.exit(main())

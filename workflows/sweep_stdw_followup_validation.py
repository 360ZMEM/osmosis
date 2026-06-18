"""Follow-up validation for router/probe and density no-quantile stability."""

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


ASYM_LINEAR_DISTURBANCES: list[dict[str, Any]] = [
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


def build_cases() -> list[dict[str, Any]]:
    cases: list[dict[str, Any]] = []

    matched_timing_extra = ["--drift_start_step", "320", "--drift_end_step", "1200"]
    router_extra = ["--auto_drift_router", "True", "--drift_router_mode", "offset_correct"]
    probe_extra = [
        "--enable_micro_probe", "True",
        "--micro_probe_apply_result", "True",
        "--micro_probe_start_step", "40",
        "--micro_probe_window_steps", "20",
        "--micro_probe_settle_steps", "5",
        "--micro_probe_axes", "0,1",
        "--micro_probe_score_mode", "paired_axis",
        "--micro_probe_baseline_each_candidate", "True",
        "--drift_start_step", "320",
        "--drift_end_step", "1200",
    ]
    for disturbance in ASYM_LINEAR_DISTURBANCES:
        for mode, extra in [
            ("matched_full", matched_timing_extra),
            ("matched_off", matched_timing_extra),
            ("router", [*matched_timing_extra, *router_extra]),
            ("micro_probe", probe_extra),
        ]:
            cases.append({
                "name": f"{disturbance['base_case']}_{mode}",
                "suite": "router_probe",
                "base_case": disturbance["base_case"],
                "group": disturbance["group"],
                "scenario": disturbance["scenario"],
                "ramp_shape": "linear",
                "embodiment": "asymmetric",
                "use_stdw": mode != "matched_off",
                "variant": mode,
                "seed": 0,
                "extra": [*disturbance["extra"], *extra],
            })

    density_extra = ["--water_density_scale_target", "0.95"]
    for seed in [1, 2]:
        for variant, stdw, extra in [
            ("full", True, []),
            ("off", False, []),
            ("no_quantile_filter", True, ["--use_quantile_filter", "False"]),
        ]:
            cases.append({
                "name": f"density095_linear_asymmetric_seed{seed}_{variant}",
                "suite": "density_no_quantile_seeds",
                "base_case": "density095_linear_asymmetric",
                "group": "density",
                "scenario": "density_095",
                "ramp_shape": "linear",
                "embodiment": "asymmetric",
                "use_stdw": stdw,
                "variant": variant,
                "seed": seed,
                "extra": [*density_extra, *extra],
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
        "--seed", str(case["seed"]),
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
    parser = argparse.ArgumentParser(description="Run follow-up STDW validation cases.")
    parser.add_argument("--results_root", default=".results/exp_stdw_followup_validation_20260613")
    parser.add_argument("--total_steps", type=int, default=1500)
    parser.add_argument("--dry_run", action="store_true", default=False)
    parser.add_argument("--run", action="store_true", default=False)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--variant_filter", default="", help="Optional comma-separated variants to run.")
    args = parser.parse_args()

    cases = build_cases()
    if args.variant_filter:
        keep = {item.strip() for item in str(args.variant_filter).split(",") if item.strip()}
        cases = [case for case in cases if str(case["variant"]) in keep]
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
            "suite": case["suite"],
            "base_case": case["base_case"],
            "group": case["group"],
            "scenario": case["scenario"],
            "ramp_shape": case["ramp_shape"],
            "embodiment": case["embodiment"],
            "use_stdw": case["use_stdw"],
            "variant": case["variant"],
            "seed": case["seed"],
            "returncode": rc,
            "started_at": started,
            "ended_at": ended,
            "summary_path": summary_path,
            "final_mse": summary.get("final_mse"),
            "final_mse_after_drift": summary.get("final_mse_after_drift"),
            "slow_loop_triggers": summary.get("slow_loop_triggers"),
            "reset_count": summary.get("reset_count"),
            "nonfinite_guard_count": summary.get("nonfinite_guard_count"),
            "micro_probe_selected_name": summary.get("micro_probe_selected_name"),
            "micro_probe_selected_target": summary.get("micro_probe_selected_target"),
            "micro_probe_selected_reason": summary.get("micro_probe_selected_reason"),
            "auto_drift_router": summary.get("auto_drift_router"),
            "target_drift": summary.get("target_drift"),
            "drift_axes": json.dumps(summary.get("drift_axes", []), ensure_ascii=False),
        })
        if rc != 0:
            print(f"[ERROR] case failed: {case['name']} rc={rc}")
            break

    out = root / "followup_validation_runs.csv"
    with out.open("w", newline="", encoding="utf-8") as f:
        fieldnames = list(rows[0].keys()) if rows else ["case"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"[INFO] wrote {out} rows={len(rows)}")
    return 0 if all(int(r["returncode"]) == 0 for r in rows) else 1


if __name__ == "__main__":
    sys.exit(main())

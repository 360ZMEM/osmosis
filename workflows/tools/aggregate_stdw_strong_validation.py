"""Aggregate STDW strong-validation sweep results."""

from __future__ import annotations

import argparse
import csv
import json
import statistics
from pathlib import Path
from typing import Any


FIELDS = [
    "case", "group", "scenario", "ramp_shape", "embodiment", "use_stdw",
    "returncode", "final_mse", "final_mse_after_drift", "mean_total_mse",
    "mean_roll_mse", "mean_pitch_mse", "mean_yaw_mse", "mean_depth_mse",
    "reset_count", "nonfinite_guard_count", "slow_loop_triggers",
    "gate_silenced_count", "fault_efficiency_min", "water_density_scale",
    "torque_pulse_level", "control_effort_mean", "control_effort_max",
    "summary_path",
]


def _float(value: Any, default: float = float("nan")) -> float:
    try:
        if value in ("", None):
            return default
        return float(value)
    except Exception:
        return default


def _read_csv_stats(csv_path: str) -> dict[str, Any]:
    out = {
        "fault_efficiency_min": "",
        "water_density_scale": "",
        "torque_pulse_level": "",
        "control_effort_mean": "",
        "control_effort_max": "",
    }
    if not csv_path or not Path(csv_path).is_file():
        return out
    rows = list(csv.DictReader(Path(csv_path).open(encoding="utf-8")))
    if not rows:
        return out
    for key in ("fault_efficiency_min", "water_density_scale", "torque_pulse_level"):
        vals = [_float(r.get(key)) for r in rows if r.get(key) not in ("", None)]
        if vals:
            out[key] = min(vals) if key == "fault_efficiency_min" else vals[-1]
    efforts = [_float(r.get("control_effort")) for r in rows if r.get("control_effort") not in ("", None)]
    efforts = [v for v in efforts if v == v]
    if efforts:
        out["control_effort_mean"] = statistics.mean(efforts)
        out["control_effort_max"] = max(efforts)
    return out


def collect(root: Path) -> list[dict[str, Any]]:
    run_csv = root / "strong_validation_runs.csv"
    base_rows = list(csv.DictReader(run_csv.open(encoding="utf-8"))) if run_csv.is_file() else []
    if not base_rows:
        base_rows = [{"case": p.parent.parent.parent.name, "summary_path": str(p)} for p in root.glob("*/results/**/summary.json")]
    rows = []
    for base in base_rows:
        summary_path = base.get("summary_path", "")
        summary = {}
        if summary_path and Path(summary_path).is_file():
            summary = json.loads(Path(summary_path).read_text(encoding="utf-8"))
        csv_stats = _read_csv_stats(summary.get("csv_path", ""))
        row = {
            "case": base.get("case", ""),
            "group": base.get("group", ""),
            "scenario": summary.get("scenario", base.get("scenario", "")),
            "ramp_shape": summary.get("ramp_shape", base.get("ramp_shape", "")),
            "embodiment": summary.get("embodiment", base.get("embodiment", "")),
            "use_stdw": summary.get("use_stdw", base.get("use_stdw", "")),
            "returncode": base.get("returncode", ""),
            "final_mse": summary.get("final_mse", base.get("final_mse", "")),
            "final_mse_after_drift": summary.get("final_mse_after_drift", base.get("final_mse_after_drift", "")),
            "mean_total_mse": summary.get("mean_total_mse", base.get("mean_total_mse", "")),
            "mean_roll_mse": summary.get("mean_roll_mse", base.get("mean_roll_mse", "")),
            "mean_pitch_mse": summary.get("mean_pitch_mse", base.get("mean_pitch_mse", "")),
            "mean_yaw_mse": summary.get("mean_yaw_mse", base.get("mean_yaw_mse", "")),
            "mean_depth_mse": summary.get("mean_depth_mse", base.get("mean_depth_mse", "")),
            "reset_count": summary.get("reset_count", base.get("reset_count", "")),
            "nonfinite_guard_count": summary.get("nonfinite_guard_count", base.get("nonfinite_guard_count", "")),
            "slow_loop_triggers": summary.get("slow_loop_triggers", base.get("slow_loop_triggers", "")),
            "gate_silenced_count": summary.get("gate_silenced_count", ""),
            "fault_efficiency_min": csv_stats["fault_efficiency_min"],
            "water_density_scale": csv_stats["water_density_scale"],
            "torque_pulse_level": csv_stats["torque_pulse_level"],
            "control_effort_mean": csv_stats["control_effort_mean"],
            "control_effort_max": csv_stats["control_effort_max"],
            "summary_path": summary_path,
        }
        rows.append(row)
    return rows


def write_report(root: Path, rows: list[dict[str, Any]]) -> Path:
    report = root / "strong_validation_report.md"
    lines = [
        "# STDW 强验证小矩阵报告",
        "",
        f"- rows: {len(rows)}",
        f"- summary csv: `{root / 'strong_validation_summary.csv'}`",
        "",
        "## 主表",
        "",
        "| case | scenario | ramp | embodiment | STDW | final_mse | after_drift | reset | nonfinite |",
        "|---|---|---|---|---:|---:|---:|---:|---:|",
    ]
    for r in rows:
        lines.append(
            f"| {r['case']} | {r['scenario']} | {r['ramp_shape']} | {r['embodiment']} | "
            f"{r['use_stdw']} | {r['final_mse']} | {r['final_mse_after_drift']} | "
            f"{r['reset_count']} | {r['nonfinite_guard_count']} |"
        )
    lines += [
        "",
        "## 人工消融建议",
        "",
        "先比较同一 scenario/ramp/embodiment 下 STDW on/off。若 STDW on 明显劣于 off，或 reset/nonfinite 增多，再选择追加 no_slow_loop、no_lyapunov、no_pseudo、no_trigger_gate、no_quantile_filter。",
    ]
    report.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=".results/exp_stdw_strong_validation_20260613")
    args = parser.parse_args()
    root = Path(args.root)
    rows = collect(root)
    out = root / "strong_validation_summary.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    report = write_report(root, rows)
    print(f"[INFO] wrote {out} rows={len(rows)}")
    print(f"[INFO] wrote {report}")


if __name__ == "__main__":
    main()

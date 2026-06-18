"""Aggregate default asymmetric-linear STDW ablation diagnostics."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


FIELDS = [
    "case", "base_case", "group", "scenario", "ramp_shape", "embodiment",
    "ablation", "returncode", "final_mse", "final_mse_after_drift",
    "mean_total_mse", "mean_pitch_mse", "mean_depth_mse", "reset_count",
    "nonfinite_guard_count", "slow_loop_triggers", "full_final_mse",
    "off_final_mse", "delta_vs_full_pct", "delta_vs_off_pct", "summary_path",
]


def _float(value: Any, default: float = float("nan")) -> float:
    try:
        if value in ("", None):
            return default
        return float(value)
    except Exception:
        return default


def _load_reference(path: Path) -> dict[str, dict[str, float]]:
    refs: dict[str, dict[str, float]] = {}
    if not path.is_file():
        return refs
    for row in csv.DictReader(path.open(encoding="utf-8")):
        case = row.get("case", "")
        if not case:
            continue
        if case.endswith("_stdw1"):
            refs.setdefault(case[:-6], {})["full"] = _float(row.get("final_mse"))
        elif case.endswith("_stdw0"):
            refs.setdefault(case[:-6], {})["off"] = _float(row.get("final_mse"))
    return refs


def collect(root: Path, reference_csv: Path) -> list[dict[str, Any]]:
    run_csv = root / "strong_ablation_runs.csv"
    base_rows = list(csv.DictReader(run_csv.open(encoding="utf-8"))) if run_csv.is_file() else []
    refs = _load_reference(reference_csv)
    rows = []
    for base in base_rows:
        summary_path = base.get("summary_path", "")
        summary = {}
        if summary_path and Path(summary_path).is_file():
            summary = json.loads(Path(summary_path).read_text(encoding="utf-8"))
        final_mse = _float(summary.get("final_mse", base.get("final_mse")))
        base_case = base.get("base_case", "")
        full = refs.get(base_case, {}).get("full", float("nan"))
        off = refs.get(base_case, {}).get("off", float("nan"))
        row = {
            "case": base.get("case", ""),
            "base_case": base_case,
            "group": base.get("group", ""),
            "scenario": summary.get("scenario", base.get("scenario", "")),
            "ramp_shape": summary.get("ramp_shape", base.get("ramp_shape", "")),
            "embodiment": summary.get("embodiment", base.get("embodiment", "")),
            "ablation": base.get("ablation", ""),
            "returncode": base.get("returncode", ""),
            "final_mse": final_mse,
            "final_mse_after_drift": summary.get("final_mse_after_drift", base.get("final_mse_after_drift", "")),
            "mean_total_mse": summary.get("mean_total_mse", base.get("mean_total_mse", "")),
            "mean_pitch_mse": summary.get("mean_pitch_mse", base.get("mean_pitch_mse", "")),
            "mean_depth_mse": summary.get("mean_depth_mse", base.get("mean_depth_mse", "")),
            "reset_count": summary.get("reset_count", base.get("reset_count", "")),
            "nonfinite_guard_count": summary.get("nonfinite_guard_count", base.get("nonfinite_guard_count", "")),
            "slow_loop_triggers": summary.get("slow_loop_triggers", base.get("slow_loop_triggers", "")),
            "full_final_mse": full,
            "off_final_mse": off,
            "delta_vs_full_pct": (final_mse / full - 1.0) * 100.0 if full == full and full else "",
            "delta_vs_off_pct": (final_mse / off - 1.0) * 100.0 if off == off and off else "",
            "summary_path": summary_path,
        }
        rows.append(row)
    return rows


def write_report(root: Path, rows: list[dict[str, Any]]) -> Path:
    report = root / "strong_ablation_report.md"
    lines = [
        "# STDW asymmetric-linear 消融诊断报告",
        "",
        f"- rows: {len(rows)}",
        f"- summary csv: `{root / 'strong_ablation_summary.csv'}`",
        "",
        "## 主表",
        "",
        "| base_case | ablation | final_mse | full | off | vs_full | vs_off | triggers | nonfinite |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for r in rows:
        lines.append(
            f"| {r['base_case']} | {r['ablation']} | {r['final_mse']} | "
            f"{r['full_final_mse']} | {r['off_final_mse']} | {r['delta_vs_full_pct']} | "
            f"{r['delta_vs_off_pct']} | {r['slow_loop_triggers']} | {r['nonfinite_guard_count']} |"
        )
    lines += ["", "## 初步判断", ""]
    best_by_case: dict[str, dict[str, Any]] = {}
    for r in rows:
        key = str(r["base_case"])
        cur = best_by_case.get(key)
        if cur is None or _float(r["final_mse"]) < _float(cur["final_mse"]):
            best_by_case[key] = r
    for key, r in sorted(best_by_case.items()):
        lines.append(
            f"- `{key}` best ablation: `{r['ablation']}`, final_mse={r['final_mse']}, "
            f"vs_full={r['delta_vs_full_pct']}, vs_off={r['delta_vs_off_pct']}."
        )
    report.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=".results/exp_stdw_strong_ablation_20260613")
    parser.add_argument("--reference_csv", default=".results/exp_stdw_strong_validation_20260613/strong_validation_summary.csv")
    args = parser.parse_args()
    root = Path(args.root)
    rows = collect(root, Path(args.reference_csv))
    out = root / "strong_ablation_summary.csv"
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

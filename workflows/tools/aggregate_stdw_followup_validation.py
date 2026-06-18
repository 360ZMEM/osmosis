"""Aggregate follow-up STDW validation results."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


FIELDS = [
    "case", "suite", "base_case", "group", "scenario", "ramp_shape", "embodiment",
    "variant", "seed", "use_stdw", "returncode", "final_mse", "final_mse_after_drift",
    "slow_loop_triggers", "reset_count", "nonfinite_guard_count", "ref_full_mse",
    "ref_off_mse", "delta_vs_full_pct", "delta_vs_off_pct", "micro_probe_selected_name",
    "micro_probe_selected_target", "micro_probe_selected_reason", "target_drift",
    "drift_axes", "summary_path",
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
    run_csv = root / "followup_validation_runs.csv"
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
            "suite": base.get("suite", ""),
            "base_case": base_case,
            "group": base.get("group", ""),
            "scenario": summary.get("scenario", base.get("scenario", "")),
            "ramp_shape": summary.get("ramp_shape", base.get("ramp_shape", "")),
            "embodiment": summary.get("embodiment", base.get("embodiment", "")),
            "variant": base.get("variant", ""),
            "seed": base.get("seed", ""),
            "use_stdw": summary.get("use_stdw", base.get("use_stdw", "")),
            "returncode": base.get("returncode", ""),
            "final_mse": final_mse,
            "final_mse_after_drift": summary.get("final_mse_after_drift", base.get("final_mse_after_drift", "")),
            "slow_loop_triggers": summary.get("slow_loop_triggers", base.get("slow_loop_triggers", "")),
            "reset_count": summary.get("reset_count", base.get("reset_count", "")),
            "nonfinite_guard_count": summary.get("nonfinite_guard_count", base.get("nonfinite_guard_count", "")),
            "ref_full_mse": full,
            "ref_off_mse": off,
            "delta_vs_full_pct": (final_mse / full - 1.0) * 100.0 if full == full and full else "",
            "delta_vs_off_pct": (final_mse / off - 1.0) * 100.0 if off == off and off else "",
            "micro_probe_selected_name": summary.get("micro_probe_selected_name", base.get("micro_probe_selected_name", "")),
            "micro_probe_selected_target": summary.get("micro_probe_selected_target", base.get("micro_probe_selected_target", "")),
            "micro_probe_selected_reason": summary.get("micro_probe_selected_reason", base.get("micro_probe_selected_reason", "")),
            "target_drift": summary.get("target_drift", base.get("target_drift", "")),
            "drift_axes": json.dumps(summary.get("drift_axes", base.get("drift_axes", "")), ensure_ascii=False),
            "summary_path": summary_path,
        }
        rows.append(row)
    return rows


def write_report(root: Path, rows: list[dict[str, Any]]) -> Path:
    report = root / "followup_validation_report.md"
    lines = [
        "# STDW 追加验证报告",
        "",
        f"- rows: {len(rows)}",
        f"- summary csv: `{root / 'followup_validation_summary.csv'}`",
        "",
        "## Router / Micro-Probe",
        "",
        "| base_case | variant | final_mse | vs_full | vs_off | selected | target | drift_axes |",
        "|---|---|---:|---:|---:|---|---:|---|",
    ]
    for r in rows:
        if r["suite"] != "router_probe":
            continue
        lines.append(
            f"| {r['base_case']} | {r['variant']} | {r['final_mse']} | "
            f"{r['delta_vs_full_pct']} | {r['delta_vs_off_pct']} | "
            f"{r['micro_probe_selected_name']} | {r['target_drift']} | {r['drift_axes']} |"
        )
    lines += [
        "",
        "## Density no-quantile seeds",
        "",
        "| seed | variant | final_mse | triggers | nonfinite |",
        "|---:|---|---:|---:|---:|",
    ]
    for r in rows:
        if r["suite"] != "density_no_quantile_seeds":
            continue
        lines.append(
            f"| {r['seed']} | {r['variant']} | {r['final_mse']} | "
            f"{r['slow_loop_triggers']} | {r['nonfinite_guard_count']} |"
        )
    report.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=".results/exp_stdw_followup_validation_20260613")
    parser.add_argument("--reference_csv", default=".results/exp_stdw_strong_validation_20260613/strong_validation_summary.csv")
    args = parser.parse_args()
    root = Path(args.root)
    rows = collect(root, Path(args.reference_csv))
    out = root / "followup_validation_summary.csv"
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

"""Aggregate dynamic parameter-family identification sweep results."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


FIELDS = [
    "case", "family", "scenario", "ramp_shape", "embodiment", "mode", "use_stdw",
    "returncode", "final_mse", "final_mse_after_drift", "slow_loop_triggers",
    "gate_silenced_count", "delta_vs_full_pct", "delta_vs_off_pct",
    "param_probe_active_excitation", "param_probe_selected_family", "param_probe_reason", "param_probe_gate_slow_loop",
    "param_probe_gate_step", "param_probe_scores", "param_probe_features",
    "summary_path",
]


def _float(value: Any, default: float = float("nan")) -> float:
    try:
        if value in ("", None):
            return default
        return float(value)
    except Exception:
        return default


def _load_json_field(value: Any) -> Any:
    if isinstance(value, (dict, list)):
        return value
    try:
        return json.loads(str(value)) if value not in ("", None) else {}
    except Exception:
        return {}


def collect(root: Path) -> list[dict[str, Any]]:
    run_csv = root / "dynamic_param_id_runs.csv"
    base_rows = list(csv.DictReader(run_csv.open(encoding="utf-8"))) if run_csv.is_file() else []
    if not base_rows:
        base_rows = [{"case": p.parent.parent.parent.name, "summary_path": str(p)} for p in root.glob("*/results/**/summary.json")]

    refs: dict[str, dict[str, float]] = {}
    summaries: list[tuple[dict[str, Any], dict[str, Any]]] = []
    for base in base_rows:
        summary_path = base.get("summary_path", "")
        summary = {}
        if summary_path and Path(summary_path).is_file():
            summary = json.loads(Path(summary_path).read_text(encoding="utf-8"))
        summaries.append((base, summary))
        family = base.get("family", "")
        mode = base.get("mode", "")
        final_mse = _float(summary.get("final_mse", base.get("final_mse")))
        if family and mode in {"full", "off"}:
            refs.setdefault(family, {})[mode] = final_mse

    rows: list[dict[str, Any]] = []
    for base, summary in summaries:
        family = base.get("family", "")
        final_mse = _float(summary.get("final_mse", base.get("final_mse")))
        full = refs.get(family, {}).get("full", float("nan"))
        off = refs.get(family, {}).get("off", float("nan"))
        scores = summary.get("param_probe_scores", _load_json_field(base.get("param_probe_scores")))
        features = summary.get("param_probe_features", _load_json_field(base.get("param_probe_features")))
        rows.append({
            "case": base.get("case", ""),
            "family": family,
            "scenario": summary.get("scenario", base.get("scenario", "")),
            "ramp_shape": summary.get("ramp_shape", base.get("ramp_shape", "")),
            "embodiment": summary.get("embodiment", base.get("embodiment", "")),
            "mode": base.get("mode", ""),
            "use_stdw": summary.get("use_stdw", base.get("use_stdw", "")),
            "returncode": base.get("returncode", ""),
            "final_mse": final_mse,
            "final_mse_after_drift": summary.get("final_mse_after_drift", base.get("final_mse_after_drift", "")),
            "slow_loop_triggers": summary.get("slow_loop_triggers", base.get("slow_loop_triggers", "")),
            "gate_silenced_count": summary.get("gate_silenced_count", base.get("gate_silenced_count", "")),
            "delta_vs_full_pct": (final_mse / full - 1.0) * 100.0 if full == full and full else "",
            "delta_vs_off_pct": (final_mse / off - 1.0) * 100.0 if off == off and off else "",
            "param_probe_active_excitation": summary.get("param_probe_active_excitation", base.get("param_probe_active_excitation", "")),
            "param_probe_selected_family": summary.get("param_probe_selected_family", base.get("param_probe_selected_family", "")),
            "param_probe_reason": summary.get("param_probe_reason", base.get("param_probe_reason", "")),
            "param_probe_gate_slow_loop": summary.get("param_probe_gate_slow_loop", base.get("param_probe_gate_slow_loop", "")),
            "param_probe_gate_step": summary.get("param_probe_gate_step", base.get("param_probe_gate_step", "")),
            "param_probe_scores": json.dumps(scores, ensure_ascii=False),
            "param_probe_features": json.dumps(features, ensure_ascii=False),
            "summary_path": base.get("summary_path", ""),
        })
    return rows


def write_report(root: Path, rows: list[dict[str, Any]]) -> Path:
    report = root / "dynamic_param_id_report.md"
    lines = [
        "# Dynamic Parameter Identification Sweep",
        "",
        f"- rows: {len(rows)}",
        f"- summary csv: `{root / 'dynamic_param_id_summary.csv'}`",
        "",
        "## Main Table",
        "",
        "| family | mode | final_mse | vs_full_pct | vs_off_pct | active_excitation | selected | gate | reason |",
        "|---|---|---:|---:|---:|---|---|---|---|",
    ]
    for row in rows:
        lines.append(
            f"| {row['family']} | {row['mode']} | {row['final_mse']} | "
            f"{row['delta_vs_full_pct']} | {row['delta_vs_off_pct']} | "
            f"{row['param_probe_active_excitation']} | {row['param_probe_selected_family']} | {row['param_probe_gate_slow_loop']} | "
            f"{row['param_probe_reason']} |"
        )

    lines += [
        "",
        "## Interpretation Checklist",
        "",
        "- If param_probe distinguishes density / thruster_efficiency / thruster_angle, A3 history contains useful family-ID evidence.",
        "- If torque_negative_control is marked ambiguous/external-disturbance-like, it supports keeping torque pulse outside parameter estimation.",
        "- If active excitation still leaves families ambiguous, the paper conclusion should emphasize need for richer sensing or structured excitation.",
        "- Conservative gating is counted as useful only if it moves final_mse toward the off baseline without parameter compensation.",
    ]
    report.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=".results/exp_dynamic_param_id_20260613")
    args = parser.parse_args()
    root = Path(args.root)
    rows = collect(root)
    out = root / "dynamic_param_id_summary.csv"
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

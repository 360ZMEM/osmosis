"""STDW (v3) sweep driver.

Drives a grid of (use_stdw, enable_filter, use_quantile_filter, g_C_lr,
target_drift) combinations through ``custom_workflows/run_with_isaac_env.sh``
and aggregates each run's ``summary.json`` into a single CSV.

Defaults follow plan §3.5: ``--limit_combinations 4`` only runs the first 4
combos so the CSV/closed-loop wiring can be validated cheaply; pass
``--full_matrix`` to expand the entire 72-element grid.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import subprocess
import sys
from datetime import datetime
from itertools import product
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


REPO_ROOT = Path(__file__).resolve().parent.parent
RUNNER = REPO_ROOT / "custom_workflows" / "run_with_isaac_env.sh"
WORKFLOW_REL = Path("workflows_new_stdw") / "play_stdw_adapt.py"


DEFAULT_MATRIX: Dict[str, List[Any]] = {
    "use_stdw": [True],
    "enable_filter": [True],
    "use_quantile_filter": [True],
    "g_C_lr": [5e-5],
    "target_drift": [0.05],
    "scenario": [
        "none",
        "sine",
        "current_bias",
        "wave_plus_fault",
        "jonswap_mild",
        "jonswap_strong",
        "current_plus_jonswap",
        "wave_plus_noise",
    ],
    "embodiment": ["base", "long_body", "heavy_moderate", "asymmetric"],
}


# Algorithm-grid matrix preserved for backwards compatibility (sweep8 reproduction).
ALGO_MATRIX: Dict[str, List[Any]] = {
    "use_stdw": [True, False],
    "enable_filter": [True, False],
    "use_quantile_filter": [True, False],
    "g_C_lr": [1e-5, 5e-5, 1e-4],
    "target_drift": [0.03, 0.05, 0.08],
}


CSV_FIELDS = [
    "run_id",
    "use_stdw",
    "enable_filter",
    "use_quantile_filter",
    "g_C_lr",
    "target_drift",
    "scenario",
    "embodiment",
    "pseudo_gain",
    "pseudo_decay",
    "lambda_reg",
    "final_mse",
    "final_mse_after_drift",
    "convergence_step",
    "mean_total_mse",
    "max_total_mse",
    "slow_loop_triggers",
    "summary_path",
    "started_at",
    "ended_at",
    "returncode",
]


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="STDW v3 sweep driver.")
    parser.add_argument("--base_logs_root", type=str, default=str(REPO_ROOT / ".tmp" / "stdw_sweep"))
    parser.add_argument("--csv_out", type=str, default=str(REPO_ROOT / "logs" / "stdw_results.csv"))
    parser.add_argument("--total_steps", type=int, default=1400)
    parser.add_argument("--matrix", type=str, default=None, help="Optional path to a JSON file overriding the default matrix.")
    parser.add_argument("--limit_combinations", type=int, default=4)
    parser.add_argument("--full_matrix", action="store_true", default=False,
                        help="Run the entire DEFAULT_MATRIX (legacy alias for --full_grid).")
    parser.add_argument("--full_grid", action="store_true", default=False,
                        help="Cartesian product of scenarios x embodiments (= 28 runs by default).")
    parser.add_argument("--scenarios_only", action="store_true", default=False,
                        help="Vary scenario only; pin embodiment=base.")
    parser.add_argument("--embodiments_only", action="store_true", default=False,
                        help="Vary embodiment only; pin scenario=none.")
    parser.add_argument("--algo_grid", action="store_true", default=False,
                        help="Use the legacy algorithm-only matrix (ALGO_MATRIX) instead of scenarios x embodiments.")
    parser.add_argument("--include_keys", type=str, default=None,
                        help="Comma list; only these keys vary, the rest take their first value.")
    parser.add_argument("--dry_run", action="store_true", default=False)
    parser.add_argument("--task", type=str, default="EasyUUV-Direct-v1")
    parser.add_argument("--load_run", type=str, default="SS4")
    parser.add_argument("--checkpoint", type=str, default="model_500.pt")
    parser.add_argument("--logs_root", type=str, default=str(Path.home() / "isaaclab" / "logs" / "rsl_rl"))
    parser.add_argument("--cpu", action="store_true", default=True)
    parser.add_argument("--headless", action="store_true", default=True)
    parser.add_argument("--stability_threshold", type=float, default=None,
                        help="Override --stability_threshold passed to play_stdw_adapt.py.")
    parser.add_argument("--stability_threshold_rel", type=float, default=None,
                        help="Relative threshold multiplier (rel * baseline_compound_error_mean).")
    parser.add_argument("--stability_window", type=int, default=None,
                        help="Override --stability_window passed to play_stdw_adapt.py.")
    parser.add_argument("--ramp_shape", type=str, default=None, choices=[None, "linear", "cosine"],
                        help="Override --ramp_shape passed to play_stdw_adapt.py.")
    return parser


def _load_matrix(path: Optional[str], *, use_algo: bool = False) -> Dict[str, List[Any]]:
    if not path:
        base = ALGO_MATRIX if use_algo else DEFAULT_MATRIX
        return {k: list(v) for k, v in base.items()}
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("matrix file must contain a JSON object mapping str -> list")
    return {str(k): list(v) for k, v in payload.items()}


def _maybe_filter_matrix(matrix: Dict[str, List[Any]], include_keys: Optional[str]) -> Dict[str, List[Any]]:
    if not include_keys:
        return matrix
    keep = {s.strip() for s in include_keys.split(",") if s.strip()}
    out: Dict[str, List[Any]] = {}
    for key, values in matrix.items():
        if key in keep:
            out[key] = list(values)
        else:
            out[key] = [values[0]]
    return out


def _expand(matrix: Dict[str, List[Any]]) -> List[Dict[str, Any]]:
    keys = list(matrix.keys())
    combos = []
    for tup in product(*[matrix[k] for k in keys]):
        combo = dict(zip(keys, tup))
        combos.append(combo)
    return combos


def _format_value(value: Any) -> str:
    if isinstance(value, bool):
        return "True" if value else "False"
    if isinstance(value, float):
        return f"{value:g}"
    return str(value)


def _make_run_id(combo: Dict[str, Any], varying_keys: Optional[List[str]] = None) -> str:
    parts = []
    keys = varying_keys if varying_keys else list(combo.keys())
    for key in keys:
        if key not in combo:
            continue
        parts.append(f"{key}={_format_value(combo[key])}")
    rid = "_".join(parts).replace(" ", "")
    # 防御：如果 run_id 仍然过长（>180 字符），用 hash 截断
    if len(rid) > 180:
        import hashlib

        rid = rid[:120] + "_" + hashlib.md5(rid.encode()).hexdigest()[:8]
    return rid


def _build_argv(combo: Dict[str, Any], args: argparse.Namespace, sweep_run_dir: Path) -> List[str]:
    if not RUNNER.exists():
        raise FileNotFoundError(f"Missing Isaac Lab runner wrapper: {RUNNER}")
    workflow = REPO_ROOT / WORKFLOW_REL
    argv: List[str] = [
        "bash",
        str(RUNNER),
        str(workflow),
        "--task", args.task,
        "--num_envs", "1",
        "--load_run", args.load_run,
        "--checkpoint", args.checkpoint,
        "--logs_root", args.logs_root,
        "--results_root", str(sweep_run_dir / "results"),
        "--artifacts_root", str(sweep_run_dir / "artifacts"),
        "--total_steps", str(args.total_steps),
    ]
    if args.headless:
        argv.append("--headless")
    if args.cpu:
        argv.append("--cpu")
    if args.stability_threshold is not None:
        argv.extend(["--stability_threshold", _format_value(args.stability_threshold)])
    if args.stability_threshold_rel is not None:
        argv.extend(["--stability_threshold_rel", _format_value(args.stability_threshold_rel)])
    if args.stability_window is not None:
        argv.extend(["--stability_window", _format_value(args.stability_window)])
    if args.ramp_shape is not None:
        argv.extend(["--ramp_shape", str(args.ramp_shape)])
    for key, value in combo.items():
        argv.extend([f"--{key}", _format_value(value)])
    return argv


def _read_summary(sweep_run_dir: Path) -> Tuple[Optional[Path], Dict[str, Any]]:
    """Find the latest summary.json under sweep_run_dir/results.

    The play script writes ``stdw_new_<timestamp>/summary.json`` deep under
    ``results_root``; pick the most recently modified summary if multiple
    exist. Returns (path, payload_dict_or_empty).
    """
    results_dir = sweep_run_dir / "results"
    if not results_dir.exists():
        return None, {}
    candidates = list(results_dir.rglob("summary.json"))
    if not candidates:
        return None, {}
    candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    summary_path = candidates[0]
    try:
        payload = json.loads(summary_path.read_text(encoding="utf-8"))
    except Exception:
        return summary_path, {}
    return summary_path, payload


def _ensure_csv(csv_path: Path) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    if not csv_path.exists():
        with csv_path.open("w", newline="", encoding="utf-8") as fp:
            writer = csv.DictWriter(fp, fieldnames=CSV_FIELDS)
            writer.writeheader()


def _append_csv(csv_path: Path, row: Dict[str, Any]) -> None:
    with csv_path.open("a", newline="", encoding="utf-8") as fp:
        writer = csv.DictWriter(fp, fieldnames=CSV_FIELDS)
        writer.writerow(row)


def main() -> int:
    parser = _build_parser()
    args = parser.parse_args()

    matrix = _load_matrix(args.matrix, use_algo=bool(args.algo_grid))
    matrix = _maybe_filter_matrix(matrix, args.include_keys)

    # Submatrix switches: pin one of the new axes to a single value to keep
    # the run count tractable.
    if args.scenarios_only and "embodiment" in matrix:
        matrix["embodiment"] = ["base"]
    if args.embodiments_only and "scenario" in matrix:
        matrix["scenario"] = ["none"]

    combos = _expand(matrix)
    if not (args.full_matrix or args.full_grid):
        combos = combos[: max(int(args.limit_combinations), 1)]

    # 仅取实际变动的维度做 run_id，避免目录名过长
    varying_keys = [k for k, v in matrix.items() if len(set(map(str, v))) > 1]
    if not varying_keys:
        varying_keys = list(matrix.keys())

    base_logs_root = Path(args.base_logs_root)
    csv_out = Path(args.csv_out)
    _ensure_csv(csv_out)

    print(f"[sweep_stdw] {len(combos)} combinations queued (full_matrix={args.full_matrix})")

    overall_rc = 0
    for combo in combos:
        run_id = _make_run_id(combo, varying_keys=varying_keys)
        sweep_run_dir = base_logs_root / run_id
        sweep_run_dir.mkdir(parents=True, exist_ok=True)

        argv = _build_argv(combo, args, sweep_run_dir)
        print(f"[sweep_stdw] >>> {run_id}\n  argv = {' '.join(argv)}")

        if args.dry_run:
            continue

        started_at = datetime.now().isoformat(timespec="seconds")
        result = subprocess.run(argv, check=False)
        ended_at = datetime.now().isoformat(timespec="seconds")
        returncode = int(result.returncode)
        if returncode != 0:
            overall_rc = returncode
            print(f"[sweep_stdw] !! {run_id} returned {returncode}")

        summary_path, payload = _read_summary(sweep_run_dir)
        final_mse = payload.get("final_mse") if payload else None
        final_mse_after_drift = payload.get("final_mse_after_drift") if payload else None
        convergence_step = payload.get("convergence_step") if payload else None
        mean_total_mse = payload.get("mean_total_mse") if payload else None
        max_total_mse = payload.get("max_total_mse") if payload else None
        slow_loop_triggers = payload.get("slow_loop_triggers") if payload else None
        # Prefer scenario/embodiment from the play summary when available; fall
        # back to the combo dict (matters when scenario/embodiment aren't varied).
        scenario_val = (payload.get("scenario") if payload else None) or combo.get("scenario", "")
        embodiment_val = (payload.get("embodiment") if payload else None) or combo.get("embodiment", "")
        row: Dict[str, Any] = {
            "run_id": run_id,
            "use_stdw": _format_value(combo.get("use_stdw", "")),
            "enable_filter": _format_value(combo.get("enable_filter", "")),
            "use_quantile_filter": _format_value(combo.get("use_quantile_filter", "")),
            "g_C_lr": _format_value(combo.get("g_C_lr", "")),
            "target_drift": _format_value(combo.get("target_drift", "")),
            "scenario": _format_value(scenario_val),
            "embodiment": _format_value(embodiment_val),
            "pseudo_gain": _format_value(combo.get("pseudo_gain", "")),
            "pseudo_decay": _format_value(combo.get("pseudo_decay", "")),
            "lambda_reg": _format_value(combo.get("lambda_reg", "")),
            "final_mse": "" if final_mse is None or (isinstance(final_mse, float) and math.isnan(final_mse)) else final_mse,
            "final_mse_after_drift": "" if final_mse_after_drift is None or (isinstance(final_mse_after_drift, float) and math.isnan(final_mse_after_drift)) else final_mse_after_drift,
            "convergence_step": "" if convergence_step is None else convergence_step,
            "mean_total_mse": "" if mean_total_mse is None else mean_total_mse,
            "max_total_mse": "" if max_total_mse is None else max_total_mse,
            "slow_loop_triggers": "" if slow_loop_triggers is None else slow_loop_triggers,
            "summary_path": str(summary_path) if summary_path is not None else "",
            "started_at": started_at,
            "ended_at": ended_at,
            "returncode": returncode,
        }
        _append_csv(csv_out, row)

    if args.dry_run:
        print("[sweep_stdw] dry-run complete; no subprocesses launched.")
    else:
        print(f"[sweep_stdw] CSV written to {csv_out}")
    return overall_rc


if __name__ == "__main__":
    sys.exit(main())

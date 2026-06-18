"""72-cell sweep matrix driver for the 8-dim Parametric policy.

Dimensions: 3 axis × 3 magnitude × 4 embodiment × 2 flag = 72 cells.

Each cell:
    custom_workflows/run_with_isaac_env.sh workflows_new_stdw/play_meta_eval.py
        --task EasyUUV-Direct-Parametric-v1
        --policy_path <ckpt>
        --steps 800
        --cob_drift_axis <x|y|z>
        --cob_drift_magnitude <0.02|0.05|0.10>
        --cob_drift_start_step 200 --cob_drift_end_step 800
        --embodiment <base|long_body|heavy_moderate|asymmetric>
        --tune_gains True / --identity_init True
        --save_dir <out_root>/<cell_id>

Aggregates each cell's meta_eval_summary.json into <out_root>/sweep_matrix.csv.

Usage:
    python workflows_new_stdw/sweep_72cell.py \\
        --policy_path .../model_1499.pt --out_root .tmp/sweep_72_<ts>
"""
from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
import time
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent.parent
RUNNER = REPO_ROOT / "custom_workflows" / "run_with_isaac_env.sh"
WORKFLOW = REPO_ROOT / "workflows_new_stdw" / "play_meta_eval.py"

AXES = ("x", "y", "z")
MAGNITUDES = (0.02, 0.05, 0.10)
EMBODIMENTS = ("base", "long_body", "heavy_moderate", "asymmetric")
FLAGS = ("tune_gains", "identity_init")


def cell_id(axis: str, mag: float, emb: str, flag: str) -> str:
    return f"{axis}_{mag:.2f}_{emb}_{flag}"


def run_cell(args, axis: str, mag: float, emb: str, flag: str) -> tuple[int, dict]:
    cid = cell_id(axis, mag, emb, flag)
    out_dir = Path(args.out_root) / cid
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / "run.log"

    cli = [
        str(RUNNER), str(WORKFLOW),
        "--task", "EasyUUV-Direct-Parametric-v1",
        "--num_envs", "1",
        "--headless",
        "--policy_path", args.policy_path,
        "--save_dir", str(out_dir),
        "--steps", str(args.steps),
        "--cob_drift_axis", axis,
        "--cob_drift_magnitude", f"{mag}",
        "--cob_drift_start_step", str(args.drift_start),
        "--cob_drift_end_step", str(args.drift_end),
        "--embodiment", emb,
    ]
    if flag == "tune_gains":
        cli += ["--tune_gains", "True", "--identity_init", "False"]
    else:
        cli += ["--tune_gains", "True", "--identity_init", "True"]

    t0 = time.time()
    with log_path.open("w") as fp:
        proc = subprocess.run(cli, stdout=fp, stderr=subprocess.STDOUT)
    dt = time.time() - t0

    summary_path = out_dir / "meta_eval_summary.json"
    summary = {}
    if summary_path.exists():
        try:
            summary = json.loads(summary_path.read_text())
        except Exception as exc:
            summary = {"_parse_error": str(exc)}
    summary["_cell_id"] = cid
    summary["_axis"] = axis
    summary["_magnitude"] = mag
    summary["_embodiment"] = emb
    summary["_flag"] = flag
    summary["_returncode"] = proc.returncode
    summary["_wall_seconds"] = dt
    return proc.returncode, summary


def main() -> int:
    p = argparse.ArgumentParser(description="72-cell sweep over 8-dim Parametric policy.")
    p.add_argument("--policy_path", required=True)
    p.add_argument("--out_root", required=True)
    p.add_argument("--steps", type=int, default=800)
    p.add_argument("--drift_start", type=int, default=200)
    p.add_argument("--drift_end", type=int, default=800)
    p.add_argument("--limit", type=int, default=0,
                   help="Optional debug limit on number of cells (0 = full 72).")
    args = p.parse_args()

    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    rows: list[dict] = []
    cells = [(a, m, e, f) for a in AXES for m in MAGNITUDES for e in EMBODIMENTS for f in FLAGS]
    if args.limit:
        cells = cells[: args.limit]
    n = len(cells)

    print(f"[INFO] 72-cell sweep: running {n} cells -> {out_root}")
    failed = 0
    for idx, (axis, mag, emb, flag) in enumerate(cells, start=1):
        cid = cell_id(axis, mag, emb, flag)
        print(f"[{idx}/{n}] {cid}", flush=True)
        rc, summary = run_cell(args, axis, mag, emb, flag)
        rows.append(summary)
        if rc != 0:
            failed += 1
            print(f"  [FAIL rc={rc}] -> {out_root / cid / 'run.log'}", flush=True)
        # Persist incrementally so a mid-run kill doesn't lose data.
        _flush_csv(out_root / "sweep_matrix.csv", rows)
        (out_root / "sweep_matrix.json").write_text(json.dumps(rows, indent=2))

    print(f"[DONE] success={n - failed} fail={failed} -> {out_root / 'sweep_matrix.csv'}")
    return 0 if failed == 0 else 1


def _flush_csv(csv_path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    keys: list[str] = []
    seen = set()
    for r in rows:
        for k in r:
            if k not in seen:
                seen.add(k)
                keys.append(k)
    # Promote sweep keys to the front for readability.
    front = ["_cell_id", "_axis", "_magnitude", "_embodiment", "_flag",
             "_returncode", "_wall_seconds",
             "zeta_runtime_over_nominal_mean", "zeta_runtime_over_nominal_min",
             "zeta_runtime_over_nominal_max",
             "pe_active_ratio", "deadzone_active_ratio",
             "ang_vel_norm_mean", "ang_vel_norm_max"]
    ordered = [k for k in front if k in keys] + [k for k in keys if k not in front]
    with csv_path.open("w", newline="") as fp:
        w = csv.DictWriter(fp, fieldnames=ordered)
        w.writeheader()
        for r in rows:
            row = {k: r.get(k, "") for k in ordered}
            for k, v in row.items():
                if isinstance(v, list):
                    row[k] = json.dumps(v)
            w.writerow(row)


if __name__ == "__main__":
    sys.exit(main())

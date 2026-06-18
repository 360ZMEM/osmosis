#!/usr/bin/env python3
"""全实验矩阵 sweep 驱动：integration of tune × wave × embodiment × stdw × seed.

每 cell 跑一次 ``play_stdw_adapt.py``，产物写到 ``<out_root>/<cell_id>/``，
聚合表增量写到 ``<out_root>/full_matrix.{csv,json}``。

矩阵维度：
  wave        ∈ {calm, medium, storm}      (3)
  embodiment  ∈ {base, long_body, heavy_moderate, asymmetric} (4)
  tune        ∈ {identity, full}            (2)
  stdw        ∈ {off, on}                   (2)
  seed        ∈ {0, 1, 2}                   (3)

合计 144 trial（48 unique × 3 seed）。每 cell ≈ 70s。

接口边界：
  * yaml 内部已经写好 (wave × tune) 6 种组合 → ``configs/matrix_wave_<w>_<t>.yaml``
  * embodiment 通过 ``--embodiment`` 传入
  * stdw 开关用 (``--use_stdw`` × ``--target_drift``) 双关：off=False+0, on=True+0.05
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
import time
from itertools import product
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
RUNNER = REPO_ROOT / "custom_workflows" / "run_with_isaac_env.sh"
WORKFLOW = REPO_ROOT / "workflows" / "play_stdw_adapt.py"
CONFIG_DIR = REPO_ROOT / "workflows" / "configs"

WAVES = ("calm", "medium", "storm")
EMBODIMENTS = ("base", "long_body", "heavy_moderate", "asymmetric")
TUNES = ("identity", "full")
STDWS = ("off", "on")


def _parse_policy_path(p: Path):
    """从 ``<root>/logs/<exp>/<run>/<ckpt>.pt`` 解析 4 个 CLI 字段。"""
    p = p.resolve()
    if not p.is_file():
        raise SystemExit(f"[FATAL] policy_path not found: {p}")
    ckpt = p.name
    run = p.parent.name
    exp = p.parent.parent.name
    logs_root = p.parent.parent.parent
    if logs_root.name != "logs":
        print(f"[WARN] inferred logs_root={logs_root} but parent name is not 'logs'; "
              "if play_stdw_adapt fails to find ckpt, pass --logs_root manually.")
    return logs_root, exp, run, ckpt


def _csv_keys():
    return [
        "cell_id", "wave", "embodiment", "tune", "stdw", "seed",
        "returncode", "wall_seconds",
        "final_mse", "final_mse_after_drift", "convergence_step",
        "use_stdw", "target_drift", "embodiment_summary",
        "gate_silenced_count",
    ]


def _flush(rows, out_root: Path):
    csv_path = out_root / "full_matrix.csv"
    json_path = out_root / "full_matrix.json"
    keys = _csv_keys()
    with csv_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
        w.writeheader()
        for row in rows:
            w.writerow(row)
    json_path.write_text(json.dumps(rows, indent=2))


def _find_summary(cell_dir: Path):
    candidates = list(cell_dir.rglob("summary.json"))
    if not candidates:
        return {}
    candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    try:
        return json.loads(candidates[0].read_text())
    except Exception as exc:
        print(f"[WARN] failed to parse {candidates[0]}: {exc}")
        return {}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--policy_path", required=True, type=Path)
    ap.add_argument("--out_root", required=True, type=Path)
    ap.add_argument("--total_steps", type=int, default=1500)
    ap.add_argument("--seeds", default="0,1,2",
                    help="comma list of integer seeds")
    ap.add_argument("--waves", default=",".join(WAVES))
    ap.add_argument("--embodiments", default=",".join(EMBODIMENTS))
    ap.add_argument("--tunes", default=",".join(TUNES))
    ap.add_argument("--stdws", default=",".join(STDWS))
    ap.add_argument("--task", default="EasyUUV-Direct-Parametric-v1")
    ap.add_argument("--drift_target_on", type=float, default=0.05)
    ap.add_argument("--enable_trigger_gate", default="True",
                    help="TAG 开关，透传到每个 cell 的 play_stdw_adapt.py")
    ap.add_argument("--trigger_threshold", type=float, default=0.05,
                    help="TAG 阈值 (rad)，透传到每个 cell")
    args = ap.parse_args()

    out_root: Path = args.out_root
    out_root.mkdir(parents=True, exist_ok=True)
    print(f"[INFO] full-matrix out_root = {out_root}")

    logs_root, exp, run, ckpt = _parse_policy_path(args.policy_path)
    print(f"[INFO] policy: logs_root={logs_root}  exp={exp}  run={run}  ckpt={ckpt}")

    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
    waves = args.waves.split(",")
    embs = args.embodiments.split(",")
    tunes = args.tunes.split(",")
    stdws = args.stdws.split(",")

    cells = list(product(waves, embs, tunes, stdws, seeds))
    n = len(cells)
    print(f"[INFO] matrix: {len(waves)} waves × {len(embs)} embs × {len(tunes)} tunes "
          f"× {len(stdws)} stdws × {len(seeds)} seeds = {n} trials")

    rows = []
    succ = 0
    fail = 0
    t0_global = time.time()
    for i, (wave, emb, tune, stdw, seed) in enumerate(cells, 1):
        cell_id = f"{wave}_{emb}_{tune}_stdw-{stdw}_s{seed}"
        cell_dir = out_root / cell_id
        cell_dir.mkdir(parents=True, exist_ok=True)
        log_path = cell_dir / "run.log"

        cfg_yaml = CONFIG_DIR / f"matrix_wave_{wave}_{tune}.yaml"
        if not cfg_yaml.is_file():
            print(f"[FAIL] {cell_id}: missing yaml {cfg_yaml}")
            rows.append({"cell_id": cell_id, "wave": wave, "embodiment": emb,
                         "tune": tune, "stdw": stdw, "seed": seed,
                         "returncode": -1, "wall_seconds": 0.0})
            fail += 1
            continue

        use_stdw = "True" if stdw == "on" else "False"
        target_drift = args.drift_target_on if stdw == "on" else 0.0

        cli = [
            str(RUNNER), str(WORKFLOW),
            "--headless",
            "--task", args.task,
            "--num_envs", "1",
            "--experiment_name", exp,
            "--logs_root", str(logs_root),
            "--load_run", run,
            "--checkpoint", ckpt,
            "--workflow_config", str(cfg_yaml),
            "--embodiment", emb,
            "--use_stdw", use_stdw,
            "--target_drift", str(target_drift),
            "--enable_trigger_gate", str(args.enable_trigger_gate),
            "--trigger_threshold", str(args.trigger_threshold),
            "--total_steps", str(args.total_steps),
            "--seed", str(seed),
            "--results_root", str(cell_dir / "results"),
            "--artifacts_root", str(cell_dir / "artifacts"),
        ]

        elapsed_global = (time.time() - t0_global) / 60
        print(f"[{i:3d}/{n}] {cell_id}  (elapsed {elapsed_global:.1f} min)")

        t0 = time.time()
        with log_path.open("w") as f:
            f.write(f"# CMD: {' '.join(cli)}\n\n")
            f.flush()
            try:
                rc = subprocess.call(cli, stdout=f, stderr=subprocess.STDOUT, cwd=REPO_ROOT)
            except KeyboardInterrupt:
                print("[ABORT] user interrupt")
                _flush(rows, out_root)
                raise
            except Exception as exc:
                print(f"[FAIL] {cell_id}: {exc}")
                rc = -2
        wall = time.time() - t0

        summary = _find_summary(cell_dir)
        # success 以 summary.json 是否真正生成为准；rc=0 但 summary 缺失说明
        # play_stdw_adapt.py 内部 ImportError 之类被 simulation_app shutdown 掩盖。
        ok = bool(summary)
        row = {
            "cell_id": cell_id,
            "wave": wave,
            "embodiment": emb,
            "tune": tune,
            "stdw": stdw,
            "seed": seed,
            "returncode": rc,
            "wall_seconds": round(wall, 2),
            "final_mse": summary.get("final_mse"),
            "final_mse_after_drift": summary.get("final_mse_after_drift"),
            "convergence_step": summary.get("convergence_step"),
            "use_stdw": summary.get("use_stdw"),
            "target_drift": summary.get("target_drift"),
            "embodiment_summary": summary.get("embodiment"),
            "gate_silenced_count": summary.get("gate_silenced_count"),
        }
        rows.append(row)
        if ok:
            succ += 1
        else:
            fail += 1
            print(f"[FAIL] {cell_id} rc={rc} summary_missing; see {log_path}")

        # 增量落盘
        _flush(rows, out_root)

    total_wall = (time.time() - t0_global) / 60
    print(f"[DONE] success={succ} fail={fail} total_wall={total_wall:.1f} min "
          f"-> {out_root / 'full_matrix.csv'}")
    return 0 if fail == 0 else 1


if __name__ == "__main__":
    sys.exit(main())

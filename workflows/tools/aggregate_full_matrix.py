#!/usr/bin/env python3
"""聚合 sweep_full_matrix.py 的产物为 (wave, embodiment, tune, stdw) 维度的 mean ± std。

输入：``<matrix_dir>/full_matrix.csv``
输出：``<matrix_dir>/summary_aggregated.json``
      ``<matrix_dir>/summary_aggregated.csv``（人读对照表）
      ``<matrix_dir>/stdw_pairwise.csv``（按 (wave,emb,tune) 配对 STDW off/on）
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from statistics import mean, stdev


METRICS = ("final_mse", "final_mse_after_drift", "convergence_step")


def _to_float(v):
    if v is None or v == "":
        return None
    try:
        x = float(v)
        if math.isnan(x):
            return None
        return x
    except Exception:
        return None


def _agg(values):
    xs = [v for v in values if v is not None]
    if not xs:
        return {"n": 0, "mean": None, "std": None}
    return {
        "n": len(xs),
        "mean": float(mean(xs)),
        "std": float(stdev(xs)) if len(xs) > 1 else 0.0,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--matrix_dir", required=True, type=Path)
    args = ap.parse_args()

    csv_path = args.matrix_dir / "full_matrix.csv"
    with csv_path.open() as f:
        rows = list(csv.DictReader(f))

    # 按 (wave, emb, tune, stdw) 分组
    groups = defaultdict(list)
    for r in rows:
        key = (r["wave"], r["embodiment"], r["tune"], r["stdw"])
        groups[key].append(r)

    aggregated = {}
    for key, items in groups.items():
        wave, emb, tune, stdw = key
        bucket = {}
        for m in METRICS:
            bucket[m] = _agg([_to_float(it.get(m)) for it in items])
        aggregated["__".join(key)] = {
            "wave": wave, "embodiment": emb, "tune": tune, "stdw": stdw,
            **bucket,
        }

    out_json = args.matrix_dir / "summary_aggregated.json"
    out_json.write_text(json.dumps(aggregated, indent=2))

    # 人读对照 CSV
    out_csv = args.matrix_dir / "summary_aggregated.csv"
    with out_csv.open("w", newline="") as f:
        keys = ["wave", "embodiment", "tune", "stdw"]
        for m in METRICS:
            keys += [f"{m}_n", f"{m}_mean", f"{m}_std"]
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for entry in aggregated.values():
            row = {k: entry[k] for k in ("wave", "embodiment", "tune", "stdw")}
            for m in METRICS:
                row[f"{m}_n"] = entry[m]["n"]
                row[f"{m}_mean"] = entry[m]["mean"]
                row[f"{m}_std"] = entry[m]["std"]
            w.writerow(row)

    # STDW 配对：按 (wave, emb, tune) 配对 off vs on
    pair_path = args.matrix_dir / "stdw_pairwise.csv"
    seen = set()
    with pair_path.open("w", newline="") as f:
        keys = ["wave", "embodiment", "tune",
                "fmse_off_mean", "fmse_off_std", "fmse_on_mean", "fmse_on_std", "fmse_delta_pct",
                "fmse_drift_off_mean", "fmse_drift_on_mean", "fmse_drift_delta_pct"]
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for entry in aggregated.values():
            base_key = (entry["wave"], entry["embodiment"], entry["tune"])
            if base_key in seen:
                continue
            off = aggregated.get("__".join((*base_key, "off")))
            on = aggregated.get("__".join((*base_key, "on")))
            if off is None or on is None:
                continue
            seen.add(base_key)
            fmse_off_m = off["final_mse"]["mean"]
            fmse_on_m = on["final_mse"]["mean"]
            delta_pct = None
            if fmse_off_m not in (None, 0.0) and fmse_on_m is not None:
                delta_pct = 100.0 * (fmse_on_m - fmse_off_m) / fmse_off_m
            fmse_drift_off_m = off["final_mse_after_drift"]["mean"]
            fmse_drift_on_m = on["final_mse_after_drift"]["mean"]
            drift_delta_pct = None
            if fmse_drift_off_m not in (None, 0.0) and fmse_drift_on_m is not None:
                drift_delta_pct = 100.0 * (fmse_drift_on_m - fmse_drift_off_m) / fmse_drift_off_m
            w.writerow({
                "wave": base_key[0], "embodiment": base_key[1], "tune": base_key[2],
                "fmse_off_mean": fmse_off_m, "fmse_off_std": off["final_mse"]["std"],
                "fmse_on_mean": fmse_on_m, "fmse_on_std": on["final_mse"]["std"],
                "fmse_delta_pct": delta_pct,
                "fmse_drift_off_mean": fmse_drift_off_m,
                "fmse_drift_on_mean": fmse_drift_on_m,
                "fmse_drift_delta_pct": drift_delta_pct,
            })

    print(f"[DONE] {len(rows)} trials -> {len(aggregated)} groups")
    print(f"  json    : {out_json}")
    print(f"  csv     : {out_csv}")
    print(f"  pairwise: {pair_path}")


if __name__ == "__main__":
    main()

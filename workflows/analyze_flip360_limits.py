#!/usr/bin/env python3
"""Flip360 物理极限 / 参考速度分析脚本。

读取 play_stdw_adapt.py 产出的逐步 CSV（stdw_output_*.csv），分析：
  1. 姿态跟踪误差随 |参考角| 的分布 —— 找“难维持的角度”。
  2. 姿态跟踪误差随 |参考角速度| 的分布 —— 判断是否“参考太快”。
  3. 误差 vs 参考角速度的相关性，给出 angle-limited / rate-limited 判据。
  4. 控制力度（control_effort / executed_action）统计与饱和比例。

判据（启发式）：
  - 若误差主要随参考角速度上升、且高速 bin 出现控制饱和 -> rate-limited，
    建议降低 ref_sine_freq（放慢参考信号）。
  - 若误差集中在某些固定角度、与速度无关 -> angle-limited，
    建议在该角度做 keep 测试 / 检查物理极限（推力臂、浮心力矩）。

用法：
  python workflows/analyze_flip360_limits.py \
      --csv .../stdw_output_*.csv [--csv ...] \
      [--label base] [--label asym] \
      [--report out.md] [--dt 0.00833] [--decimation 10]

也可直接给 pressure_runs.csv 的 results_root，自动发现 CSV：
  python workflows/analyze_flip360_limits.py --root .results/flip360_curric_eval_2846
"""
from __future__ import annotations

import argparse
import csv
import glob
import math
import os
from typing import List, Optional, Sequence


def _angle_remap(a: float) -> float:
    return (a + math.pi) % (2.0 * math.pi) - math.pi


def _read_csv(path: str) -> List[dict]:
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def _fnum(row: dict, key: str) -> Optional[float]:
    v = row.get(key, "")
    if v is None or v == "":
        return None
    try:
        return float(v)
    except ValueError:
        return None


def _discover_csv(root: str) -> List[tuple[str, str]]:
    """在 results_root 下发现每个 case 的逐步 CSV，返回 [(label, path)]。"""
    out: List[tuple[str, str]] = []
    runs_csv = os.path.join(root, "pressure_runs.csv")
    if os.path.exists(runs_csv):
        for r in _read_csv(runs_csv):
            case = r.get("case", "case")
            sp = r.get("summary_path", "")
            if not sp:
                continue
            d = os.path.dirname(sp)
            hits = sorted(glob.glob(os.path.join(d, "stdw_output_*.csv")))
            if hits:
                out.append((case, hits[0]))
        if out:
            return out
    # 退化：递归找所有 stdw_output_*.csv
    for p in sorted(glob.glob(os.path.join(root, "**", "stdw_output_*.csv"), recursive=True)):
        out.append((os.path.basename(os.path.dirname(p)), p))
    return out


def _percentile(xs: Sequence[float], q: float) -> float:
    if not xs:
        return float("nan")
    s = sorted(xs)
    idx = min(len(s) - 1, max(0, int(round(q * (len(s) - 1)))))
    return s[idx]


def _bin_stats(pairs: Sequence[tuple[float, float]], edges: Sequence[float]) -> List[dict]:
    """pairs=(x, err)。按 edges 分箱，返回每箱 count / rmse(err)。"""
    bins = [[] for _ in range(len(edges) - 1)]
    for x, e in pairs:
        for i in range(len(edges) - 1):
            if edges[i] <= x < edges[i + 1] or (i == len(edges) - 2 and x == edges[i + 1]):
                bins[i].append(e)
                break
    rows = []
    for i, b in enumerate(bins):
        if b:
            rmse = math.sqrt(sum(v * v for v in b) / len(b))
        else:
            rmse = float("nan")
        rows.append({"lo": edges[i], "hi": edges[i + 1], "count": len(b), "rmse": rmse})
    return rows


def _corr(xs: Sequence[float], ys: Sequence[float]) -> float:
    n = len(xs)
    if n < 3:
        return float("nan")
    mx = sum(xs) / n
    my = sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    syy = sum((y - my) ** 2 for y in ys)
    sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    if sxx <= 0 or syy <= 0:
        return float("nan")
    return sxy / math.sqrt(sxx * syy)


def analyze_one(label: str, path: str, ctrl_dt: float) -> dict:
    rows = _read_csv(path)
    n = len(rows)
    angle_err_pairs_roll: List[tuple[float, float]] = []
    angle_err_pairs_pitch: List[tuple[float, float]] = []
    rate_err_pairs: List[tuple[float, float]] = []
    efforts: List[float] = []
    rates: List[float] = []
    errs: List[float] = []

    prev_des = None
    for r in rows:
        dr, dp = _fnum(r, "des_roll"), _fnum(r, "des_pitch")
        tr, tp = _fnum(r, "true_roll"), _fnum(r, "true_pitch")
        ce = _fnum(r, "control_effort")
        if None in (dr, dp, tr, tp):
            prev_des = (dr, dp)
            continue
        er = _angle_remap(tr - dr)
        ep = _angle_remap(tp - dp)
        comb_err = math.sqrt(er * er + ep * ep)
        angle_err_pairs_roll.append((abs(dr), abs(er)))
        angle_err_pairs_pitch.append((abs(dp), abs(ep)))
        if prev_des is not None and None not in prev_des:
            d_des_r = _angle_remap(dr - prev_des[0]) / ctrl_dt
            d_des_p = _angle_remap(dp - prev_des[1]) / ctrl_dt
            ref_rate = math.sqrt(d_des_r * d_des_r + d_des_p * d_des_p)
            rate_err_pairs.append((ref_rate, comb_err))
            rates.append(ref_rate)
            errs.append(comb_err)
        if ce is not None:
            efforts.append(ce)
        prev_des = (dr, dp)

    angle_edges = [0.0, math.pi / 6, math.pi / 3, math.pi / 2, 2 * math.pi / 3, 5 * math.pi / 6, math.pi + 1e-6]
    rmax = _percentile(rates, 0.999) if rates else 1.0
    rate_edges = [i * rmax / 6.0 for i in range(7)] if rmax > 0 else [0, 1]

    effort_p95 = _percentile(efforts, 0.95)
    effort_max = max(efforts) if efforts else float("nan")
    sat_thr = 0.9 * effort_max if efforts else float("nan")
    sat_frac = (sum(1 for e in efforts if e >= sat_thr) / len(efforts)) if efforts else float("nan")

    rate_bins = _bin_stats(rate_err_pairs, rate_edges)
    # rate-limited 判据：误差-速度相关 & 高速箱 rmse 明显高于低速箱
    rate_corr = _corr(rates, errs)
    valid = [b for b in rate_bins if b["count"] > 5 and not math.isnan(b["rmse"])]
    if len(valid) >= 2:
        lo_rmse = valid[0]["rmse"]
        hi_rmse = valid[-1]["rmse"]
        rate_ratio = hi_rmse / lo_rmse if lo_rmse > 0 else float("inf")
    else:
        rate_ratio = float("nan")

    return {
        "label": label,
        "path": path,
        "n": n,
        "overall_rmse": math.sqrt(sum(e * e for e in errs) / len(errs)) if errs else float("nan"),
        "angle_roll_bins": _bin_stats(angle_err_pairs_roll, angle_edges),
        "angle_pitch_bins": _bin_stats(angle_err_pairs_pitch, angle_edges),
        "rate_bins": rate_bins,
        "rate_edges": rate_edges,
        "rate_corr": rate_corr,
        "rate_ratio": rate_ratio,
        "effort_p95": effort_p95,
        "effort_max": effort_max,
        "sat_frac": sat_frac,
    }


def _verdict(a: dict) -> str:
    rc = a["rate_corr"]
    rr = a["rate_ratio"]
    sat = a["sat_frac"]
    rate_limited = (not math.isnan(rc) and rc > 0.25) and (not math.isnan(rr) and rr > 1.5)
    saturated = (not math.isnan(sat) and sat > 0.10)
    if rate_limited and saturated:
        return ("RATE-LIMITED + 饱和：误差随参考角速度显著上升且控制力度接近上限。"
                "建议降低 ref_sine_freq（放慢参考），或检查推力极限。")
    if rate_limited:
        return ("RATE-LIMITED：误差随参考角速度上升但未明显饱和。"
                "优先放慢参考信号（降低 ref_sine_freq）再评估。")
    # angle-limited：找最差角度箱
    worst = None
    for tag, key in (("roll", "angle_roll_bins"), ("pitch", "angle_pitch_bins")):
        for b in a[key]:
            if b["count"] > 5 and not math.isnan(b["rmse"]):
                if worst is None or b["rmse"] > worst[2]:
                    worst = (tag, (b["lo"], b["hi"]), b["rmse"])
    if worst:
        return ("ANGLE-LIMITED：误差与速度相关性弱，集中在 "
                f"{worst[0]} ∈ [{worst[1][0]:.2f},{worst[1][1]:.2f}] rad（rmse={worst[2]:.3f}）。"
                "建议在该角度做 keep 测试并核查物理力矩极限。")
    return "数据不足以判定。"


def _fmt_bins(bins: List[dict]) -> str:
    lines = ["| 区间(rad) | count | rmse |", "|---|---:|---:|"]
    for b in bins:
        rmse = "nan" if math.isnan(b["rmse"]) else f"{b['rmse']:.4f}"
        lines.append(f"| [{b['lo']:.2f}, {b['hi']:.2f}) | {b['count']} | {rmse} |")
    return "\n".join(lines)


def build_report(results: List[dict]) -> str:
    out = ["# Flip360 物理极限 / 参考速度分析", ""]
    for a in results:
        out.append(f"## {a['label']}")
        out.append("")
        out.append(f"- 源文件：`{a['path']}`")
        out.append(f"- 步数：{a['n']}")
        out.append(f"- 总体姿态 RMSE：{a['overall_rmse']:.4f} rad")
        out.append(f"- 误差-参考角速度相关：{a['rate_corr']:.3f}")
        rr = a["rate_ratio"]
        out.append(f"- 高/低速箱 rmse 比：{'nan' if math.isnan(rr) else f'{rr:.2f}'}")
        out.append(f"- control_effort p95 / max：{a['effort_p95']:.3f} / {a['effort_max']:.3f}")
        sf = a["sat_frac"]
        out.append(f"- 近饱和步占比(≥0.9·max)：{'nan' if math.isnan(sf) else f'{sf*100:.1f}%'}")
        out.append("")
        out.append("**误差随 |参考 roll| 分布**")
        out.append(_fmt_bins(a["angle_roll_bins"]))
        out.append("")
        out.append("**误差随 |参考 pitch| 分布**")
        out.append(_fmt_bins(a["angle_pitch_bins"]))
        out.append("")
        out.append("**误差随 |参考角速度| 分布**")
        out.append(_fmt_bins(a["rate_bins"]))
        out.append("")
        out.append(f"**判据**：{_verdict(a)}")
        out.append("")
    return "\n".join(out)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", action="append", default=[], help="逐步 CSV 路径，可多次")
    ap.add_argument("--label", action="append", default=[], help="与 --csv 对应的标签")
    ap.add_argument("--root", default=None, help="results_root，自动发现 CSV")
    ap.add_argument("--report", default=None, help="输出 markdown 报告路径")
    ap.add_argument("--dt", type=float, default=1.0 / 120.0, help="sim dt（秒）")
    ap.add_argument("--decimation", type=int, default=10, help="控制 decimation")
    args = ap.parse_args()

    ctrl_dt = args.dt * args.decimation
    pairs: List[tuple[str, str]] = []
    if args.root:
        pairs.extend(_discover_csv(args.root))
    for i, p in enumerate(args.csv):
        label = args.label[i] if i < len(args.label) else os.path.basename(p)
        pairs.append((label, p))
    if not pairs:
        ap.error("需要 --csv 或 --root")

    results = [analyze_one(lbl, p, ctrl_dt) for lbl, p in pairs]
    report = build_report(results)
    print(report)
    if args.report:
        os.makedirs(os.path.dirname(os.path.abspath(args.report)), exist_ok=True)
        with open(args.report, "w") as f:
            f.write(report)
        print(f"\n[written] {args.report}")


if __name__ == "__main__":
    main()

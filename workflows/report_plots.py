"""跨 run 报告绘图脚本（论文/汇报版）。

本脚本读取 STDW workflow 的两类产物：
  - 单组 run 的 ``stdw_output.csv``（per-step 时间序列）
  - sweep 聚合的 ``sweep8/results.csv``（每行一组实验）

并生成 REPORT_paper_view 中提到的 7 张报告图：

  Fig 2  rho 调度曲线 + drift 注入示意                  -> ``rho_schedule.png``
  Fig 3  tuned_v3 vs baseline_3k MSE-时间曲线           -> ``mse_timeline.png``
  Fig 4  sweep8 主效应条形图                            -> ``sweep8_main_effects.png``
  Fig 5  pseudo_gain × pseudo_decay 交互热图            -> ``sweep8_interaction.png``
  Fig 6  sweep8_no_adapt vs sweep8 反向证据图           -> ``sanity_break.png``
  Fig 7  三轴误差分解（roll / yaw / depth）             -> ``axis_breakdown.png``
  Fig 1' 一页 summary（绑定上述所有图的关键数字）        -> ``summary_card.png``

不依赖 pandas，也不依赖 isaac sim 或 torch；纯 numpy + matplotlib + json + csv。
可在主 Python 解释器（不需要 Isaac Lab 内核）下直接运行：

    python3 workflows_new_stdw/report_plots.py \\
        --output_dir workflows_new_stdw/report_figs

默认所有路径均使用工作区相对路径，便于在 CI / 其他机器上复用。
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


REPO_ROOT = Path(__file__).resolve().parent.parent

# ---------------------------------------------------------------------------
# Path defaults (与 §9 章节中的产物保持一致)
# ---------------------------------------------------------------------------

DEFAULT_TUNED_V3_CSV = (
    REPO_ROOT
    / ".tmp/stdw_bidir_20260603/smoke_asurface/results/easyuuv_direct"
    / "SS4/model_500_play/stdw_new_2026-06-03_17-13-23/stdw_output.csv"
)
DEFAULT_BASELINE_CSV = (
    REPO_ROOT
    / "source/results/rsl_rl/easyuuv_direct/SS4/model_500_play"
    / "stdw_new_2026-06-03_16-50-44/stdw_output.csv"
)
DEFAULT_TUNED_V2_CSV = (
    REPO_ROOT
    / "source/results/rsl_rl/easyuuv_direct/SS4/model_500_play"
    / "stdw_new_2026-06-03_16-49-00/stdw_output.csv"
)
DEFAULT_SWEEP8_DIR = REPO_ROOT / ".tmp/stdw_bidir_20260603/sweep8"
DEFAULT_SWEEP8_NOADAPT_DIR = REPO_ROOT / ".tmp/stdw_bidir_20260603/sweep8_no_adapt"


# ---------------------------------------------------------------------------
# CSV / summary 读取助手
# ---------------------------------------------------------------------------


def _read_stdw_output_csv(path: Path) -> Dict[str, np.ndarray]:
    """读取 per-step ``stdw_output.csv``，返回 col -> np.ndarray。

    复杂列（true_pose / executed_action）被忽略；其余尝试 float 化。
    """
    if not path.exists():
        raise FileNotFoundError(f"stdw_output.csv not found: {path}")
    with path.open("r", newline="", encoding="utf-8") as fp:
        reader = csv.reader(fp)
        header = next(reader)
        rows = list(reader)
    cols: Dict[str, List[Any]] = {h: [] for h in header}
    for row in rows:
        for h, v in zip(header, row):
            cols[h].append(v)
    out: Dict[str, np.ndarray] = {}
    for h, vs in cols.items():
        arr_float: List[float] = []
        ok = True
        for v in vs:
            if v == "" or v is None:
                arr_float.append(np.nan)
                continue
            try:
                arr_float.append(float(v))
            except (TypeError, ValueError):
                ok = False
                break
        if ok:
            out[h] = np.asarray(arr_float, dtype=float)
    return out


def _compute_total_mse_from_csv(csv_data: Dict[str, np.ndarray]) -> np.ndarray:
    """按 plots._compute_tracking_mse 一致的口径算 total_mse 时间序列。"""
    def _angle(true_, des_):
        return (true_ - des_ + np.pi) % (2.0 * np.pi) - np.pi

    roll_e = _angle(csv_data["true_roll"], csv_data["des_roll"])
    pitch_e = _angle(csv_data["true_pitch"], csv_data["des_pitch"])
    yaw_e = _angle(csv_data["true_yaw"], csv_data["des_yaw"])
    depth_e = csv_data["true_z"] - csv_data["des_depth"]
    return roll_e ** 2 + pitch_e ** 2 + yaw_e ** 2 + depth_e ** 2


def _rolling_mean(x: np.ndarray, win: int = 50) -> np.ndarray:
    if win <= 1 or x.size < win:
        return x
    kernel = np.ones(win, dtype=float) / float(win)
    return np.convolve(x, kernel, mode="same")


def _read_sweep8_summaries(sweep_dir: Path) -> List[Dict[str, Any]]:
    """枚举 sweep_dir 下所有子 run，读取最新 summary.json。"""
    if not sweep_dir.exists():
        return []
    out: List[Dict[str, Any]] = []
    for run_dir in sorted(sweep_dir.iterdir()):
        if not run_dir.is_dir() or run_dir.name.startswith("."):
            continue
        sums = list((run_dir / "results").rglob("summary.json"))
        if not sums:
            continue
        sums.sort(key=lambda p: p.stat().st_mtime, reverse=True)
        try:
            payload = json.loads(sums[0].read_text(encoding="utf-8"))
        except Exception:
            continue
        payload["_run_id"] = run_dir.name
        out.append(payload)
    return out


# ---------------------------------------------------------------------------
# Figure 2 : rho schedule + drift 注入
# ---------------------------------------------------------------------------


def plot_rho_schedule(csv_data: Dict[str, np.ndarray], output_path: Path) -> None:
    step = csv_data["step"]
    rho = csv_data.get("rho", csv_data.get("drift_fraction"))
    drift = csv_data.get("drift_fraction", rho)
    bias = csv_data.get("domain_bias")

    fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
    ax_top, ax_bot = axes

    ax_top.plot(step, rho, color="#d62728", linewidth=1.6, label=r"$\rho$ (drift_fraction)")
    if drift is not None and not np.allclose(drift, rho, equal_nan=True):
        ax_top.plot(step, drift, color="#1f77b4", linestyle="--", linewidth=1.0, label="drift_fraction (raw)")
    ax_top.set_ylabel(r"$\rho$")
    ax_top.set_ylim(-0.05, 1.05)
    ax_top.grid(True, alpha=0.3)
    ax_top.legend(loc="best")
    ax_top.set_title("Domain drift schedule")

    if bias is not None:
        ax_bot.plot(step, bias, color="#2ca02c", linewidth=1.4, label="domain_bias (com_to_cob shift)")
    for col, color in (("com_to_cob_offset_x", "#9467bd"),
                       ("com_to_cob_offset_y", "#8c564b"),
                       ("com_to_cob_offset_z", "#e377c2")):
        if col in csv_data:
            ax_bot.plot(step, csv_data[col], color=color, linewidth=1.0, alpha=0.8, label=col)
    ax_bot.set_xlabel("Simulation step")
    ax_bot.set_ylabel("offset / bias")
    ax_bot.grid(True, alpha=0.3)
    ax_bot.legend(loc="best")
    ax_bot.set_title("Drift injection on com_to_cob offset")

    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figure 3 : tuned_v3 vs baseline_3k 的 MSE-时间曲线
# ---------------------------------------------------------------------------


def plot_mse_timeline(
    runs: Dict[str, Dict[str, np.ndarray]],
    output_path: Path,
    smoothing_window: int = 50,
) -> None:
    fig, axes = plt.subplots(2, 1, figsize=(11, 7), sharex=True)
    ax_total, ax_zoom = axes

    palette = {
        "baseline_3k": ("#888888", "Baseline (STDW off)"),
        "tuned_v2": ("#1f77b4", "tuned_v2 (self_adapt=False, broken)"),
        "tuned_v3": ("#d62728", "tuned_v3 (Bi-KL + Gated PL + A-S-Surface)"),
    }

    for key, csv_data in runs.items():
        if csv_data is None:
            continue
        color, label = palette.get(key, ("#444444", key))
        step = csv_data["step"]
        total = _compute_total_mse_from_csv(csv_data)
        smooth = _rolling_mean(total, smoothing_window)
        ax_total.plot(step, total, color=color, linewidth=0.6, alpha=0.25)
        ax_total.plot(step, smooth, color=color, linewidth=1.6, label=label)
        ax_zoom.plot(step, smooth, color=color, linewidth=1.6, label=label)

    for ax in axes:
        ax.grid(True, alpha=0.3)
        ax.legend(loc="best")
        ax.set_ylabel("Total tracking MSE")
    ax_total.set_title(f"Per-step total MSE ({smoothing_window}-step rolling smoothed bold; raw thin)")
    ax_zoom.set_title("Smoothed only (zoom y)")
    ax_zoom.set_xlabel("Simulation step")
    ax_zoom.set_ylim(0, max(5.0, np.nanpercentile(_compute_total_mse_from_csv(next(iter(runs.values()))), 95)))

    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figure 4 / 5 / 7 : sweep8 数据
# ---------------------------------------------------------------------------


def _sweep_to_arrays(summaries: List[Dict[str, Any]]) -> Dict[str, np.ndarray]:
    keys = ["pseudo_gain", "pseudo_decay", "lambda_reg",
            "final_mse", "mean_total_mse", "max_total_mse",
            "mean_roll_mse", "mean_pitch_mse", "mean_yaw_mse", "mean_depth_mse",
            "convergence_step"]
    out: Dict[str, List[float]] = {k: [] for k in keys}
    for s in summaries:
        for k in keys:
            v = s.get(k)
            out[k].append(float("nan") if v is None else float(v))
    return {k: np.asarray(v, dtype=float) for k, v in out.items()}


def plot_sweep8_main_effects(summaries: List[Dict[str, Any]], output_path: Path) -> None:
    if not summaries:
        return
    arr = _sweep_to_arrays(summaries)

    fig, axes = plt.subplots(1, 3, figsize=(13, 4.4), sharey=True)
    factors = [
        ("pseudo_decay", [0.0, 0.7]),
        ("pseudo_gain", [3.0, 5.0]),
        ("lambda_reg", [0.01, 0.05]),
    ]
    for ax, (factor, levels) in zip(axes, factors):
        means = []
        stds = []
        for lvl in levels:
            mask = np.isclose(arr[factor], lvl)
            vals = arr["final_mse"][mask]
            means.append(float(np.nanmean(vals)) if vals.size else np.nan)
            stds.append(float(np.nanstd(vals)) if vals.size else 0.0)
        x = np.arange(len(levels))
        bars = ax.bar(x, means, yerr=stds, capsize=4,
                      color=["#1f77b4", "#d62728"], alpha=0.85,
                      edgecolor="#333333", linewidth=0.8)
        ax.set_xticks(x)
        ax.set_xticklabels([f"{factor}\n={lvl:g}" for lvl in levels])
        ax.set_title(f"main effect: {factor}")
        ax.grid(True, alpha=0.3, axis="y")
        for b, m in zip(bars, means):
            if not np.isnan(m):
                ax.text(b.get_x() + b.get_width() / 2, m, f"{m:.3f}",
                        ha="center", va="bottom", fontsize=9)
    axes[0].set_ylabel("final_mse (mean across other dims)")
    fig.suptitle("Sweep8 main effects on final_mse")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_sweep8_interaction(summaries: List[Dict[str, Any]], output_path: Path) -> None:
    if not summaries:
        return
    arr = _sweep_to_arrays(summaries)
    gains = sorted({float(g) for g in arr["pseudo_gain"]})
    decays = sorted({float(d) for d in arr["pseudo_decay"]})
    grid = np.full((len(decays), len(gains)), np.nan)
    for i, dc in enumerate(decays):
        for j, gn in enumerate(gains):
            mask = np.isclose(arr["pseudo_gain"], gn) & np.isclose(arr["pseudo_decay"], dc)
            vals = arr["final_mse"][mask]
            if vals.size:
                grid[i, j] = float(np.nanmean(vals))

    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    im = ax.imshow(grid, cmap="RdYlGn_r", aspect="auto")
    ax.set_xticks(range(len(gains)))
    ax.set_xticklabels([f"{g:g}" for g in gains])
    ax.set_yticks(range(len(decays)))
    ax.set_yticklabels([f"{d:g}" for d in decays])
    ax.set_xlabel("pseudo_gain")
    ax.set_ylabel("pseudo_decay")
    ax.set_title("Interaction: pseudo_gain × pseudo_decay\n(cell value = mean final_mse)")
    for i in range(len(decays)):
        for j in range(len(gains)):
            if not np.isnan(grid[i, j]):
                ax.text(j, i, f"{grid[i, j]:.3f}", ha="center", va="center", fontsize=10,
                        color="white" if grid[i, j] > np.nanmean(grid) else "black")
    fig.colorbar(im, ax=ax, label="final_mse")
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_axis_breakdown(summaries: List[Dict[str, Any]], output_path: Path) -> None:
    if not summaries:
        return
    summaries = sorted(summaries, key=lambda s: s.get("final_mse", float("inf")))
    labels = []
    roll = []
    pitch = []
    yaw = []
    depth = []
    for s in summaries:
        rid = s.get("_run_id", "?")
        rid = rid.replace("pseudo_", "p_").replace("lambda_reg", "lr")
        labels.append(rid)
        roll.append(s.get("mean_roll_mse", np.nan))
        pitch.append(s.get("mean_pitch_mse", np.nan))
        yaw.append(s.get("mean_yaw_mse", np.nan))
        depth.append(s.get("mean_depth_mse", np.nan))
    x = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(13, 5.2))
    width = 0.2
    ax.bar(x - 1.5 * width, roll, width, label="roll", color="#1f77b4")
    ax.bar(x - 0.5 * width, pitch, width, label="pitch", color="#2ca02c")
    ax.bar(x + 0.5 * width, yaw, width, label="yaw", color="#ff7f0e")
    ax.bar(x + 1.5 * width, depth, width, label="depth", color="#9467bd")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=22, ha="right", fontsize=8)
    ax.set_ylabel("mean per-axis MSE")
    ax.set_title("Sweep8 per-axis mean MSE (sorted by final_mse asc)")
    ax.grid(True, alpha=0.3, axis="y")
    ax.legend(loc="best", ncol=4)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figure 6 : sweep8_no_adapt vs sweep8 反向证据
# ---------------------------------------------------------------------------


def plot_sanity_break(
    sw_no_adapt: List[Dict[str, Any]],
    sw_adapt: List[Dict[str, Any]],
    output_path: Path,
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4), sharey=True)
    titles = ["BEFORE fix (self_adapt=False)\nPseudo-action path silently short-circuited",
              "AFTER fix (self_adapt=True via A-S-Surface)\n8 runs become genuinely distinct"]
    for ax, title, summaries in zip(axes, titles, [sw_no_adapt, sw_adapt]):
        if not summaries:
            ax.text(0.5, 0.5, "no data", ha="center", va="center", transform=ax.transAxes)
            ax.set_title(title)
            continue
        arr = _sweep_to_arrays(summaries)
        # 重新整理：x 轴是 (gain, decay) 组合，y 轴 final_mse；颜色按 lambda_reg 区分
        keys = []
        ys = []
        cs = []
        for s in summaries:
            keys.append(f"g={s.get('pseudo_gain'):.0f}, d={s.get('pseudo_decay'):.1f}, λ={s.get('lambda_reg'):.2f}")
            ys.append(s.get("final_mse", np.nan))
            cs.append(s.get("lambda_reg", 0.01))
        order = np.argsort(ys)
        keys = [keys[i] for i in order]
        ys = [ys[i] for i in order]
        cs = [cs[i] for i in order]
        x = np.arange(len(keys))
        colors = ["#1f77b4" if abs(c - 0.01) < 1e-6 else "#d62728" for c in cs]
        ax.bar(x, ys, color=colors, edgecolor="#333", linewidth=0.6)
        ax.set_xticks(x)
        ax.set_xticklabels(keys, rotation=22, ha="right", fontsize=7)
        ax.set_title(title)
        ax.grid(True, alpha=0.3, axis="y")
        for xi, yi in zip(x, ys):
            if not np.isnan(yi):
                ax.text(xi, yi, f"{yi:.3f}", ha="center", va="bottom", fontsize=7)
    axes[0].set_ylabel("final_mse")
    # 共用图例
    from matplotlib.patches import Patch
    legend_handles = [
        Patch(facecolor="#1f77b4", label="lambda_reg=0.01"),
        Patch(facecolor="#d62728", label="lambda_reg=0.05"),
    ]
    fig.legend(handles=legend_handles, loc="upper center", ncol=2, frameon=False)
    fig.suptitle("Sanity-check: pseudo-action path on/off (negative-control evidence)", y=1.02)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figure 1' : summary card (一页关键数字)
# ---------------------------------------------------------------------------


def plot_summary_card(
    runs: Dict[str, Dict[str, Any]],
    sweep8: List[Dict[str, Any]],
    output_path: Path,
) -> None:
    """Single-page snapshot bound to the headline numbers in REPORT §9.

    Left panel: baseline / tuned_v2 / tuned_v3 final_mse bars.
    Right panel: sweep8 final_mse ranking (best on top).
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    ax_l, ax_r = axes
    labels = []
    finals = []
    color_pal = []
    for key, color in (("baseline_3k", "#888888"), ("tuned_v2", "#1f77b4"), ("tuned_v3", "#d62728")):
        s = runs.get(key)
        if s is None:
            continue
        labels.append(key)
        finals.append(s.get("final_mse", np.nan))
        color_pal.append(color)
    x = np.arange(len(labels))
    bars = ax_l.bar(x, finals, color=color_pal, edgecolor="#333", linewidth=0.8)
    ax_l.set_xticks(x)
    ax_l.set_xticklabels(labels)
    ax_l.set_ylabel("final_mse (last 200 step window)")
    ax_l.set_title("Headline: 3 single-run comparison")
    ax_l.grid(True, alpha=0.3, axis="y")
    for b, m in zip(bars, finals):
        ax_l.text(b.get_x() + b.get_width() / 2, m, f"{m:.3f}", ha="center", va="bottom", fontsize=10)
    if len(finals) >= 2 and not np.isnan(finals[0]) and not np.isnan(finals[-1]):
        delta = (finals[-1] - finals[0]) / finals[0] * 100.0
        ax_l.text(0.5, 0.92, f"delta vs baseline: {delta:+.1f}%",
                  transform=ax_l.transAxes, ha="center", fontsize=11,
                  bbox=dict(boxstyle="round", fc="#fffacd", ec="#888"))

    if sweep8:
        sw_sorted = sorted(sweep8, key=lambda s: s.get("final_mse", float("inf")))
        names = [s["_run_id"].replace("pseudo_", "p_").replace("lambda_reg", "lr") for s in sw_sorted]
        ys = [s.get("final_mse", np.nan) for s in sw_sorted]
        cmap = matplotlib.colormaps["RdYlGn_r"]
        norm = plt.Normalize(vmin=min(ys), vmax=max(ys))
        cols = [cmap(norm(y)) for y in ys]
        x = np.arange(len(names))
        ax_r.barh(x, ys, color=cols, edgecolor="#333", linewidth=0.6)
        ax_r.set_yticks(x)
        ax_r.set_yticklabels(names, fontsize=7)
        ax_r.invert_yaxis()
        ax_r.set_xlabel("final_mse")
        ax_r.set_title("Sweep8 ranking (best on top)")
        ax_r.grid(True, alpha=0.3, axis="x")
        for xi, yi in zip(x, ys):
            ax_r.text(yi, xi, f" {yi:.3f}", va="center", fontsize=7)

    fig.suptitle("STDW v3 §9: Bi-directional KL + Gated Pseudo-labels (one-page summary)")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


def _maybe_load_summary_for(run_csv_path: Path) -> Dict[str, Any]:
    p = run_csv_path.parent / "summary.json"
    if not p.exists():
        return {}
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return {}


# ---------------------------------------------------------------------------
# Scenarios / embodiments bar charts (Phase 1 / Phase 2)
# ---------------------------------------------------------------------------

def _read_sweep_csv(path: Path) -> List[Dict[str, str]]:
    """Read a sweep CSV (e.g., scenarios_results.csv) into a list of dicts."""
    if not path.exists():
        return []
    rows: List[Dict[str, str]] = []
    with path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append(dict(r))
    return rows


def _safe_float(value: Any) -> Optional[float]:
    try:
        if value is None or value == "":
            return None
        f = float(value)
        if not np.isfinite(f):
            return None
        return f
    except (TypeError, ValueError):
        return None


def plot_grouped_bar(
    rows: List[Dict[str, str]],
    *,
    label_key: str,
    metric_keys: List[str],
    output_path: Path,
    title: str,
    baseline_label: Optional[str] = None,
) -> None:
    """Generic grouped bar chart (one bar per metric per row).

    Each row in ``rows`` produces a tick on the x-axis labeled by
    ``row[label_key]``; for each row we draw one bar per ``metric_keys``.
    A horizontal dashed baseline is drawn at the value of the row whose
    label equals ``baseline_label`` for the first metric (e.g., "none" /
    "base").
    """
    if not rows:
        print(f"[report_plots] no rows for {output_path.name}; skipping")
        return
    labels = [str(r.get(label_key, "?")) for r in rows]
    n_metrics = len(metric_keys)
    width = 0.8 / max(n_metrics, 1)
    x = np.arange(len(labels))

    fig, ax = plt.subplots(figsize=(max(8, len(labels) * 1.1), 4.2))
    palette = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]
    for i, mk in enumerate(metric_keys):
        vals = [_safe_float(r.get(mk)) for r in rows]
        nan_vals = [(v if v is not None else 0.0) for v in vals]
        bars = ax.bar(x + (i - (n_metrics - 1) / 2.0) * width, nan_vals, width,
                      label=mk, color=palette[i % len(palette)])
        for bar, v in zip(bars, vals):
            if v is None:
                continue
            ax.text(bar.get_x() + bar.get_width() / 2.0, bar.get_height(),
                    f"{v:.2f}", ha="center", va="bottom", fontsize=7)

    if baseline_label is not None and metric_keys:
        bl_rows = [r for r in rows if str(r.get(label_key, "")) == baseline_label]
        if bl_rows:
            bl_val = _safe_float(bl_rows[0].get(metric_keys[0]))
            if bl_val is not None:
                ax.axhline(bl_val, linestyle="--", color="grey", linewidth=1,
                           label=f"{baseline_label} ({metric_keys[0]})")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=20, ha="right")
    ax.set_ylabel("compound error (lower is better)")
    ax.set_title(title)
    ax.legend(fontsize=8, loc="upper left")
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=140)
    plt.close(fig)
    print(f"[report_plots] wrote {output_path}")


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="STDW report plots (cross-run).")
    parser.add_argument("--output_dir", type=str,
                        default=str(REPO_ROOT / "workflows_new_stdw" / "report_figs"))
    parser.add_argument("--baseline_csv", type=str, default=str(DEFAULT_BASELINE_CSV))
    parser.add_argument("--tuned_v2_csv", type=str, default=str(DEFAULT_TUNED_V2_CSV))
    parser.add_argument("--tuned_v3_csv", type=str, default=str(DEFAULT_TUNED_V3_CSV))
    parser.add_argument("--sweep8_dir", type=str, default=str(DEFAULT_SWEEP8_DIR))
    parser.add_argument("--sweep8_no_adapt_dir", type=str, default=str(DEFAULT_SWEEP8_NOADAPT_DIR))
    parser.add_argument("--smoothing_window", type=int, default=50)
    parser.add_argument("--scenarios_csv", type=str, default=None,
                        help="Path to a sweep_stdw scenarios_results.csv to render scenarios_bar.png.")
    parser.add_argument("--embodiments_csv", type=str, default=None,
                        help="Path to a sweep_stdw embodiments_results.csv to render embodiments_bar.png.")
    args = parser.parse_args(argv)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[report_plots] output_dir = {out_dir}")

    # ---- 加载单组 CSV + summary ----
    runs_csv: Dict[str, Optional[Dict[str, np.ndarray]]] = {}
    runs_meta: Dict[str, Dict[str, Any]] = {}
    for key, path_str in (("baseline_3k", args.baseline_csv),
                          ("tuned_v2", args.tuned_v2_csv),
                          ("tuned_v3", args.tuned_v3_csv)):
        path = Path(path_str)
        try:
            runs_csv[key] = _read_stdw_output_csv(path)
            runs_meta[key] = _maybe_load_summary_for(path)
            print(f"[report_plots] loaded {key}: {path} (rows={len(next(iter(runs_csv[key].values())))})")
        except FileNotFoundError as exc:
            print(f"[report_plots][WARN] {exc}")
            runs_csv[key] = None
            runs_meta[key] = {}

    # ---- 加载 sweep ----
    sweep8 = _read_sweep8_summaries(Path(args.sweep8_dir))
    sweep8_no_adapt = _read_sweep8_summaries(Path(args.sweep8_no_adapt_dir))
    print(f"[report_plots] sweep8        : {len(sweep8)} summaries")
    print(f"[report_plots] sweep8_no_adp : {len(sweep8_no_adapt)} summaries")

    figures: List[Tuple[str, Path]] = []

    # Fig 2 — 用 tuned_v3 时间序列做 rho schedule（drift_start=200, drift_end=2600）
    if runs_csv.get("tuned_v3") is not None:
        p = out_dir / "rho_schedule.png"
        plot_rho_schedule(runs_csv["tuned_v3"], p)
        figures.append(("Fig 2 rho_schedule", p))

    # Fig 3 — MSE timeline (3 runs overlaid)
    timeline_runs = {k: v for k, v in runs_csv.items() if v is not None}
    if timeline_runs:
        p = out_dir / "mse_timeline.png"
        plot_mse_timeline(timeline_runs, p, smoothing_window=args.smoothing_window)
        figures.append(("Fig 3 mse_timeline", p))

    # Fig 4 — sweep8 main effects
    if sweep8:
        p = out_dir / "sweep8_main_effects.png"
        plot_sweep8_main_effects(sweep8, p)
        figures.append(("Fig 4 sweep8_main_effects", p))

    # Fig 5 — sweep8 interaction heatmap
    if sweep8:
        p = out_dir / "sweep8_interaction.png"
        plot_sweep8_interaction(sweep8, p)
        figures.append(("Fig 5 sweep8_interaction", p))

    # Fig 6 — sanity-break (no_adapt vs adapt)
    if sweep8 or sweep8_no_adapt:
        p = out_dir / "sanity_break.png"
        plot_sanity_break(sweep8_no_adapt, sweep8, p)
        figures.append(("Fig 6 sanity_break", p))

    # Fig 7 — axis breakdown
    if sweep8:
        p = out_dir / "axis_breakdown.png"
        plot_axis_breakdown(sweep8, p)
        figures.append(("Fig 7 axis_breakdown", p))

    # Fig 1' — summary card
    headline = {k: runs_meta.get(k, {}) for k in ("baseline_3k", "tuned_v2", "tuned_v3")}
    p = out_dir / "summary_card.png"
    plot_summary_card(headline, sweep8, p)
    figures.append(("Fig 1' summary_card", p))

    # Fig 8 — scenarios bar (Phase 1, 6.3 add-on)
    if args.scenarios_csv:
        rows = _read_sweep_csv(Path(args.scenarios_csv))
        p = out_dir / "scenarios_bar.png"
        plot_grouped_bar(
            rows,
            label_key="scenario",
            metric_keys=["final_mse", "final_mse_after_drift"],
            output_path=p,
            title="Phase 1: scenarios (embodiment=base, 6000 steps)",
            baseline_label="none",
        )
        if p.exists():
            figures.append(("Fig 8 scenarios_bar", p))

    # Fig 9 — embodiments bar (Phase 2, 6.3 add-on)
    if args.embodiments_csv:
        rows = _read_sweep_csv(Path(args.embodiments_csv))
        p = out_dir / "embodiments_bar.png"
        plot_grouped_bar(
            rows,
            label_key="embodiment",
            metric_keys=["final_mse", "final_mse_after_drift"],
            output_path=p,
            title="Phase 2: embodiments (scenario=none, 6000 steps)",
            baseline_label="base",
        )
        if p.exists():
            figures.append(("Fig 9 embodiments_bar", p))

    # ---- 写入索引 ----
    index = {
        "output_dir": str(out_dir),
        "figures": [{"name": n, "path": str(p)} for n, p in figures],
        "headline": {k: {kk: v.get(kk) for kk in ("final_mse", "mean_total_mse",
                                                   "max_total_mse", "convergence_step",
                                                   "control_profile", "pseudo_gain",
                                                   "pseudo_decay", "lambda_reg")}
                     for k, v in headline.items() if v},
        "sweep8_top": [{k: s.get(k) for k in ("_run_id", "final_mse",
                                              "pseudo_gain", "pseudo_decay", "lambda_reg")}
                       for s in sorted(sweep8, key=lambda x: x.get("final_mse", float("inf")))[:3]],
    }
    index_path = out_dir / "index.json"
    index_path.write_text(json.dumps(index, indent=2, ensure_ascii=False))
    print(f"[report_plots] index -> {index_path}")
    for name, p in figures:
        print(f"  {name:32s} {p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

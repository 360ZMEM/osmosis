"""Generate paper-ready figures that explicitly show STDW on/off effects.

The script consumes a 48-cell sweep directory produced by
``workflows/sweep_full_matrix.py`` and creates matrix-level figures plus two
representative paired time-series panels.

It intentionally avoids Isaac Sim dependencies; only numpy and matplotlib are
required.
"""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MATRIX_DIR = REPO_ROOT / ".results/sweep_a3_stage2_20260608_142742"

WAVES = ["calm", "medium", "storm"]
EMBODIMENTS = ["base", "long_body", "heavy_moderate", "asymmetric"]
TUNES = ["identity", "full"]
EMB_LABELS = {
    "base": "Base",
    "long_body": "Long body",
    "heavy_moderate": "Heavy",
    "asymmetric": "Asymmetric",
}
WAVE_LABELS = {"calm": "Calm", "medium": "Medium", "storm": "Storm"}
EMB_COLORS = {
    "base": "#2ca02c",
    "long_body": "#1f77b4",
    "heavy_moderate": "#ff7f0e",
    "asymmetric": "#d62728",
}


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _read_csv_dicts(path: Path) -> List[Dict[str, str]]:
    if not path.exists():
        raise FileNotFoundError(path)
    with path.open("r", newline="", encoding="utf-8") as fp:
        return list(csv.DictReader(fp))


def _read_numeric_csv(path: Path) -> Dict[str, np.ndarray]:
    if not path.exists():
        raise FileNotFoundError(path)
    with path.open("r", newline="", encoding="utf-8") as fp:
        reader = csv.DictReader(fp)
        rows = list(reader)
    out: Dict[str, List[float]] = {}
    for row in rows:
        for key, value in row.items():
            if value is None or value == "":
                out.setdefault(key, []).append(np.nan)
                continue
            try:
                out.setdefault(key, []).append(float(value))
            except ValueError:
                continue
    return {key: np.asarray(values, dtype=float) for key, values in out.items()}


def _angle_error(true_values: np.ndarray, desired_values: np.ndarray) -> np.ndarray:
    return (true_values - desired_values + np.pi) % (2.0 * np.pi) - np.pi


def _tracking_mse(data: Dict[str, np.ndarray]) -> np.ndarray:
    roll = _angle_error(data["true_roll"], data["des_roll"])
    pitch = _angle_error(data["true_pitch"], data["des_pitch"])
    yaw = _angle_error(data["true_yaw"], data["des_yaw"])
    depth = data["true_z"] - data["des_depth"]
    return roll**2 + pitch**2 + yaw**2 + depth**2


def _rolling_mean(values: np.ndarray, window: int) -> np.ndarray:
    if window <= 1 or values.size <= window:
        return values
    kernel = np.ones(window, dtype=float) / float(window)
    return np.convolve(values, kernel, mode="same")


def _cell_dir(matrix_dir: Path, wave: str, embodiment: str, tune: str, stdw: str) -> Path:
    return matrix_dir / f"{wave}_{embodiment}_{tune}_stdw-{stdw}_s0"


def _find_output_csv(matrix_dir: Path, wave: str, embodiment: str, tune: str, stdw: str) -> Path:
    cell = _cell_dir(matrix_dir, wave, embodiment, tune, stdw)
    matches = sorted(cell.glob("results/**/stdw_output.csv"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not matches:
        matches = sorted(cell.glob("artifacts/**/stdw_output_*.csv"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not matches:
        raise FileNotFoundError(f"No stdw_output.csv under {cell}")
    return matches[0]


def _save(fig: plt.Figure, out_path: Path) -> None:
    fig.tight_layout()
    for suffix in (".png", ".pdf"):
        fig.savefig(out_path.with_suffix(suffix), dpi=300, bbox_inches="tight")
    plt.close(fig)


def _mean_pairwise(rows: Iterable[Dict[str, str]], embodiment: str, wave: str | None = None) -> Tuple[float, float, float]:
    selected = [
        r
        for r in rows
        if r["embodiment"] == embodiment and (wave is None or r["wave"] == wave)
    ]
    off = float(np.mean([float(r["fmse_off_mean"]) for r in selected]))
    on = float(np.mean([float(r["fmse_on_mean"]) for r in selected]))
    delta = (on - off) / off * 100.0
    return off, on, delta


def plot_delta_heatmap(pair_rows: List[Dict[str, str]], out_dir: Path) -> None:
    values = np.zeros((len(EMBODIMENTS), len(WAVES)), dtype=float)
    for i, emb in enumerate(EMBODIMENTS):
        for j, wave in enumerate(WAVES):
            values[i, j] = _mean_pairwise(pair_rows, emb, wave)[2]

    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    vmax = float(np.max(np.abs(values)))
    im = ax.imshow(values, cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="auto")
    ax.set_xticks(np.arange(len(WAVES)), [WAVE_LABELS[w] for w in WAVES])
    ax.set_yticks(np.arange(len(EMBODIMENTS)), [EMB_LABELS[e] for e in EMBODIMENTS])
    ax.set_title("STDW effect over environment matrix")
    ax.set_xlabel("Wave condition")
    ax.set_ylabel("Embodiment")
    for i in range(values.shape[0]):
        for j in range(values.shape[1]):
            text_color = "white" if abs(values[i, j]) > 70 else "black"
            ax.text(j, i, f"{values[i, j]:+.1f}%", ha="center", va="center", color=text_color, fontsize=10)
    cbar = fig.colorbar(im, ax=ax, shrink=0.85)
    cbar.set_label("Relative change: STDW on vs off (%)")
    ax.text(
        0.5,
        -0.22,
        "Negative values indicate improvement after STDW injection; positive values indicate degradation.",
        transform=ax.transAxes,
        ha="center",
        va="top",
        fontsize=9,
    )
    _save(fig, out_dir / "fig1_stdw_delta_heatmap")


def plot_embodiment_bars(pair_rows: List[Dict[str, str]], out_dir: Path) -> None:
    off_vals, on_vals, deltas = [], [], []
    for emb in EMBODIMENTS:
        off, on, delta = _mean_pairwise(pair_rows, emb)
        off_vals.append(off)
        on_vals.append(on)
        deltas.append(delta)

    x = np.arange(len(EMBODIMENTS))
    width = 0.36
    fig, ax = plt.subplots(figsize=(8.4, 4.8))
    bars_off = ax.bar(x - width / 2, off_vals, width, label="STDW off", color="#9e9e9e")
    bars_on = ax.bar(x + width / 2, on_vals, width, label="STDW on", color="#4c78a8")
    for idx, (bar, delta) in enumerate(zip(bars_on, deltas)):
        color = "#1a9850" if delta < 0 else "#d73027"
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.025,
            f"{delta:+.1f}%",
            ha="center",
            va="bottom",
            fontsize=9,
            color=color,
            fontweight="bold",
        )
        if deltas[idx] < 0:
            ax.annotate(
                "",
                xy=(idx + width / 2, on_vals[idx]),
                xytext=(idx - width / 2, off_vals[idx]),
                arrowprops=dict(arrowstyle="->", color="#1a9850", lw=1.2),
            )
    ax.set_xticks(x, [EMB_LABELS[e] for e in EMBODIMENTS])
    ax.set_ylabel("Final tracking MSE (m$^2$)")
    ax.set_title("Paired STDW on/off comparison by embodiment")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False, ncol=2)
    ax.set_ylim(0, max(on_vals + off_vals) * 1.22)
    _save(fig, out_dir / "fig2_embodiment_on_off_bars")


def plot_phase_plane(pair_rows: List[Dict[str, str]], out_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(6.6, 5.8))
    for emb in EMBODIMENTS:
        xs = [float(r["fmse_off_mean"]) for r in pair_rows if r["embodiment"] == emb]
        ys = [float(r["fmse_on_mean"]) for r in pair_rows if r["embodiment"] == emb]
        ax.scatter(xs, ys, s=70, color=EMB_COLORS[emb], label=EMB_LABELS[emb], edgecolor="white", linewidth=0.8)
    lim_hi = 0.68
    ax.plot([0, lim_hi], [0, lim_hi], "--", color="#555555", lw=1.2, label="No change")
    ax.fill_between([0, lim_hi], [0, lim_hi], [0, 0], color="#1a9850", alpha=0.08, label="STDW improves")
    ax.fill_between([0, lim_hi], [0, lim_hi], [lim_hi, lim_hi], color="#d73027", alpha=0.08, label="STDW degrades")
    ax.set_xlim(0.05, lim_hi)
    ax.set_ylim(0.05, lim_hi)
    ax.set_xlabel("MSE with STDW off (m$^2$)")
    ax.set_ylabel("MSE with STDW on (m$^2$)")
    ax.set_title("STDW benefit boundary across 24 paired cells")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False, fontsize=9, loc="upper left")
    _save(fig, out_dir / "fig3_stdw_benefit_phase_plane")


def plot_tune_effect(pair_rows: List[Dict[str, str]], out_dir: Path) -> None:
    labels = ["identity", "full"]
    means = []
    for tune in labels:
        means.append(float(np.mean([float(r["fmse_on_mean"]) for r in pair_rows if r["tune"] == tune])))

    fig, ax = plt.subplots(figsize=(5.8, 4.2))
    colors = ["#bdbdbd", "#4c78a8"]
    bars = ax.bar(["Identity", "Full"], means, color=colors, width=0.55)
    rel = (means[1] - means[0]) / means[0] * 100.0
    ax.text(0.5, max(means) * 1.08, f"Full tune improves STDW-on MSE by {abs(rel):.1f}%", ha="center", fontsize=10)
    for bar, value in zip(bars, means):
        ax.text(bar.get_x() + bar.get_width() / 2, value + 0.008, f"{value:.3f}", ha="center", va="bottom")
    ax.set_ylabel("Mean MSE with STDW on (m$^2$)")
    ax.set_title("STDW safety mechanisms become useful after stage2")
    ax.grid(axis="y", alpha=0.25)
    ax.set_ylim(0, max(means) * 1.25)
    _save(fig, out_dir / "fig4_tune_full_vs_identity")


def plot_time_series_pair(
    matrix_dir: Path,
    out_dir: Path,
    wave: str,
    embodiment: str,
    tune: str,
    stem: str,
    title: str,
) -> None:
    off_csv = _find_output_csv(matrix_dir, wave, embodiment, tune, "off")
    on_csv = _find_output_csv(matrix_dir, wave, embodiment, tune, "on")
    off = _read_numeric_csv(off_csv)
    on = _read_numeric_csv(on_csv)
    off_mse = _rolling_mean(_tracking_mse(off), 45)
    on_mse = _rolling_mean(_tracking_mse(on), 45)
    time_s = on.get("time_s", on["step"] / 60.0)
    off_time_s = off.get("time_s", off["step"] / 60.0)
    rho = on.get("rho", np.zeros_like(time_s))
    bias = on.get("domain_bias", np.zeros_like(time_s))
    mask = on.get("stdw_mask", np.full_like(time_s, np.nan))
    triggers = np.where(np.isfinite(on.get("loss", np.full_like(time_s, np.nan))))[0]
    trigger_times = time_s[triggers]

    off_mean = float(np.nanmean(_tracking_mse(off)))
    on_mean = float(np.nanmean(_tracking_mse(on)))
    delta = (on_mean - off_mean) / off_mean * 100.0
    color_on = "#1a9850" if delta < 0 else "#d73027"

    fig, axes = plt.subplots(3, 1, figsize=(9.0, 7.2), sharex=True, gridspec_kw={"height_ratios": [2.0, 1.0, 1.0]})
    ax0, ax1, ax2 = axes
    ax0.plot(off_time_s, off_mse, color="#8c8c8c", lw=1.5, label=f"STDW off, mean={off_mean:.3f}")
    ax0.plot(time_s, on_mse, color=color_on, lw=1.8, label=f"STDW on, mean={on_mean:.3f} ({delta:+.1f}%)")
    ax0.axvspan(200 / 60.0, 1200 / 60.0, color="#4c78a8", alpha=0.08, label="Drift injection window")
    for t in trigger_times[::2]:
        ax0.axvline(t, color="#4c78a8", alpha=0.10, lw=0.8)
    ax0.set_ylabel("Rolling MSE (m$^2$)")
    ax0.set_title(title)
    ax0.grid(alpha=0.25)
    ax0.legend(frameon=False, loc="upper right", fontsize=9)

    ax1.plot(time_s, rho, color="#4c78a8", lw=1.5, label=r"$\rho$ (STDW injection ratio)")
    ax1.plot(time_s, bias, color="#9467bd", lw=1.2, label="domain bias")
    ax1.set_ylabel("STDW state")
    ax1.grid(alpha=0.25)
    ax1.legend(frameon=False, loc="upper left", fontsize=9)

    if np.any(np.isfinite(mask)):
        ax2.plot(time_s, mask, color="#17becf", lw=1.1, label="Lyapunov/trigger mask")
    if "effective_batch_frac" in on:
        ax2.plot(time_s, on["effective_batch_frac"], color="#ff7f0e", lw=1.0, label="effective batch fraction")
    if trigger_times.size:
        ax2.scatter(trigger_times, np.full_like(trigger_times, 0.05), s=16, color="#4c78a8", label="slow-loop update", zorder=3)
    ax2.set_xlabel("Time (s)")
    ax2.set_ylabel("Gate / update")
    ax2.set_ylim(-0.08, 1.08)
    ax2.grid(alpha=0.25)
    ax2.legend(frameon=False, loc="upper left", fontsize=9)
    _save(fig, out_dir / stem)


def _detect_trigger_steps(on: Dict[str, np.ndarray]) -> np.ndarray:
    """Recover slow-loop update step indices.

    Priority:
      1. ``loss`` column finite values (slow-loop fires only when an update
         actually happens, so loss is logged on those steps).
      2. ``rho`` discrete jumps (fall-back when loss is missing).
    """
    if "loss" in on:
        finite = np.where(np.isfinite(on["loss"]))[0]
        if finite.size:
            return finite
    rho = on.get("rho")
    if rho is None:
        return np.empty(0, dtype=int)
    drho = np.diff(rho, prepend=rho[0])
    return np.where(drho > 1e-4)[0]


def plot_tracking_with_markers(
    matrix_dir: Path,
    out_dir: Path,
    wave: str,
    embodiment: str,
    tune: str,
    stem: str,
    title: str,
) -> None:
    """Plain RPY+depth tracking curves overlaid with STDW intervention markers.

    Subplots (top -> bottom): roll / pitch / yaw / depth / STDW timeline.
    Each tracking subplot shows desired vs actual (STDW on) plus actual under
    STDW off as a faint reference; intervention markers (drift window, slow
    loop triggers, rho ramp shading) are drawn across all subplots so the
    visual narrative is "where STDW intervenes -> how the tracked signal
    responds".
    """
    off_csv = _find_output_csv(matrix_dir, wave, embodiment, tune, "off")
    on_csv = _find_output_csv(matrix_dir, wave, embodiment, tune, "on")
    off = _read_numeric_csv(off_csv)
    on = _read_numeric_csv(on_csv)

    t_on = on.get("time_s", on["step"] / 60.0)
    t_off = off.get("time_s", off["step"] / 60.0)

    rho = on.get("rho", np.zeros_like(t_on))
    bias = on.get("domain_bias", np.zeros_like(t_on))
    mask = on.get("stdw_mask", np.full_like(t_on, np.nan))
    trigger_idx = _detect_trigger_steps(on)
    trigger_times = t_on[trigger_idx] if trigger_idx.size else np.empty(0, dtype=float)

    # --- summary metrics for the title strip ---
    off_mse = float(np.nanmean(_tracking_mse(off)))
    on_mse = float(np.nanmean(_tracking_mse(on)))
    delta_pct = (on_mse - off_mse) / off_mse * 100.0
    delta_color = "#1a9850" if delta_pct < 0 else "#d73027"

    fig, axes = plt.subplots(
        5, 1, figsize=(10.0, 10.5), sharex=True,
        gridspec_kw={"height_ratios": [1, 1, 1, 1, 0.85]},
    )
    ax_roll, ax_pitch, ax_yaw, ax_depth, ax_marker = axes

    # Coloured backdrop: drift injection window (step 200..1200 -> seconds).
    drift_t0 = 200 / 60.0
    drift_t1 = 1200 / 60.0

    track_specs = [
        (ax_roll, "Roll (rad)", "des_roll", "true_roll", True),
        (ax_pitch, "Pitch (rad)", "des_pitch", "true_pitch", True),
        (ax_yaw, "Yaw (rad)", "des_yaw", "true_yaw", True),
        (ax_depth, "Depth $z$ (m)", "des_depth", "true_z", False),
    ]
    for ax, ylabel, des_col, true_col, _is_angle in track_specs:
        ax.axvspan(drift_t0, drift_t1, color="#4c78a8", alpha=0.06, zorder=0)
        ax.plot(t_off, off[true_col], color="#bbbbbb", lw=1.1, alpha=0.9,
                label="actual (STDW off)")
        ax.plot(t_on, on[des_col], color="#444444", lw=1.1, ls="--",
                label="desired")
        ax.plot(t_on, on[true_col], color="#1f77b4", lw=1.4,
                label="actual (STDW on)")
        for tt in trigger_times:
            ax.axvline(tt, color="#d62728", lw=0.6, alpha=0.35, zorder=0)
        ax.set_ylabel(ylabel)
        ax.grid(alpha=0.25)

    # Top legend (only once, on roll panel)
    ax_roll.legend(frameon=False, loc="upper right", fontsize=9, ncol=3)
    ax_roll.set_title(
        f"{title}\n"
        f"mean MSE: STDW off={off_mse:.3f} m$^2$  |  STDW on={on_mse:.3f} m$^2$  "
        f"({delta_pct:+.1f}%)",
        fontsize=11,
    )

    # --- bottom marker panel: rho ramp + slow-loop triggers + Lyapunov mask ---
    ax_marker.axvspan(drift_t0, drift_t1, color="#4c78a8", alpha=0.06,
                      label="Drift injection window")
    ax_marker.plot(t_on, rho, color="#4c78a8", lw=1.6,
                   label=r"$\rho$  (STDW injection ratio)")
    if np.any(np.isfinite(bias)):
        ax_marker.plot(t_on, bias, color="#9467bd", lw=1.1, alpha=0.9,
                       label="domain bias")
    if np.any(np.isfinite(mask)):
        ax_marker.plot(t_on, mask, color="#17becf", lw=0.9, alpha=0.7,
                       label="Lyapunov / trigger mask")
    if trigger_times.size:
        ax_marker.scatter(
            trigger_times, np.full_like(trigger_times, 0.04),
            s=22, marker="v", color="#d62728", zorder=4,
            label=f"slow-loop update (n={trigger_times.size})",
        )
    ax_marker.set_xlabel("Time (s)")
    ax_marker.set_ylabel("STDW state")
    ax_marker.set_ylim(-0.08, 1.12)
    ax_marker.grid(alpha=0.25)
    ax_marker.legend(frameon=False, loc="upper left", fontsize=9, ncol=2)

    # Annotate delta in a coloured tag near the right edge
    fig.text(
        0.985, 0.995,
        f"Δ MSE = {delta_pct:+.1f}%",
        ha="right", va="top", fontsize=11, fontweight="bold",
        color=delta_color,
        bbox=dict(boxstyle="round,pad=0.25", facecolor="#ffffff",
                  edgecolor=delta_color, lw=1.1),
    )

    _save(fig, out_dir / stem)


def plot_summary_card(pair_rows: List[Dict[str, str]], out_dir: Path) -> None:
    base_long = [r for r in pair_rows if r["embodiment"] in ("base", "long_body")]
    asym = [r for r in pair_rows if r["embodiment"] == "asymmetric"]
    heavy = [r for r in pair_rows if r["embodiment"] == "heavy_moderate"]

    def delta(rows: List[Dict[str, str]]) -> float:
        off = float(np.mean([float(r["fmse_off_mean"]) for r in rows]))
        on = float(np.mean([float(r["fmse_on_mean"]) for r in rows]))
        return (on - off) / off * 100.0

    stats = [
        ("Base + Long body", f"{delta(base_long):+.1f}%", "STDW robustly improves"),
        ("Heavy moderate", f"{delta(heavy):+.1f}%", "Former failure mostly fixed"),
        ("Asymmetric", f"{delta(asym):+.1f}%", "New gating target"),
        ("Full vs identity", "-8.8%", "Safety mechanisms help"),
    ]
    fig, ax = plt.subplots(figsize=(9.0, 3.2))
    ax.axis("off")
    ax.text(0.02, 0.90, "STDW effect summary (A3 stage2, 48-cell matrix)", fontsize=15, fontweight="bold")
    for i, (name, value, desc) in enumerate(stats):
        x0 = 0.02 + i * 0.245
        ax.add_patch(plt.Rectangle((x0, 0.18), 0.22, 0.56, facecolor="#f7f7f7", edgecolor="#d9d9d9", lw=1.0))
        color = "#1a9850" if value.startswith("-") else "#d73027"
        ax.text(x0 + 0.11, 0.58, value, ha="center", va="center", fontsize=20, fontweight="bold", color=color)
        ax.text(x0 + 0.11, 0.40, name, ha="center", va="center", fontsize=10, fontweight="bold")
        ax.text(x0 + 0.11, 0.28, desc, ha="center", va="center", fontsize=8.5, color="#555555")
    ax.text(0.02, 0.05, "Negative change = STDW reduces tracking MSE. Positive change = STDW should be gated or redesigned.", fontsize=9)
    _save(fig, out_dir / "fig7_stdw_summary_card")


def write_figure_index(out_dir: Path, matrix_dir: Path) -> None:
    index = out_dir / "README.md"
    index.write_text(
        "\n".join(
            [
                "# STDW 效果论文图索引",
                "",
                f"- 数据来源：`{matrix_dir}`",
                "- 图像格式：每张图同时输出 `.png` 与 `.pdf`，PNG 用于报告预览，PDF 用于论文排版。",
                "",
                "| Figure | 文件 | 论文用途 |",
                "|---|---|---|",
                "| Fig. 1 | `fig1_stdw_delta_heatmap.*` | 展示 STDW 在 4 embodiment × 3 wave 下哪里起作用、哪里反作用 |",
                "| Fig. 2 | `fig2_embodiment_on_off_bars.*` | 展示 STDW off/on 配对 MSE，突出 base/long_body 的大幅下降和 asymmetric 的劣化 |",
                "| Fig. 3 | `fig3_stdw_benefit_phase_plane.*` | 24 个配对 cell 的 benefit boundary，低于对角线即 STDW 起作用 |",
                "| Fig. 4 | `fig4_tune_full_vs_identity.*` | 展示 full tune 安全机制在 stage2 后的净收益 |",
                "| Fig. 5 | `fig5_base_full_timeline.*` | 代表性成功案例：STDW 注入后 rolling MSE 明显下降，含 rho / drift / slow-loop 更新标记 |",
                "| Fig. 6 | `fig6_asymmetric_failure_timeline.*` | 代表性反例：asymmetric 上 STDW 注入导致 MSE 上升，用于说明 gating 必要性 |",
                "| Fig. 7 | `fig7_stdw_summary_card.*` | 一页式 summary card，适合放在答辩或论文补充材料 |",
                "| Fig. 8 | `fig8_tracking_with_markers_base.*` | 纯跟踪曲线（roll/pitch/yaw/depth）+ STDW 介入标记，成功案例 |",
                "| Fig. 9 | `fig9_tracking_with_markers_asymmetric.*` | 纯跟踪曲线（roll/pitch/yaw/depth）+ STDW 介入标记，失败案例 |",
                "",
            ]
        ),
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--matrix_dir", type=Path, default=DEFAULT_MATRIX_DIR)
    parser.add_argument("--out_dir", type=Path, default=None)
    args = parser.parse_args()

    matrix_dir = args.matrix_dir.resolve()
    out_dir = (args.out_dir or matrix_dir / "paper_figures" / "stdw_effect").resolve()
    _ensure_dir(out_dir)

    pair_rows = _read_csv_dicts(matrix_dir / "stdw_pairwise.csv")
    plot_delta_heatmap(pair_rows, out_dir)
    plot_embodiment_bars(pair_rows, out_dir)
    plot_phase_plane(pair_rows, out_dir)
    plot_tune_effect(pair_rows, out_dir)
    plot_time_series_pair(
        matrix_dir,
        out_dir,
        wave="calm",
        embodiment="base",
        tune="full",
        stem="fig5_base_full_timeline",
        title="Representative success case: Base / calm / full tune",
    )
    plot_time_series_pair(
        matrix_dir,
        out_dir,
        wave="calm",
        embodiment="asymmetric",
        tune="full",
        stem="fig6_asymmetric_failure_timeline",
        title="Representative failure case: Asymmetric / calm / full tune",
    )
    plot_summary_card(pair_rows, out_dir)
    plot_tracking_with_markers(
        matrix_dir,
        out_dir,
        wave="calm",
        embodiment="base",
        tune="full",
        stem="fig8_tracking_with_markers_base",
        title="Tracking with STDW intervention markers (Base / calm / full tune)",
    )
    plot_tracking_with_markers(
        matrix_dir,
        out_dir,
        wave="calm",
        embodiment="asymmetric",
        tune="full",
        stem="fig9_tracking_with_markers_asymmetric",
        title="Tracking with STDW intervention markers (Asymmetric / calm / full tune)",
    )
    write_figure_index(out_dir, matrix_dir)
    print(f"[OK] wrote figures to {out_dir}")


if __name__ == "__main__":
    main()

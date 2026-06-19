"""PPT-ready figures for the HKU SRP defense (Slide 9 / Slide 10).

This script does not re-implement data loading or metric math; it reuses the
helpers in ``plot_stdw_effect_matrix.py`` and only adds a PPT-oriented visual
layer (landscape aspect, compact height, clean spines) on top of the same
mainline A3 stage2 sweep data.

Figures produced:
  * fig21_tracking_timeline   -> Slide 9 centre (multi-channel tracking timeline)
  * fig22_generalization_heatmap -> Slide 9 left (48-cell 4x3 generalization heatmap)
  * fig23_opr_recovery        -> Slide 10 left (observable-only OPR recovery)

Data provenance:
  * Timeline + heatmap: ``.results/sweep_a3_stage2_20260608_142742``
  * OPR recovery numbers: ``docs/engineering/DIAG_p1_p2_p5_20260610.md`` (router
    summary_fmse averaged over waves).
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from plot_stdw_effect_matrix import (  # reuse, do not re-implement
    DEFAULT_MATRIX_DIR,
    EMB_LABELS,
    EMBODIMENTS,
    WAVE_LABELS,
    WAVES,
    _ensure_dir,
    _find_output_csv,
    _mean_pairwise,
    _read_csv_dicts,
    _read_numeric_csv,
    _tracking_mse,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT_DIR = REPO_ROOT / "docs" / "figures" / "ppt"
# Shared-reference (deterministic_reference) full matrix re-run. STDW on/off
# cells track an identical desired trajectory, so overlays are like-for-like.
DETREF_MATRIX_DIR = REPO_ROOT / ".results" / "sweep_a3_stage2_detref_20260619_223735"

# --- shared palette ---------------------------------------------------------
C_DESIRED = "#222222"
C_ON = "#1f5fbf"      # academic blue (Ours, STDW on)
C_OFF = "#9aa0a6"     # muted gray (STDW off baseline)
C_RHO = "#1f5fbf"
C_BIAS = "#8e44ad"    # purple gravity bias
C_GATE = "#2e86de"
C_TRIG = "#d62728"
C_GREEN = "#1a9850"

# OPR recovery numbers (asymmetric, storm, full tune, shared deterministic
# reference). Full-trajectory raw tracking-MSE so the bars are consistent with
# the fig26 OPR-rescue overlay annotations.
# Source: .results/sweep_a3_stage2_detref_20260619_223735 (blind on/off) +
# .results/opr_asym_storm_full_20260619_235828 (OPR rescue).
OPR_ASYM_DEFAULT = 0.2095   # asymmetric, blind STDW-on co-adaptation collapse
OPR_ASYM_ROUTED = 0.0953    # asymmetric, STDW-on + offset_correct router rescue
OPR_BASE_REF = 0.0897       # symmetric base (storm/full) STDW-on nominal level


def _style_axis(ax: plt.Axes) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", alpha=0.18, lw=0.7)
    ax.tick_params(labelsize=8)


def _save_ppt(fig: plt.Figure, out_path: Path) -> None:
    for suffix in (".png", ".pdf"):
        fig.savefig(out_path.with_suffix(suffix), dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"[OK] {out_path.with_suffix('.png')}")


# ---------------------------------------------------------------------------
# Figure 2.1 - multi-channel tracking timeline (Slide 9 centre)
# ---------------------------------------------------------------------------
def plot_tracking_timeline(
    matrix_dir: Path,
    out_dir: Path,
    wave: str = "storm",
    embodiment: str = "base",
    tune: str = "full",
    stem: str = "fig21_tracking_timeline",
) -> None:
    off = _read_numeric_csv(_find_output_csv(matrix_dir, wave, embodiment, tune, "off"))
    on = _read_numeric_csv(_find_output_csv(matrix_dir, wave, embodiment, tune, "on"))

    t = on.get("time_s", on["step"] / 60.0)
    t_off = off.get("time_s", off["step"] / 60.0)
    rho = on.get("rho", np.zeros_like(t))
    bias = on.get("domain_bias", np.zeros_like(t))
    mask = on.get("stdw_mask", np.full_like(t, np.nan))
    trig_t = _real_trigger_times(on, t)

    off_mse = float(np.nanmean(_tracking_mse(off)))
    on_mse = float(np.nanmean(_tracking_mse(on)))
    delta = (on_mse - off_mse) / off_mse * 100.0

    # Drift window is defined in control steps (200..1200). Map to the actual
    # time axis so the shading stays inside the recorded episode regardless of
    # the control rate.
    n = t.size
    drift_t0 = float(t[min(200, n - 1)])
    drift_t1 = float(t[min(1200, n - 1)])
    t_lo, t_hi = float(t[0]), float(t[-1])

    fig, axes = plt.subplots(
        6, 1, figsize=(9.6, 5.0), sharex=True,
        gridspec_kw={"height_ratios": [1, 1, 1, 1, 0.7, 0.55], "hspace": 0.18},
    )
    fig.subplots_adjust(top=0.88, bottom=0.085, left=0.10, right=0.985)
    ax_r, ax_p, ax_y, ax_d, ax_s, ax_g = axes

    track = [
        (ax_r, "Roll\n(rad)", "des_roll", "true_roll"),
        (ax_p, "Pitch\n(rad)", "des_pitch", "true_pitch"),
        (ax_y, "Yaw\n(rad)", "des_yaw", "true_yaw"),
        (ax_d, "Depth\nz (m)", "des_depth", "true_z"),
    ]
    for ax, ylabel, dcol, tcol in track:
        ax.axvspan(drift_t0, drift_t1, color="#4c78a8", alpha=0.06, zorder=0)
        ax.plot(t_off, off[tcol], color=C_OFF, lw=1.3, alpha=0.95, zorder=1)
        ax.plot(t, on[dcol], color=C_DESIRED, lw=1.3, ls="--", zorder=2)
        ax.plot(t, on[tcol], color=C_ON, lw=1.7, zorder=3)
        ax.set_ylabel(ylabel, fontsize=8.5, rotation=0, ha="right", va="center", labelpad=3)
        _style_axis(ax)

    fig.legend(
        handles=[
            plt.Line2D([], [], color=C_DESIRED, ls="--", lw=1.3, label="Desired (reference)"),
            plt.Line2D([], [], color=C_ON, lw=1.7, label="STDW On (Ours)"),
            plt.Line2D([], [], color=C_OFF, lw=1.3, label="STDW Off (nominal)"),
        ],
        frameon=False, loc="upper center", fontsize=8.5, ncol=3,
        bbox_to_anchor=(0.5, 0.955),
    )

    # Panel 5: STDW adaptive state (rho ramp + gravity bias)
    ax_s.axvspan(drift_t0, drift_t1, color="#4c78a8", alpha=0.06, zorder=0)
    ax_s.plot(t, rho, color=C_RHO, lw=1.8, label=r"$\varrho$ (STDW inject ratio)")
    ax_s.plot(t, bias, color=C_BIAS, lw=1.0, alpha=0.9, label="gravity bias")
    ax_s.set_ylabel("STDW\nstate", fontsize=8, rotation=0, ha="right", va="center", labelpad=3)
    ax_s.set_ylim(-0.05, 1.25)
    ax_s.legend(frameon=False, loc="upper left", fontsize=6.8, ncol=2, handlelength=1.4,
                columnspacing=1.0, borderaxespad=0.2)
    _style_axis(ax_s)

    # Panel 6: safety gate + slow-loop update markers
    ax_g.axvspan(drift_t0, drift_t1, color="#4c78a8", alpha=0.06, zorder=0)
    if np.any(np.isfinite(mask)):
        ax_g.fill_between(t, 0, mask, step="pre", color=C_GATE, alpha=0.30, lw=0)
        ax_g.plot(t, mask, color=C_GATE, lw=0.7, alpha=0.8, drawstyle="steps-pre")
    if trig_t.size:
        ax_g.scatter(trig_t, np.full_like(trig_t, 0.5), marker="v", s=26,
                     color=C_TRIG, zorder=5,
                     label=f"slow-loop update (n={trig_t.size})")
    ax_g.set_ylabel("Gate /\nupdate", fontsize=8, rotation=0, ha="right", va="center", labelpad=3)
    ax_g.set_ylim(-0.1, 1.35)
    ax_g.set_xlim(t_lo, t_hi)
    ax_g.set_xlabel("Time (s)", fontsize=9)
    ax_g.legend(frameon=False, loc="upper left", fontsize=6.8, borderaxespad=0.2)
    _style_axis(ax_g)

    # Drift-window label placed inside the band on the depth panel (free space).
    ax_d.text(0.5 * (drift_t0 + drift_t1), ax_d.get_ylim()[1],
              "Active Parametric Drift Window", ha="center", va="top",
              fontsize=7, color="#4c78a8", style="italic")

    fig.text(
        0.985, 0.985, f"$\\Delta$ MSE = {delta:+.1f}%",
        ha="right", va="top", fontsize=11, fontweight="bold", color=C_GREEN,
        bbox=dict(boxstyle="round,pad=0.3", facecolor="#e8f5e9",
                  edgecolor=C_GREEN, lw=1.2, alpha=0.92),
    )
    fig.suptitle(
        f"Multi-channel tracking under {WAVE_LABELS[wave]} sea state + COM drift "
        f"({EMB_LABELS[embodiment]} hull)",
        fontsize=10.5, y=0.995, x=0.10, ha="left",
    )
    _save_ppt(fig, out_dir / stem)


# ---------------------------------------------------------------------------
# Figure 2.2 - 48-cell generalization heatmap (Slide 9 left)
# ---------------------------------------------------------------------------
def plot_generalization_heatmap(
    pair_rows: List[Dict[str, str]],
    out_dir: Path,
    stem: str = "fig22_generalization_heatmap",
    provenance: str = "",
) -> None:
    values = np.zeros((len(EMBODIMENTS), len(WAVES)))
    for i, emb in enumerate(EMBODIMENTS):
        for j, wave in enumerate(WAVES):
            values[i, j] = _mean_pairwise(pair_rows, emb, wave)[2]

    fig, ax = plt.subplots(figsize=(11.0, 4.1))
    im = ax.imshow(values, cmap="RdBu_r", vmin=-150, vmax=150, aspect="auto")

    ax.set_xticks(np.arange(len(WAVES)))
    ax.set_xticklabels(
        ["Calm (hs=0.3m)", "Medium (hs=0.8m)", "Storm (hs=1.5m)"], fontsize=9
    )
    ax.set_yticks(np.arange(len(EMBODIMENTS)))
    ax.set_yticklabels(
        ["Base", "Long Body", "Heavy Moderate", "Asymmetric"], fontsize=9
    )
    ax.set_xticks(np.arange(len(WAVES) + 1) - 0.5, minor=True)
    ax.set_yticks(np.arange(len(EMBODIMENTS) + 1) - 0.5, minor=True)
    ax.grid(which="minor", color="white", lw=2.0)
    ax.tick_params(which="minor", length=0)

    for i in range(values.shape[0]):
        for j in range(values.shape[1]):
            v = values[i, j]
            color = "white" if abs(v) > 60 else "#1a1a1a"
            ax.text(j, i, f"{v:+.1f}%", ha="center", va="center",
                    color=color, fontsize=11, fontweight="bold")

    cbar = fig.colorbar(im, ax=ax, shrink=0.9, pad=0.02)
    cbar.set_label("STDW-induced tracking-error change (%)", fontsize=8.5)
    cbar.ax.tick_params(labelsize=8)
    ax.set_title("Cross-environment generalization of STDW (4 hulls x 3 sea states)",
                 fontsize=11, pad=8)
    fig.text(
        0.5, -0.02,
        "Negative (blue) = stable improvement. STDW robustly compresses tracking MSE on "
        "base & long-body hulls across all spectra,\nbut triggers co-adaptation collapse "
        "on asymmetric hulls without active routing (resolved by OPR on Slide 10).",
        ha="center", va="top", fontsize=8, color="#444444",
    )
    if provenance:
        fig.text(0.5, -0.16, provenance, ha="center", va="top",
                 fontsize=6.5, color="#888888", style="italic")
    _save_ppt(fig, out_dir / stem)


def _full_traj_delta(matrix_dir: Path, emb: str, wave: str) -> float:
    """Mean (over identity/full tune) full-trajectory raw tracking-MSE delta %.

    Uses the same ``_tracking_mse`` (per-step squared RPY+depth error) that the
    overlay figures label, so the heatmap FULL panel is numerically consistent
    with fig24/fig26/fig27 MSE annotations.
    """
    deltas = []
    for tune in ("identity", "full"):
        off = _read_numeric_csv(_find_output_csv(matrix_dir, wave, emb, tune, "off"))
        on = _read_numeric_csv(_find_output_csv(matrix_dir, wave, emb, tune, "on"))
        fo = float(np.nanmean(_tracking_mse(off)))
        fn = float(np.nanmean(_tracking_mse(on)))
        deltas.append((fn - fo) / fo * 100.0)
    return float(np.mean(deltas))


def _tail_delta(pair_rows: List[Dict[str, str]], emb: str, wave: str) -> float:
    """Project-standard post-drift delta %: 5s-RMS filtered compound error.

    Reads ``fmse_drift_{off,on}_mean`` (i.e. ``final_mse_after_drift`` averaged
    over identity/full), matching the headline -53.9% tail figure and every
    prior STDW report. This is the stationary-window metric after the drift ramp
    has fully settled (steps > drift_end_step).
    """
    sel = [r for r in pair_rows if r["embodiment"] == emb and r["wave"] == wave]
    off = float(np.mean([float(r["fmse_drift_off_mean"]) for r in sel]))
    on = float(np.mean([float(r["fmse_drift_on_mean"]) for r in sel]))
    return (on - off) / off * 100.0


def _draw_heat_panel(ax: plt.Axes, values: np.ndarray, title: str) -> "plt.cm.ScalarMappable":
    im = ax.imshow(values, cmap="RdBu_r", vmin=-150, vmax=150, aspect="auto")
    ax.set_xticks(np.arange(len(WAVES)))
    ax.set_xticklabels(["Calm\n(hs=0.3m)", "Medium\n(hs=0.8m)", "Storm\n(hs=1.5m)"], fontsize=8.5)
    ax.set_yticks(np.arange(len(EMBODIMENTS)))
    ax.set_yticklabels(["Base", "Long Body", "Heavy Moderate", "Asymmetric"], fontsize=8.5)
    ax.set_xticks(np.arange(len(WAVES) + 1) - 0.5, minor=True)
    ax.set_yticks(np.arange(len(EMBODIMENTS) + 1) - 0.5, minor=True)
    ax.grid(which="minor", color="white", lw=2.0)
    ax.tick_params(which="minor", length=0)
    for i in range(values.shape[0]):
        for j in range(values.shape[1]):
            v = values[i, j]
            color = "white" if abs(v) > 60 else "#1a1a1a"
            ax.text(j, i, f"{v:+.1f}%", ha="center", va="center",
                    color=color, fontsize=10, fontweight="bold")
    ax.set_title(title, fontsize=9.5, pad=6)
    return im


def plot_generalization_heatmap_dual(
    matrix_dir: Path,
    pair_rows: List[Dict[str, str]],
    out_dir: Path,
    stem: str = "fig22_generalization_heatmap",
    provenance: str = "",
) -> None:
    """Two-panel 48-cell heatmap: full-trajectory MSE and post-drift tail MSE.

    Left  = full-trajectory raw tracking-MSE delta (conservative, whole episode).
    Right = post-drift stationary-window filtered error delta (where STDW acts).
    Both panels share one diverging colourbar; the tail panel exposes the larger
    adaptation gain while the full panel keeps the honest whole-episode number.
    """
    full_vals = np.zeros((len(EMBODIMENTS), len(WAVES)))
    tail_vals = np.zeros((len(EMBODIMENTS), len(WAVES)))
    for i, emb in enumerate(EMBODIMENTS):
        for j, wave in enumerate(WAVES):
            full_vals[i, j] = _full_traj_delta(matrix_dir, emb, wave)
            tail_vals[i, j] = _tail_delta(pair_rows, emb, wave)

    fig, (ax_l, ax_r) = plt.subplots(1, 2, figsize=(13.4, 4.3))
    fig.subplots_adjust(left=0.075, right=0.9, top=0.82, bottom=0.16, wspace=0.28)
    _draw_heat_panel(ax_l, full_vals, "Full-trajectory tracking-MSE change")
    im = _draw_heat_panel(ax_r, tail_vals, "Post-drift tail error change (5s-RMS filtered)")
    ax_r.set_yticklabels([])

    cbar = fig.colorbar(im, ax=[ax_l, ax_r], shrink=0.85, pad=0.02)
    cbar.set_label("STDW-induced tracking-error change (%)", fontsize=8.5)
    cbar.ax.tick_params(labelsize=8)

    fig.suptitle("Cross-environment generalization of STDW (4 hulls x 3 sea states)",
                 fontsize=11.5, y=0.95)
    fig.text(
        0.49, 0.045,
        "Negative (blue) = improvement. STDW compresses tracking error on base / long-body / heavy hulls across all spectra "
        "(full-traj mean -21%, tail mean -42%),\nbut collapses on asymmetric hulls without active routing "
        "(resolved by OPR, Slide 10). Left = whole-episode raw MSE; right = stationary post-drift filtered error.",
        ha="center", va="top", fontsize=7.6, color="#444444",
    )
    if provenance:
        fig.text(0.49, -0.02, provenance, ha="center", va="top",
                 fontsize=6.5, color="#888888", style="italic")
    _save_ppt(fig, out_dir / stem)


# ---------------------------------------------------------------------------
# Figure 2.3 - observable-only OPR recovery (Slide 10 left)
# ---------------------------------------------------------------------------
def plot_opr_recovery(out_dir: Path, stem: str = "fig23_opr_recovery") -> None:
    plunge = (OPR_ASYM_DEFAULT - OPR_ASYM_ROUTED) / OPR_ASYM_DEFAULT * 100.0

    fig, ax = plt.subplots(figsize=(11.0, 3.7))
    xs = [0.0, 1.0]
    heights = [OPR_ASYM_DEFAULT, OPR_ASYM_ROUTED]
    colors = [C_OFF, C_GREEN]
    bars = ax.bar(xs, heights, width=0.42, color=colors, zorder=3)

    ax.set_xticks(xs)
    ax.set_xticklabels(
        ["Asymmetric, blind STDW-on\n(co-adaptation collapse)",
         "Asymmetric, STDW-on + OPR\n(offset_correct router)"],
        fontsize=9,
    )
    ax.set_ylabel("Full-trajectory tracking MSE (m$^2$)", fontsize=9)
    ax.set_ylim(0, OPR_ASYM_DEFAULT * 1.28)
    _style_axis(ax)

    for b, h in zip(bars, heights):
        ax.text(b.get_x() + b.get_width() / 2, h + 0.005, f"{h:.4f}",
                ha="center", va="bottom", fontsize=10, fontweight="bold")

    # base reference dashed line aligned to the routed bar
    ax.axhline(OPR_BASE_REF, color=C_GREEN, ls=":", lw=1.4, alpha=0.9, zorder=2)
    ax.text(-0.05, OPR_BASE_REF + 0.009, f"symmetric base ref = {OPR_BASE_REF:.4f}",
            va="bottom", ha="left", fontsize=8, color=C_GREEN)

    # downward plunge arrow from default bar top to routed bar top
    ax.annotate(
        "", xy=(0.74, OPR_ASYM_ROUTED + 0.012), xytext=(0.26, OPR_ASYM_DEFAULT - 0.004),
        arrowprops=dict(arrowstyle="-|>", color=C_GREEN, lw=3.0, alpha=0.55),
    )
    # callout placed in the free space above the short routed bar (clear of arrow)
    ax.text(1.30, OPR_ASYM_DEFAULT * 0.66,
            f"{plunge:.1f}% Tracking Error\nPlunge via OPR",
            ha="center", va="center", fontsize=11, fontweight="bold",
            color=C_GREEN,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="#e8f5e9",
                      edgecolor=C_GREEN, lw=1.2, alpha=0.9))

    ax.set_title("Observable-only OPR rescues asymmetric hulls to nominal level",
                 fontsize=11, pad=8)
    ax.set_xlim(-0.55, 2.05)
    _save_ppt(fig, out_dir / stem)


# ---------------------------------------------------------------------------
# Figure 2.4 - publication-grade 4-row tracking overlay (secondary prompt P1)
# ---------------------------------------------------------------------------
def _clean_overlay_axis(ax: plt.Axes) -> None:
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.spines["left"].set_color("#444444")
    ax.spines["bottom"].set_color("#444444")
    ax.grid(axis="y", color="#cccccc", alpha=0.45, lw=0.6)
    ax.set_axisbelow(True)
    ax.tick_params(labelsize=8, color="#444444")


def _real_trigger_times(data: Dict[str, np.ndarray], t: np.ndarray) -> np.ndarray:
    """Slow-loop update times taken from the ``loss`` column only.

    ``_detect_trigger_steps`` falls back to rho jumps when no finite loss is
    present, which on STDW-off cells produces a dense spurious cluster (~1000
    points) even though no slow-loop update ever fired. For marker rendering we
    require genuine finite-loss steps and otherwise draw nothing, so the online
    adaptation markers stay faithful to the run regardless of embodiment.
    """
    loss = data.get("loss")
    if loss is None:
        return np.empty(0)
    idx = np.where(np.isfinite(loss))[0]
    return t[idx] if idx.size else np.empty(0)


def plot_publication_overlay(
    matrix_dir: Path,
    out_dir: Path,
    wave: str = "storm",
    embodiment: str = "base",
    tune: str = "full",
    stem: str = "fig24_publication_overlay",
    on_csv: Path | None = None,
    off_csv: Path | None = None,
) -> None:
    """Clean publication-grade RPY+depth overlay (no legend box, leader text).

    Three curves per panel: Desired (charcoal dashed), STDW Off (muted gray),
    STDW On / Ours (classic blue). A faint blue band marks the parametric drift
    window; slow-loop update triggers are drawn as small blue down-triangles at
    the bottom of the depth panel only.

    When ``on_csv`` / ``off_csv`` are given they are read directly (used for the
    shared-reference re-run where on/off track an identical desired trajectory,
    so the attitude panels are genuinely comparable rather than overlaying two
    different randomized references).
    """
    off_path = off_csv if off_csv is not None else _find_output_csv(matrix_dir, wave, embodiment, tune, "off")
    on_path = on_csv if on_csv is not None else _find_output_csv(matrix_dir, wave, embodiment, tune, "on")
    off = _read_numeric_csv(off_path)
    on = _read_numeric_csv(on_path)

    t = on.get("time_s", on["step"] / 60.0)
    t_off = off.get("time_s", off["step"] / 60.0)
    trig_t = _real_trigger_times(on, t)

    off_mse = float(np.nanmean(_tracking_mse(off)))
    on_mse = float(np.nanmean(_tracking_mse(on)))

    n = t.size
    drift_t0 = float(t[min(200, n - 1)])
    drift_t1 = float(t[min(1200, n - 1)])
    t_lo, t_hi = float(t[0]), float(t[-1])

    C_REF = "#3a3a3a"     # charcoal dashed reference
    C_BASE = "#9aa0a6"    # muted gray baseline
    C_OURS = "#1f5fbf"    # classic blue ours

    fig, axes = plt.subplots(
        4, 1, figsize=(9.2, 5.4), sharex=True,
        gridspec_kw={"hspace": 0.30},
    )
    fig.subplots_adjust(top=0.9, bottom=0.1, left=0.085, right=0.83)

    panels = [
        (axes[0], "Roll (rad)", "des_roll", "true_roll"),
        (axes[1], "Pitch (rad)", "des_pitch", "true_pitch"),
        (axes[2], "Yaw (rad)", "des_yaw", "true_yaw"),
        (axes[3], "Depth z (m)", "des_depth", "true_z"),
    ]
    for ax, ylabel, dcol, tcol in panels:
        ax.axvspan(drift_t0, drift_t1, color="#4c78a8", alpha=0.08, zorder=0)
        ax.plot(t_off, off[tcol], color=C_BASE, lw=1.5, alpha=0.5, zorder=1)
        ax.plot(t, on[dcol], color=C_REF, lw=2.0, ls=(0, (5, 3)), zorder=2)
        ax.plot(t, on[tcol], color=C_OURS, lw=2.3, zorder=3)
        ax.set_ylabel(ylabel, fontsize=9)
        ax.set_xlim(t_lo, t_hi)
        _clean_overlay_axis(ax)

    # drift-window label (top panel, above the band)
    axes[0].text(0.5 * (drift_t0 + drift_t1), axes[0].get_ylim()[1],
                 "Active Parametric Drift Window", ha="center", va="bottom",
                 fontsize=7.5, color="#4c78a8", style="italic")

    # leader-text annotations in the right margin (no legend box)
    axes[0].annotate("Desired Reference", xy=(t_hi, on["des_roll"][-1]),
                     xytext=(t_hi + 0.6, axes[0].get_ylim()[1] * 0.7),
                     fontsize=8, color=C_REF, va="center",
                     arrowprops=dict(arrowstyle="-", color=C_REF, lw=0.8))
    axes[1].annotate(f"Static PPO Baseline\n(MSE = {off_mse:.3f})",
                     xy=(t_hi, off["true_pitch"][-1]),
                     xytext=(t_hi + 0.6, axes[1].get_ylim()[0] * 0.9),
                     fontsize=8, color="#6b7075", va="center", fontweight="bold",
                     arrowprops=dict(arrowstyle="-", color=C_BASE, lw=0.8))
    axes[2].annotate(f"Ours (STDW Active\nMSE = {on_mse:.3f})",
                     xy=(t_hi, on["true_yaw"][-1]),
                     xytext=(t_hi + 0.6, axes[2].get_ylim()[1] * 0.7),
                     fontsize=8, color=C_OURS, va="center", fontweight="bold",
                     arrowprops=dict(arrowstyle="-", color=C_OURS, lw=0.8))

    # slow-loop triggers on the depth panel baseline only
    ax_d = axes[3]
    y0 = ax_d.get_ylim()[0]
    if trig_t.size:
        ax_d.scatter(trig_t, np.full_like(trig_t, y0), marker="v", s=24,
                     color=C_OURS, clip_on=False, zorder=5)
        ax_d.annotate(f"Adaptation Triggers (n={trig_t.size})",
                      xy=(t_hi, y0), xytext=(t_hi + 0.6, y0),
                      fontsize=7.5, color=C_OURS, va="center")
    ax_d.set_xlabel("Time (seconds)", fontsize=9)

    fig.suptitle(
        f"Attitude-depth tracking overlay under {WAVE_LABELS[wave]} sea state + COM drift "
        f"({EMB_LABELS[embodiment]} hull)",
        fontsize=10.5, x=0.085, ha="left", y=0.985,
    )
    _save_ppt(fig, out_dir / stem)


# ---------------------------------------------------------------------------
# Figure 2.6 - OPR rescue overlay (three actual curves on a shared reference)
# ---------------------------------------------------------------------------
def plot_opr_rescue_overlay(
    off_csv: Path,
    on_csv: Path,
    opr_csv: Path,
    out_dir: Path,
    wave: str = "storm",
    embodiment: str = "asymmetric",
    stem: str = "fig26_opr_rescue_asym_storm_full",
) -> None:
    """Three-curve rescue overlay on a single shared desired reference.

    Curves per attitude/depth panel: static PPO baseline (gray), blind STDW-on
    co-adaptation collapse (orange), and STDW-on + OPR offset-correct router
    rescue (blue). A dedicated bottom strip renders the online-adaptation
    schedule (rho injection ramp + genuine slow-loop update triggers) so the
    markers read cleanly instead of being crowded against a tracking trace.

    The three runs share an identical desired trajectory (deterministic
    reference), so the panels are a fair like-for-like comparison.
    """
    off = _read_numeric_csv(off_csv)
    on = _read_numeric_csv(on_csv)
    opr = _read_numeric_csv(opr_csv)

    t = opr.get("time_s", opr["step"] / 60.0)
    t_off = off.get("time_s", off["step"] / 60.0)
    t_on = on.get("time_s", on["step"] / 60.0)
    rho = opr.get("rho", np.zeros_like(t))
    trig_t = _real_trigger_times(opr, t)

    off_mse = float(np.nanmean(_tracking_mse(off)))
    on_mse = float(np.nanmean(_tracking_mse(on)))
    opr_mse = float(np.nanmean(_tracking_mse(opr)))

    n = t.size
    drift_t0 = float(t[min(200, n - 1)])
    drift_t1 = float(t[min(1200, n - 1)])
    t_lo, t_hi = float(t[0]), float(t[-1])

    C_REF = "#3a3a3a"      # charcoal dashed reference
    C_BASE = "#9aa0a6"     # muted gray static baseline
    C_COLLAPSE = "#e06c00"  # orange blind-STDW collapse
    C_RESCUE = "#1f5fbf"   # blue STDW + OPR rescue
    C_TRIG = "#d62728"

    fig, axes = plt.subplots(
        5, 1, figsize=(9.4, 6.2), sharex=True,
        gridspec_kw={"height_ratios": [1, 1, 1, 1, 0.7], "hspace": 0.30},
    )
    fig.subplots_adjust(top=0.88, bottom=0.085, left=0.085, right=0.80)

    panels = [
        (axes[0], "Roll (rad)", "des_roll", "true_roll"),
        (axes[1], "Pitch (rad)", "des_pitch", "true_pitch"),
        (axes[2], "Yaw (rad)", "des_yaw", "true_yaw"),
        (axes[3], "Depth z (m)", "des_depth", "true_z"),
    ]
    for ax, ylabel, dcol, tcol in panels:
        ax.axvspan(drift_t0, drift_t1, color="#4c78a8", alpha=0.08, zorder=0)
        ax.plot(t, opr[dcol], color=C_REF, lw=2.0, ls=(0, (5, 3)), zorder=2)
        ax.plot(t_off, off[tcol], color=C_BASE, lw=1.4, alpha=0.6, zorder=1)
        ax.plot(t_on, on[tcol], color=C_COLLAPSE, lw=1.7, alpha=0.9, zorder=3)
        ax.plot(t, opr[tcol], color=C_RESCUE, lw=2.2, zorder=4)
        ax.set_ylabel(ylabel, fontsize=9)
        ax.set_xlim(t_lo, t_hi)
        _clean_overlay_axis(ax)

    axes[0].text(0.5 * (drift_t0 + drift_t1), axes[0].get_ylim()[1],
                 "Active Parametric Drift Window", ha="center", va="bottom",
                 fontsize=7.5, color="#4c78a8", style="italic")

    # leader-text annotations in the right margin (no legend box)
    axes[0].annotate("Desired Reference", xy=(t_hi, opr["des_roll"][-1]),
                     xytext=(t_hi + 0.6, axes[0].get_ylim()[1] * 0.7),
                     fontsize=8, color=C_REF, va="center",
                     arrowprops=dict(arrowstyle="-", color=C_REF, lw=0.8))
    axes[1].annotate(f"Static PPO Baseline\n(MSE = {off_mse:.3f})",
                     xy=(t_hi, off["true_pitch"][-1]),
                     xytext=(t_hi + 0.6, axes[1].get_ylim()[0] * 0.9),
                     fontsize=8, color="#6b7075", va="center", fontweight="bold",
                     arrowprops=dict(arrowstyle="-", color=C_BASE, lw=0.8))
    axes[2].annotate(f"Blind STDW Collapse\n(MSE = {on_mse:.3f})",
                     xy=(t_hi, on["true_yaw"][-1]),
                     xytext=(t_hi + 0.6, axes[2].get_ylim()[1] * 0.75),
                     fontsize=8, color=C_COLLAPSE, va="center", fontweight="bold",
                     arrowprops=dict(arrowstyle="-", color=C_COLLAPSE, lw=0.8))
    axes[3].annotate(f"STDW + OPR Rescue\n(MSE = {opr_mse:.3f})",
                     xy=(t_hi, opr["true_z"][-1]),
                     xytext=(t_hi + 0.6, axes[3].get_ylim()[1] * 0.7),
                     fontsize=8, color=C_RESCUE, va="center", fontweight="bold",
                     arrowprops=dict(arrowstyle="-", color=C_RESCUE, lw=0.8))

    # bottom strip: online-adaptation schedule (rho ramp + slow-loop triggers)
    ax_a = axes[4]
    ax_a.axvspan(drift_t0, drift_t1, color="#4c78a8", alpha=0.08, zorder=0)
    ax_a.plot(t, rho, color=C_RESCUE, lw=1.8, zorder=2,
              label=r"$\varrho$ STDW inject ratio")
    ax_a.set_ylim(-0.08, 1.18)
    ax_a.set_ylabel("Adaptation", fontsize=9)
    ax_a.set_xlabel("Time (seconds)", fontsize=9)
    ax_a.set_xlim(t_lo, t_hi)
    _clean_overlay_axis(ax_a)
    if trig_t.size:
        ax_a.scatter(trig_t, np.full_like(trig_t, 1.02), marker="v", s=30,
                     color=C_TRIG, edgecolor="white", linewidth=0.4,
                     clip_on=False, zorder=5)
        ax_a.annotate(f"slow-loop updates (n={trig_t.size})",
                      xy=(t_hi, 1.02), xytext=(t_hi + 0.6, 1.02),
                      fontsize=7.5, color=C_TRIG, va="center")
    ax_a.annotate(r"$\varrho$ ramp", xy=(t_hi, rho[-1]),
                  xytext=(t_hi + 0.6, 0.35),
                  fontsize=7.5, color=C_RESCUE, va="center",
                  arrowprops=dict(arrowstyle="-", color=C_RESCUE, lw=0.8))

    fig.suptitle(
        f"OPR rescues the {EMB_LABELS[embodiment]} hull from blind-STDW collapse "
        f"({WAVE_LABELS[wave]} sea state, shared reference)",
        fontsize=10.5, x=0.085, ha="left", y=0.985,
    )
    _save_ppt(fig, out_dir / stem)


# ---------------------------------------------------------------------------
# Figure 2.7 - online recovery of a mis-set initial controller (side study)
# ---------------------------------------------------------------------------
MISSET_ROOT = REPO_ROOT / ".results" / "stdw_online_misset_20260620"


def _latest_misset_csv(cell: str) -> Path:
    """Newest stdw_output.csv for a mis-set side-experiment cell."""
    base = MISSET_ROOT / cell / "results"
    cands = sorted(base.rglob("stdw_output.csv"), key=lambda p: p.stat().st_mtime)
    if not cands:
        raise FileNotFoundError(f"no stdw_output.csv under {base}")
    return cands[-1]


def plot_misset_online_overlay(
    out_dir: Path,
    embodiment: str = "base",
    stem: str | None = None,
) -> None:
    """Online recovery of a deliberately mis-set initial controller.

    The depth proportional gain is halved (depth_zeta1 x0.5, the 1/2x boundary
    of the requested 1/2x..2x band) and the *unchanged* stage2 checkpoint is run
    with STDW off vs on under a shared storm reference. STDW's slow loop adapts
    exactly the zeta1 column, so it can re-grow the under-damped depth gain
    online -- a more realistic "wrong gains at deployment" tracking scenario.
    """
    cell_off = f"{embodiment}_off"
    cell_on = f"{embodiment}_on"
    off = _read_numeric_csv(_latest_misset_csv(cell_off))
    on = _read_numeric_csv(_latest_misset_csv(cell_on))

    t = on.get("time_s", on["step"] / 60.0)
    t_off = off.get("time_s", off["step"] / 60.0)
    trig_t = _real_trigger_times(on, t)

    off_mse = float(np.nanmean(_tracking_mse(off)))
    on_mse = float(np.nanmean(_tracking_mse(on)))

    n = t.size
    drift_t0 = float(t[min(200, n - 1)])
    drift_t1 = float(t[min(1200, n - 1)])
    t_lo, t_hi = float(t[0]), float(t[-1])

    C_REF = "#3a3a3a"
    C_BASE = "#d1495b"    # red-ish: mis-set static baseline
    C_OURS = "#1f5fbf"    # blue: STDW online recovery

    fig, axes = plt.subplots(
        4, 1, figsize=(9.2, 5.4), sharex=True,
        gridspec_kw={"hspace": 0.30},
    )
    fig.subplots_adjust(top=0.9, bottom=0.1, left=0.085, right=0.82)

    panels = [
        (axes[0], "Roll (rad)", "des_roll", "true_roll"),
        (axes[1], "Pitch (rad)", "des_pitch", "true_pitch"),
        (axes[2], "Yaw (rad)", "des_yaw", "true_yaw"),
        (axes[3], "Depth z (m)", "des_depth", "true_z"),
    ]
    for ax, ylabel, dcol, tcol in panels:
        ax.axvspan(drift_t0, drift_t1, color="#4c78a8", alpha=0.08, zorder=0)
        ax.plot(t_off, off[tcol], color=C_BASE, lw=1.6, alpha=0.85, zorder=1)
        ax.plot(t, on[dcol], color=C_REF, lw=2.0, ls=(0, (5, 3)), zorder=2)
        ax.plot(t, on[tcol], color=C_OURS, lw=2.3, zorder=3)
        ax.set_ylabel(ylabel, fontsize=9)
        ax.set_xlim(t_lo, t_hi)
        _clean_overlay_axis(ax)

    axes[0].text(0.5 * (drift_t0 + drift_t1), axes[0].get_ylim()[1],
                 "Active Parametric Drift Window", ha="center", va="bottom",
                 fontsize=7.5, color="#4c78a8", style="italic")

    axes[0].annotate("Desired Reference", xy=(t_hi, on["des_roll"][-1]),
                     xytext=(t_hi + 0.6, axes[0].get_ylim()[1] * 0.7),
                     fontsize=8, color=C_REF, va="center",
                     arrowprops=dict(arrowstyle="-", color=C_REF, lw=0.8))
    axes[1].annotate(f"Mis-set Static PPO\n(depth P x0.5, MSE = {off_mse:.3f})",
                     xy=(t_hi, off["true_pitch"][-1]),
                     xytext=(t_hi + 0.6, axes[1].get_ylim()[0] * 0.9),
                     fontsize=8, color=C_BASE, va="center", fontweight="bold",
                     arrowprops=dict(arrowstyle="-", color=C_BASE, lw=0.8))
    axes[2].annotate(f"STDW Online Recovery\n(MSE = {on_mse:.3f})",
                     xy=(t_hi, on["true_yaw"][-1]),
                     xytext=(t_hi + 0.6, axes[2].get_ylim()[1] * 0.7),
                     fontsize=8, color=C_OURS, va="center", fontweight="bold",
                     arrowprops=dict(arrowstyle="-", color=C_OURS, lw=0.8))

    ax_d = axes[3]
    y0 = ax_d.get_ylim()[0]
    if trig_t.size:
        ax_d.scatter(trig_t, np.full_like(trig_t, y0), marker="v", s=24,
                     color=C_OURS, clip_on=False, zorder=5)
        ax_d.annotate(f"Adaptation Triggers (n={trig_t.size})",
                      xy=(t_hi, y0), xytext=(t_hi + 0.6, y0),
                      fontsize=7.5, color=C_OURS, va="center")
    ax_d.set_xlabel("Time (seconds)", fontsize=9)

    fig.suptitle(
        f"STDW recovers a mis-set initial controller online "
        f"(depth P gain x0.5, {EMB_LABELS[embodiment]} hull, storm sea state)",
        fontsize=10.5, x=0.085, ha="left", y=0.985,
    )
    out_stem = stem or f"fig28_misset_online_{embodiment}"
    _save_ppt(fig, out_dir / out_stem)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--matrix_dir", type=Path, default=DETREF_MATRIX_DIR)
    parser.add_argument("--out_dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--timeline_wave", default="storm")
    parser.add_argument("--timeline_embodiment", default="base")
    parser.add_argument("--timeline_tune", default="full")
    parser.add_argument(
        "--overlay_on_csv", type=Path, default=None,
        help="Explicit stdw_output.csv for the STDW-on curve in fig2.4 "
        "(shared-reference re-run). If omitted, the cell from --matrix_dir is used.",
    )
    parser.add_argument(
        "--overlay_off_csv", type=Path, default=None,
        help="Explicit stdw_output.csv for the STDW-off curve in fig2.4.",
    )
    parser.add_argument(
        "--misset_only", action="store_true",
        help="Only (re)generate the mis-set online-recovery side figures "
        "(fig28_*) without touching the validated fig21-24.",
    )
    args = parser.parse_args()

    matrix_dir = args.matrix_dir.resolve()
    out_dir = args.out_dir.resolve()
    _ensure_dir(out_dir)

    if args.misset_only:
        for emb in ("base", "heavy_moderate"):
            plot_misset_online_overlay(out_dir, embodiment=emb)
        print(f"[DONE] mis-set side figures written to {out_dir}")
        return

    pair_rows = _read_csv_dicts(matrix_dir / "stdw_pairwise.csv")

    plot_tracking_timeline(
        matrix_dir, out_dir,
        wave=args.timeline_wave,
        embodiment=args.timeline_embodiment,
        tune=args.timeline_tune,
    )
    heatmap_provenance = (
        f"Source: {matrix_dir.name} | policy model_2398 (stage2 2026-06-08), "
        "deterministic_reference shared-trajectory re-run. Full = whole-episode raw "
        "tracking-MSE; Tail = post-drift (step>1200) 5s-RMS filtered error "
        "(final_mse_after_drift), the project-standard adaptation metric."
    )
    plot_generalization_heatmap_dual(
        matrix_dir, pair_rows, out_dir, provenance=heatmap_provenance
    )
    plot_opr_recovery(out_dir)
    plot_publication_overlay(
        matrix_dir, out_dir,
        wave=args.timeline_wave,
        embodiment=args.timeline_embodiment,
        tune=args.timeline_tune,
        on_csv=args.overlay_on_csv.resolve() if args.overlay_on_csv else None,
        off_csv=args.overlay_off_csv.resolve() if args.overlay_off_csv else None,
    )
    for emb in ("base", "heavy_moderate"):
        plot_misset_online_overlay(out_dir, embodiment=emb)
    print(f"[DONE] PPT figures written to {out_dir}")


if __name__ == "__main__":
    main()

"""Cross-run comparison figures for the STDW safety-pressure matrix (M6).

Consumes the ``pressure_runs.csv`` produced by
``workflows/sweep_stdw_safety_pressure.py`` and renders variant-centric
comparison figures so different Lyapunov-V definitions / direction-guard modes
can be compared side by side across groups:

* ``fig_final_mse_by_variant.{png,pdf}`` — grouped bars of mean final MSE per
  (group, variant).
* ``fig_delta_vs_off_heatmap.{png,pdf}`` — mean ``delta_vs_off_pct`` as a
  variant × group heatmap (negative = STDW-on beats matched off_clean).
* ``fig_safety_metrics_by_variant.{png,pdf}`` — mean Lyapunov block count and
  STDW update-reject count per variant.

Isaac-independent: only numpy + matplotlib + stdlib csv are required.  Every
figure degrades gracefully — missing columns / empty groups are skipped rather
than raising, so the tool stays usable while the M6 sweep is only partially
populated.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]

# Stable variant ordering + colours (extra variants fall back to the cycle).
VARIANT_ORDER = [
    "off_clean",
    "stdw_default",
    "stdw_batch_trust",
    "lyap_strict",
    "lyap_guard_zero",
]
VARIANT_COLORS = {
    "off_clean": "#9e9e9e",
    "stdw_default": "#4c78a8",
    "stdw_batch_trust": "#1a9850",
    "lyap_strict": "#ff7f0e",
    "lyap_guard_zero": "#9467bd",
}
_FALLBACK_CYCLE = ["#17becf", "#d62728", "#8c564b", "#e377c2", "#bcbd22", "#7f7f7f"]


def _read_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        raise FileNotFoundError(path)
    with path.open("r", newline="", encoding="utf-8") as fp:
        return list(csv.DictReader(fp))


def _to_float(value: Any) -> float:
    if value is None or value == "":
        return float("nan")
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def _ordered_variants(rows: list[dict[str, str]]) -> list[str]:
    present = {r.get("variant", "") for r in rows if r.get("variant")}
    ordered = [v for v in VARIANT_ORDER if v in present]
    ordered += sorted(v for v in present if v not in VARIANT_ORDER)
    return ordered


def _ordered_groups(rows: list[dict[str, str]]) -> list[str]:
    seen: list[str] = []
    for r in rows:
        g = r.get("group", "")
        if g and g not in seen:
            seen.append(g)
    return seen


def _variant_color(variant: str, fallback_idx: int) -> str:
    if variant in VARIANT_COLORS:
        return VARIANT_COLORS[variant]
    return _FALLBACK_CYCLE[fallback_idx % len(_FALLBACK_CYCLE)]


def _mean_metric(rows: list[dict[str, str]], group: str, variant: str, key: str) -> float:
    vals = [
        _to_float(r.get(key))
        for r in rows
        if r.get("group") == group and r.get("variant") == variant
    ]
    vals = [v for v in vals if np.isfinite(v)]
    return float(np.mean(vals)) if vals else float("nan")


def _save(fig: plt.Figure, out_path: Path) -> None:
    fig.tight_layout()
    for suffix in (".png", ".pdf"):
        fig.savefig(out_path.with_suffix(suffix), dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_final_mse_by_variant(rows: list[dict[str, str]], out_dir: Path) -> bool:
    groups = _ordered_groups(rows)
    variants = _ordered_variants(rows)
    if not groups or not variants:
        return False
    x = np.arange(len(groups))
    width = min(0.8 / max(len(variants), 1), 0.22)
    fig, ax = plt.subplots(figsize=(max(8.0, 1.6 * len(groups)), 5.0))
    drew = False
    for i, variant in enumerate(variants):
        means = [_mean_metric(rows, g, variant, "final_mse") for g in groups]
        if not any(np.isfinite(m) for m in means):
            continue
        offset = (i - (len(variants) - 1) / 2.0) * width
        plot_means = [0.0 if not np.isfinite(m) else m for m in means]
        ax.bar(x + offset, plot_means, width, label=variant, color=_variant_color(variant, i))
        drew = True
    if not drew:
        plt.close(fig)
        return False
    ax.set_xticks(x)
    ax.set_xticklabels(groups, rotation=20, ha="right", fontsize=9)
    ax.set_ylabel("Mean final MSE")
    ax.set_title("STDW safety-pressure: final MSE by group × variant")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False, ncol=min(len(variants), 4), fontsize=8)
    _save(fig, out_dir / "fig_final_mse_by_variant")
    return True


def plot_delta_vs_off_heatmap(rows: list[dict[str, str]], out_dir: Path) -> bool:
    groups = _ordered_groups(rows)
    # off_clean has no delta (it is the reference), so drop it from the columns.
    variants = [v for v in _ordered_variants(rows) if v != "off_clean"]
    if not groups or not variants:
        return False
    values = np.full((len(variants), len(groups)), np.nan, dtype=float)
    for i, variant in enumerate(variants):
        for j, group in enumerate(groups):
            values[i, j] = _mean_metric(rows, group, variant, "delta_vs_off_pct")
    if not np.any(np.isfinite(values)):
        return False
    finite_abs = np.abs(values[np.isfinite(values)])
    vmax = float(np.max(finite_abs)) if finite_abs.size else 1.0
    vmax = max(vmax, 1.0)
    fig, ax = plt.subplots(figsize=(max(7.0, 1.5 * len(groups)), max(4.0, 0.7 * len(variants) + 2)))
    im = ax.imshow(values, cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="auto")
    ax.set_xticks(np.arange(len(groups)))
    ax.set_xticklabels(groups, rotation=25, ha="right", fontsize=9)
    ax.set_yticks(np.arange(len(variants)))
    ax.set_yticklabels(variants, fontsize=9)
    ax.set_title("STDW effect vs matched off_clean (Δ final MSE %)")
    for i in range(values.shape[0]):
        for j in range(values.shape[1]):
            if not np.isfinite(values[i, j]):
                ax.text(j, i, "—", ha="center", va="center", color="#888888", fontsize=9)
                continue
            color = "white" if abs(values[i, j]) > 0.7 * vmax else "black"
            ax.text(j, i, f"{values[i, j]:+.1f}", ha="center", va="center", color=color, fontsize=8)
    cbar = fig.colorbar(im, ax=ax, shrink=0.85)
    cbar.set_label("Δ final MSE vs off_clean (%)")
    ax.text(
        0.5, -0.28,
        "Negative (blue) = STDW-on lowers MSE vs matched off_clean; positive (red) = degradation.",
        transform=ax.transAxes, ha="center", va="top", fontsize=8,
    )
    _save(fig, out_dir / "fig_delta_vs_off_heatmap")
    return True


def plot_safety_metrics_by_variant(rows: list[dict[str, str]], out_dir: Path) -> bool:
    variants = _ordered_variants(rows)
    if not variants:
        return False

    def _variant_mean(variant: str, key: str) -> float:
        vals = [_to_float(r.get(key)) for r in rows if r.get("variant") == variant]
        vals = [v for v in vals if np.isfinite(v)]
        return float(np.mean(vals)) if vals else float("nan")

    blocks = [_variant_mean(v, "lyapunov_block_count") for v in variants]
    rejects = [_variant_mean(v, "stdw_update_rejected_count") for v in variants]
    if not any(np.isfinite(b) for b in blocks) and not any(np.isfinite(r) for r in rejects):
        return False
    x = np.arange(len(variants))
    width = 0.38
    fig, ax = plt.subplots(figsize=(max(7.0, 1.4 * len(variants)), 4.6))
    ax.bar(x - width / 2, [0.0 if not np.isfinite(b) else b for b in blocks], width,
           label="Lyapunov block count", color="#8c564b")
    ax.bar(x + width / 2, [0.0 if not np.isfinite(r) else r for r in rejects], width,
           label="STDW update rejects", color="#d62728")
    ax.set_xticks(x)
    ax.set_xticklabels(variants, rotation=20, ha="right", fontsize=9)
    ax.set_ylabel("Mean count per run")
    ax.set_title("Safety-gate activity by variant")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False, ncol=2, fontsize=9)
    _save(fig, out_dir / "fig_safety_metrics_by_variant")
    return True


def _find_default_runs_csv() -> Path | None:
    results_root = REPO_ROOT / ".results"
    if not results_root.exists():
        return None
    candidates = sorted(
        results_root.glob("stdw_safety_pressure_*/pressure_runs.csv"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    return candidates[0] if candidates else None


def generate(runs_csv: Path, out_dir: Path) -> list[str]:
    rows = _read_rows(runs_csv)
    out_dir.mkdir(parents=True, exist_ok=True)
    written: list[str] = []
    if plot_final_mse_by_variant(rows, out_dir):
        written.append("fig_final_mse_by_variant")
    if plot_delta_vs_off_heatmap(rows, out_dir):
        written.append("fig_delta_vs_off_heatmap")
    if plot_safety_metrics_by_variant(rows, out_dir):
        written.append("fig_safety_metrics_by_variant")
    return written


def main() -> int:
    parser = argparse.ArgumentParser(description="Cross-run figures for the STDW safety-pressure matrix.")
    parser.add_argument("--runs_csv", type=Path, default=None,
                        help="Path to pressure_runs.csv (default: latest under .results/).")
    parser.add_argument("--out_dir", type=Path, default=None,
                        help="Output directory (default: <runs_csv parent>/paper_figures/safety_pressure).")
    args = parser.parse_args()

    runs_csv = (args.runs_csv or _find_default_runs_csv())
    if runs_csv is None:
        print("[ERROR] no pressure_runs.csv found; pass --runs_csv explicitly.")
        return 1
    runs_csv = Path(runs_csv).resolve()
    out_dir = (args.out_dir or runs_csv.parent / "paper_figures" / "safety_pressure").resolve()
    written = generate(runs_csv, out_dir)
    if not written:
        print(f"[WARN] no figures produced from {runs_csv} (empty / missing metric columns).")
        return 0
    print(f"[OK] wrote {len(written)} figure(s) to {out_dir}: {', '.join(written)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

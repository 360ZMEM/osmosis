"""Compare tracking ripple across multiple play runs.

Walks one or more run roots, finds every stdw_output.csv beneath them,
runs analyze_tracking_ripple on each, and prints a comparison table sorted
by run timestamp. Useful for sweeping fixes (B1, D1, A3, A1, ...) and
spotting regressions at a glance.

Usage:
    python workflows/tools/compare_runs_ripple.py <run_root_1> [<run_root_2> ...]
                                                  [--ctrl_hz 60] [--smooth_window 31]
                                                  [--label_pattern "{stem}"]

Each <run_root> can be either a directory containing stdw_output.csv directly
or any ancestor (the script recurses).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

# allow `python workflows/tools/compare_runs_ripple.py` from repo root
sys.path.insert(0, str(Path(__file__).resolve().parent))
from analyze_tracking_ripple import analyze  # noqa: E402


def _find_csvs(root: Path) -> list[Path]:
    if root.is_file() and root.name == "stdw_output.csv":
        return [root]
    return sorted(root.rglob("stdw_output.csv"))


def _short_label(csv_path: Path, root: Path) -> str:
    try:
        rel = csv_path.relative_to(root)
    except ValueError:
        rel = csv_path
    return str(rel.parent)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("roots", nargs="+", type=Path)
    p.add_argument("--ctrl_hz", type=float, default=60.0)
    p.add_argument("--smooth_window", type=int, default=31)
    args = p.parse_args()

    rows: list[tuple[str, dict]] = []
    for root in args.roots:
        for csv in _find_csvs(root):
            label = _short_label(csv, root)
            try:
                rep = analyze(csv, args.ctrl_hz, args.smooth_window)
            except Exception as exc:
                print(f"[WARN] skip {csv}: {exc}", file=sys.stderr)
                continue
            rows.append((label, rep))

    if not rows:
        print("[ERR] no stdw_output.csv found", file=sys.stderr)
        sys.exit(1)

    rows.sort(key=lambda kv: kv[0])
    width = max(len(r[0]) for r in rows)
    header = (
        f"{'run':<{width}}  "
        f"{'roll_MSE':>10} {'roll_std':>9} {'roll_fft':>8}  "
        f"{'pitch_MSE':>10} {'pitch_std':>9} {'pitch_fft':>9}  "
        f"{'yaw_MSE':>10} {'yaw_std':>9} {'yaw_fft':>8}"
    )
    print(header)
    print("-" * len(header))
    for label, rep in rows:
        ax = rep["axes"]
        print(
            f"{label:<{width}}  "
            f"{ax['roll']['mse']:10.4e} {ax['roll']['ripple_std']:9.4f} {ax['roll']['fft_peak_freq_hz']:8.2f}  "
            f"{ax['pitch']['mse']:10.4e} {ax['pitch']['ripple_std']:9.4f} {ax['pitch']['fft_peak_freq_hz']:9.2f}  "
            f"{ax['yaw']['mse']:10.4e} {ax['yaw']['ripple_std']:9.4f} {ax['yaw']['fft_peak_freq_hz']:8.2f}"
        )


if __name__ == "__main__":
    main()

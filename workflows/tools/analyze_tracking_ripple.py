"""Tracking ripple analysis for stdw_output.csv.

Quantifies the high-frequency ripple superimposed on the slow sinusoidal
reference using a detrend + zero-crossing approach plus FFT dominant-peak
extraction. Produces a per-axis report and a JSON for run-to-run comparison.

Designed for the Roll/Yaw ripple debugging loop documented in
docs/控制器稳定调节记录.md (sections 6 onwards).

Usage:
    python workflows/tools/analyze_tracking_ripple.py <stdw_output.csv> [--out report.json]
                                                                       [--ctrl_hz 60]
                                                                       [--smooth_window 31]

The CSV is expected to expose des_{roll,pitch,yaw} and true_{roll,pitch,yaw}
columns (produced by play_stdw_adapt.py).
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np


AXES = ("roll", "pitch", "yaw")


def _load_columns(csv_path: Path) -> dict[str, np.ndarray]:
    with csv_path.open() as fp:
        rows = list(csv.DictReader(fp))
    cols: dict[str, np.ndarray] = {}
    for ax in AXES:
        for prefix in ("des_", "true_"):
            key = prefix + ax
            cols[key] = np.array([float(r[key]) for r in rows], dtype=float)
    return cols


def _ripple_stats(err: np.ndarray, ctrl_dt: float, smooth_window: int) -> dict:
    half = max(smooth_window // 2, 1)
    kern = np.ones(2 * half + 1) / (2 * half + 1)
    trend = np.convolve(err, kern, mode="same")
    ripple = (err - trend)[half:-half]

    zc = int(np.sum(np.diff(np.sign(ripple)) != 0) / 2)
    duration = len(ripple) * ctrl_dt
    zc_freq = zc / duration if duration > 0 else 0.0

    n = len(ripple)
    if n >= 16:
        spectrum = np.abs(np.fft.rfft(ripple - ripple.mean()))
        freqs = np.fft.rfftfreq(n, d=ctrl_dt)
        if len(spectrum) > 1:
            peak = int(np.argmax(spectrum[1:]) + 1)
            fft_freq = float(freqs[peak])
            fft_power = float(spectrum[peak])
        else:
            fft_freq = 0.0
            fft_power = 0.0
    else:
        fft_freq = 0.0
        fft_power = 0.0

    return {
        "ripple_std": float(ripple.std()),
        "ripple_p2p": float(ripple.max() - ripple.min()),
        "zero_cross_freq_hz": float(zc_freq),
        "fft_peak_freq_hz": fft_freq,
        "fft_peak_power": fft_power,
        "samples": int(n),
        "duration_s": float(duration),
    }


def analyze(csv_path: Path, ctrl_hz: float, smooth_window: int) -> dict:
    cols = _load_columns(csv_path)
    ctrl_dt = 1.0 / ctrl_hz
    report: dict = {
        "csv": str(csv_path),
        "ctrl_hz": ctrl_hz,
        "smooth_window": smooth_window,
        "axes": {},
    }
    for ax in AXES:
        err = cols["true_" + ax] - cols["des_" + ax]
        stats = _ripple_stats(err, ctrl_dt, smooth_window)
        stats["mse"] = float(np.mean(err ** 2))
        report["axes"][ax] = stats
    return report


def _print_human(report: dict) -> None:
    print(f"# tracking ripple report  csv={report['csv']}")
    print(f"  ctrl_hz={report['ctrl_hz']}  smooth_window={report['smooth_window']}")
    header = f"  {'axis':6s} {'MSE':>10s} {'ripple_std':>10s} {'p2p':>8s} {'zc_freq':>9s} {'fft_freq':>9s}"
    print(header)
    for ax, s in report["axes"].items():
        print(
            f"  {ax:6s} {s['mse']:10.4e} {s['ripple_std']:10.4f} "
            f"{s['ripple_p2p']:8.4f} {s['zero_cross_freq_hz']:9.2f} {s['fft_peak_freq_hz']:9.2f}"
        )


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("csv", type=Path, help="path to stdw_output.csv")
    p.add_argument("--out", type=Path, default=None, help="optional JSON output")
    p.add_argument("--ctrl_hz", type=float, default=60.0, help="control loop rate (Hz)")
    p.add_argument(
        "--smooth_window",
        type=int,
        default=31,
        help="moving-average window (samples) used for trend removal",
    )
    args = p.parse_args()

    if not args.csv.is_file():
        print(f"[ERR] csv not found: {args.csv}", file=sys.stderr)
        sys.exit(1)

    report = analyze(args.csv, args.ctrl_hz, args.smooth_window)
    _print_human(report)

    if args.out is not None:
        args.out.write_text(json.dumps(report, indent=2))
        print(f"[INFO] wrote {args.out}")


if __name__ == "__main__":
    main()

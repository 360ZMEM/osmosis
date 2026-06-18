"""Observable-only dynamic parameter family probe.

This module intentionally consumes only logged response rows: errors, actions,
and control effort. It does not read true env-side density, thruster fault,
thruster angle, or torque pulse state.
"""

from __future__ import annotations

import ast
import json
import math
from dataclasses import dataclass
from statistics import mean, median, pstdev
from typing import Iterable


FAMILIES = ("density", "thruster_efficiency", "thruster_angle")


def _f(row: dict, key: str, default: float = 0.0) -> float:
    try:
        value = row.get(key, default)
        if value in ("", None):
            return default
        return float(value)
    except Exception:
        return default


def _abs_error(row: dict, axis: str) -> float:
    if axis == "depth":
        return abs(_f(row, "des_depth") - _f(row, "true_z"))
    return abs(_f(row, f"des_{axis}") - _f(row, f"true_{axis}"))


def _actions(row: dict) -> list[float]:
    raw = row.get("executed_action", "[]")
    if isinstance(raw, list):
        values = raw
    else:
        try:
            values = ast.literal_eval(str(raw))
        except Exception:
            values = []
    out = []
    for value in values:
        try:
            out.append(float(value))
        except Exception:
            pass
    return out


def _window(rows: Iterable[dict], start: int, end: int) -> list[dict]:
    out = []
    for row in rows:
        step = int(_f(row, "step", -1.0))
        if start <= step < end:
            out.append(row)
    return out


def _avg(rows: list[dict], fn, default: float = 0.0) -> float:
    if not rows:
        return default
    return float(mean(fn(row) for row in rows))


def _mean(values: list[float], default: float = 0.0) -> float:
    values = [float(v) for v in values if math.isfinite(float(v))]
    return float(mean(values)) if values else default


@dataclass
class ParamProbeResult:
    selected_family: str
    scores: dict[str, float]
    reason: str
    features: dict[str, float]

    def scores_json(self) -> str:
        return json.dumps(self.scores, sort_keys=True)

    def features_json(self) -> str:
        return json.dumps(self.features, sort_keys=True)


def evaluate_param_family(
    rows: list[dict],
    *,
    start_step: int,
    window_steps: int,
    families: Iterable[str] = FAMILIES,
) -> ParamProbeResult:
    """Classify the likely dynamic-parameter family from observable history."""
    families = tuple(f for f in families if f in FAMILIES)
    if not families:
        families = FAMILIES

    pre = _window(rows, 0, max(start_step, 1))
    post = _window(rows, start_step, start_step + max(window_steps, 1))
    if len(post) < max(5, min(window_steps, 10)):
        return ParamProbeResult(
            selected_family="ambiguous",
            scores={family: 0.0 for family in families},
            reason="insufficient_window",
            features={"post_rows": float(len(post))},
        )

    pre_ref = pre or rows[: max(1, min(len(rows), 20))]
    depth_pre = _avg(pre_ref, lambda r: _abs_error(r, "depth"))
    depth_post = _avg(post, lambda r: _abs_error(r, "depth"))
    roll_pre = _avg(pre_ref, lambda r: _abs_error(r, "roll"))
    roll_post = _avg(post, lambda r: _abs_error(r, "roll"))
    pitch_pre = _avg(pre_ref, lambda r: _abs_error(r, "pitch"))
    pitch_post = _avg(post, lambda r: _abs_error(r, "pitch"))
    yaw_pre = _avg(pre_ref, lambda r: _abs_error(r, "yaw"))
    yaw_post = _avg(post, lambda r: _abs_error(r, "yaw"))
    effort_pre = _avg(pre_ref, lambda r: abs(_f(r, "control_effort")))
    effort_post = _avg(post, lambda r: abs(_f(r, "control_effort")))
    compound_vals = [_f(r, "compound_error") for r in post]
    compound_std = float(pstdev(compound_vals)) if len(compound_vals) > 1 else 0.0
    compound_mean = float(mean(compound_vals)) if compound_vals else 0.0
    diffs = [abs(compound_vals[i] - compound_vals[i - 1]) for i in range(1, len(compound_vals))]
    diff_med = float(median(diffs)) if diffs else 0.0
    diff_max = max(diffs) if diffs else 0.0
    impulse_ratio = diff_max / max(diff_med, eps := 1.0e-6)
    impulse_threshold = max(5.0 * diff_med, 0.05 * max(abs(compound_mean), eps), eps)
    impulse_density = float(sum(1 for value in diffs if value > impulse_threshold)) / max(len(diffs), 1)
    thirds = max(len(compound_vals) // 3, 1)
    early_mean = _mean(compound_vals[:thirds])
    late_mean = _mean(compound_vals[-thirds:])
    trend_delta = (late_mean - early_mean) / max(abs(early_mean), eps)
    action_asym = _avg(
        post,
        lambda r: abs(sum(_actions(r)[4:6]) - sum(_actions(r)[6:8])) if len(_actions(r)) >= 8 else 0.0,
    )
    excitation_active_frac = _avg(post, lambda r: 1.0 if _f(r, "param_probe_excitation_active") > 0.0 else 0.0)
    excitation_energy = _avg(post, lambda r: abs(_f(r, "param_probe_excitation_value")))
    active_compound = [_f(r, "compound_error") for r in post if _f(r, "param_probe_excitation_active") > 0.0]
    inactive_compound = [_f(r, "compound_error") for r in post if _f(r, "param_probe_excitation_active") <= 0.0]
    excitation_response_delta = (
        (_mean(active_compound) - _mean(inactive_compound)) / max(abs(_mean(inactive_compound)), eps)
        if active_compound and inactive_compound
        else 0.0
    )

    depth_delta = (depth_post - depth_pre) / max(depth_pre, eps)
    attitude_pre = roll_pre + pitch_pre + yaw_pre
    attitude_post = roll_post + pitch_post + yaw_post
    attitude_delta = (attitude_post - attitude_pre) / max(attitude_pre, eps)
    effort_delta = (effort_post - effort_pre) / max(effort_pre, eps)
    spike_ratio = compound_std / max(compound_mean, eps)
    persistent_shift = max(
        abs(depth_delta),
        abs(attitude_delta),
        abs(effort_delta),
        abs(trend_delta),
    )

    scores = {
        "density": 0.75 * depth_delta + 0.35 * effort_delta - 0.45 * max(attitude_delta, 0.0) - 0.10 * impulse_density,
        "thruster_efficiency": 0.75 * effort_delta + 0.35 * max(attitude_delta, 0.0) + 0.20 * action_asym - 0.20 * max(depth_delta, 0.0),
        "thruster_angle": 0.85 * max(attitude_delta, 0.0) + 0.30 * action_asym - 0.35 * max(effort_delta, 0.0),
    }
    scores = {k: float(scores[k]) for k in families}
    best_family, best_score = max(scores.items(), key=lambda kv: kv[1])
    ordered = sorted(scores.values(), reverse=True)
    margin = ordered[0] - ordered[1] if len(ordered) > 1 else ordered[0]

    if impulse_ratio > 10.0 and impulse_density < 0.12 and persistent_shift < 0.35:
        selected = "external_disturbance"
        reason = (
            "layer1_external_impulse "
            f"impulse_ratio={impulse_ratio:.3f} impulse_density={impulse_density:.3f} "
            f"persistent_shift={persistent_shift:.3f}"
        )
    elif persistent_shift < 0.12 and abs(excitation_response_delta) < 0.05:
        selected = "ambiguous"
        reason = (
            "layer2_weak_persistent_shift "
            f"persistent_shift={persistent_shift:.3f} excitation_response_delta={excitation_response_delta:.3f}"
        )
    elif best_score < 0.15 or margin < 0.10:
        selected = "ambiguous"
        reason = (
            "layer2_nonseparable_persistent_shift "
            f"best={best_family} score={best_score:.3f} margin={margin:.3f} "
            f"persistent_shift={persistent_shift:.3f}"
        )
    else:
        selected = best_family
        reason = (
            f"layer2_selected {best_family} score={best_score:.3f} margin={margin:.3f} "
            f"persistent_shift={persistent_shift:.3f}"
        )

    features = {
        "depth_delta": float(depth_delta),
        "attitude_delta": float(attitude_delta),
        "effort_delta": float(effort_delta),
        "action_asym": float(action_asym),
        "spike_ratio": float(spike_ratio),
        "impulse_ratio": float(impulse_ratio),
        "impulse_density": float(impulse_density),
        "trend_delta": float(trend_delta),
        "persistent_shift": float(persistent_shift),
        "excitation_active_frac": float(excitation_active_frac),
        "excitation_energy": float(excitation_energy),
        "excitation_response_delta": float(excitation_response_delta),
        "post_rows": float(len(post)),
    }
    return ParamProbeResult(selected, scores, reason, features)

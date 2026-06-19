"""STDW-style on-policy fine-tuning workflow for EasyUUV (v3 plan).

Replaces ``workflows/play_stdw_adapt.py`` (the legacy teacher-student
distillation pipeline) with a true STDW-style gradual domain adaptation loop:

- A ``EasyUUVStdwWrapper`` drives a linear ``com_to_cob_offset`` drift on
  selectable axes between ``drift_start_step`` and ``drift_end_step`` and
  performs a 5s rolling RMS low-pass on a compound tracking error signal.
- A ``StdwReplayBuffer`` collects (s, a, a_pseudo, r, s', error, mask, V, tag,
  step) tuples and exposes a ``sample_pair`` API for STDW source/target batch
  matching.
- The loaded ``actor_critic`` is fine-tuned in-place (no teacher/student
  duplication) with the STDW loss
  ``(1-rho) * L_src + rho * L_tgt + lambda_reg * ||theta - theta_pre||^2``.
- ``rho`` is synchronised with the drift fraction.
- A Lyapunov ``V_t = 0.5*e^T P e`` "physical sieve" mask filters the per-sample
  loss so that only energy-decreasing samples produce gradients.

The workflow remains compatible with ``custom_workflows/run_with_isaac_env.sh``
and ``experiment_runner.py``.
"""

from __future__ import annotations

import argparse
import copy
import importlib.util
import json
import shutil
import sys
from collections import deque
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import gymnasium as gym
import matplotlib

matplotlib.use("Agg")
import numpy as np
import torch


# ---------------------------------------------------------------------------
# Bootstrap path so the script can be launched standalone via run_with_isaac_env.
# ---------------------------------------------------------------------------

WORKFLOW_DIR = Path(__file__).resolve().parent
REPO_ROOT = WORKFLOW_DIR.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
CUSTOM_WORKFLOWS_DIR = REPO_ROOT / "custom_workflows"
if str(CUSTOM_WORKFLOWS_DIR) not in sys.path:
    sys.path.insert(0, str(CUSTOM_WORKFLOWS_DIR))


def _bootstrap_local_lab_tasks_package() -> None:
    """让 gym entry_point 能解析到本地的 ``easyuuv_stdw`` 包。

    实施细节非常脆弱：

    - gym 注册的 entry_point 是
      ``omni.isaac.lab_tasks.direct.easyuuv_stdw:EasyUUVEnv``，模块名
      含 ``-``，无法用 ``import`` 语法访问。脚本必须先以
      ``importlib.util.spec_from_file_location`` 的方式把本地 ``__init__.py``
      注入 ``sys.modules['omni.isaac.lab_tasks']``，并执行其中的
      ``gym.register(...)``。
    - 但本地 ``__init__.py`` 内部会 ``from . import agents``，agents 又 import
      ``omni.isaac.lab_tasks.utils.wrappers.rsl_rl``。一旦上面的注入把真正
      ``omni.isaac.lab_tasks`` 命名空间包覆盖掉，``utils.wrappers`` 就解析不到。
    - 因此必须先 ``import`` 真正的 ``omni.isaac.lab_tasks.utils.wrappers.rsl_rl``，
      让其与所有父链子模块进入 ``sys.modules`` 缓存；再做覆盖式注入；
      子模块 import 时会命中 sys.modules 缓存而不再触发父模块解析。
    """
    package_name = "omni.isaac.lab_tasks"
    if package_name in sys.modules and getattr(
        sys.modules[package_name], "__file__", None
    ) == str(REPO_ROOT / "__init__.py"):
        return

    # 1) 预先把真正命名空间包及其依赖子模块加载入 sys.modules。
    try:
        import omni.isaac.lab_tasks  # noqa: F401
        import omni.isaac.lab_tasks.utils  # noqa: F401
        import omni.isaac.lab_tasks.utils.wrappers  # noqa: F401
        import omni.isaac.lab_tasks.utils.wrappers.rsl_rl  # noqa: F401
    except Exception:
        # 即便预加载失败，也继续走原来的注入路径，让真正的错误在后面 import 时暴露。
        pass

    package_init = REPO_ROOT / "__init__.py"
    spec = importlib.util.spec_from_file_location(
        package_name,
        package_init,
        submodule_search_locations=[str(REPO_ROOT)],
    )
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to bootstrap {package_name} from {package_init}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[package_name] = module
    spec.loader.exec_module(module)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _bool_arg(value: str) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "on", "y", "t"}


def _parse_axes(spec: str) -> Tuple[int, ...]:
    spec = (spec or "").strip()
    if not spec:
        return (0,)
    return tuple(int(x.strip()) for x in spec.split(",") if x.strip())


def _parse_p_diag(spec: str) -> Tuple[float, float, float, float]:
    parts = [float(x.strip()) for x in (spec or "1,1,1,1").split(",") if x.strip()]
    if len(parts) != 4:
        raise argparse.ArgumentTypeError("--lyapunov_p_diag must have 4 comma-separated floats")
    return tuple(parts)  # type: ignore[return-value]


def _read_initial_cob_xy(raw_env) -> Tuple[float, float]:
    """Read the current base COM->COB xy offset for optional drift routing."""
    env = raw_env.unwrapped
    base = getattr(env, "_base_com_to_cob_offsets", None)
    if base is None:
        base = getattr(env, "com_to_cob_offsets", None)
    if base is None:
        return (0.0, 0.0)
    try:
        row = base[0].detach().cpu().tolist()
        return (float(row[0]), float(row[1]))
    except Exception:
        return (0.0, 0.0)


def _resolve_drift_router(raw_env, args: argparse.Namespace) -> Tuple[float, Tuple[int, ...], Tuple[float, float]]:
    """Resolve the effective drift target/axes without changing defaults.

    The current wrapper applies ``base_offset + frac * target_drift``.  For an
    already asymmetric body (e.g. x=y=0.05), adding another +0.05 on x is a
    perturbation rather than a correction.  This optional router maps such cases
    to a corrective drift while keeping the legacy path unchanged by default.
    """
    target = float(args.target_drift)
    axes = _parse_axes(args.drift_axes)
    xy = _read_initial_cob_xy(raw_env)
    if not bool(args.auto_drift_router) or str(args.drift_router_mode) == "off":
        return target, axes, xy

    if str(args.drift_router_mode) != "offset_correct":
        raise ValueError(f"Unsupported drift_router_mode: {args.drift_router_mode!r}")

    threshold = float(args.drift_router_xy_threshold)
    candidates = [(0, xy[0]), (1, xy[1])]
    selected = [(axis, value) for axis, value in candidates if abs(value) >= threshold]
    if not selected:
        print(
            f"[DRIFT-ROUTER] no xy offset above threshold={threshold:.4f}; "
            f"keeping target={target:.4f}, axes={axes}",
            flush=True,
        )
        return target, axes, xy

    signs = {1 if value > 0.0 else -1 for _, value in selected}
    if len(signs) > 1:
        # The wrapper currently supports one scalar target for all selected axes.
        # Use only the dominant axis to avoid correcting one axis while worsening another.
        axis, value = max(selected, key=lambda item: abs(item[1]))
        selected = [(axis, value)]

    axes = tuple(axis for axis, _ in selected)
    target = -sum(value for _, value in selected) / float(len(selected))
    print(
        f"[DRIFT-ROUTER] mode=offset_correct xy=({xy[0]:.4f},{xy[1]:.4f}) "
        f"threshold={threshold:.4f} -> target={target:.4f}, axes={axes}",
        flush=True,
    )
    return target, axes, xy


def _format_axes(axes: Tuple[int, ...]) -> str:
    if not axes:
        return "none"
    return ",".join(str(int(a)) for a in axes)


class MicroProbeController:
    """Observable-only drift router for scheme B.

    The controller never reads COM/COB offsets to make its decision.  It injects
    small reversible offsets for short windows, scores the observable tracking
    error, and then selects the candidate with the lowest mean error.
    """

    def __init__(
        self,
        *,
        enabled: bool,
        start_step: int,
        window_steps: int,
        settle_steps: int,
        axes: Tuple[int, ...],
        magnitude: float,
        metric: str,
        score_mode: str,
        min_improvement_abs: float,
        min_improvement_rel: float,
        consistency_margin_abs: float,
        consistency_margin_rel: float,
        baseline_each_candidate: bool,
        apply_result: bool,
    ) -> None:
        self.enabled = bool(enabled)
        self.start_step = int(start_step)
        self.window_steps = max(int(window_steps), 1)
        self.settle_steps = max(int(settle_steps), 0)
        self.axes = tuple(int(a) for a in axes)
        self.magnitude = float(magnitude)
        self.metric = str(metric)
        self.score_mode = str(score_mode)
        self.min_improvement_abs = float(min_improvement_abs)
        self.min_improvement_rel = float(min_improvement_rel)
        self.consistency_margin_abs = float(consistency_margin_abs)
        self.consistency_margin_rel = float(consistency_margin_rel)
        self.baseline_each_candidate = bool(baseline_each_candidate)
        self.apply_result = bool(apply_result)
        baseline = {"name": "baseline", "axes": tuple(), "target": 0.0}
        self.candidates: List[Dict[str, object]] = [baseline]
        for axis in self.axes:
            self.candidates.append({"name": f"axis{axis}_pos", "axes": (axis,), "target": abs(self.magnitude)})
            self.candidates.append({"name": f"axis{axis}_neg", "axes": (axis,), "target": -abs(self.magnitude)})
        if self.baseline_each_candidate:
            self.schedule: List[Dict[str, object]] = []
            for cand in self.candidates[1:]:
                self.schedule.append(baseline)
                self.schedule.append(cand)
            self.schedule.append(baseline)
        else:
            self.schedule = list(self.candidates)
        self.end_step = self.start_step + self.window_steps * len(self.schedule)
        self.scores: Dict[str, List[float]] = {str(c["name"]): [] for c in self.candidates}
        self.window_scores: List[List[float]] = [[] for _ in self.schedule]
        self.score_details: Dict[str, object] = {}
        self.selected: Optional[Dict[str, object]] = None
        self._selection_applied = False

    def _active_index(self, step: int) -> Optional[int]:
        if not self.enabled or step < self.start_step or step >= self.end_step:
            return None
        idx = (step - self.start_step) // self.window_steps
        if idx < 0 or idx >= len(self.schedule):
            return None
        return int(idx)

    def active_candidate(self, step: int) -> Optional[Dict[str, object]]:
        idx = self._active_index(step)
        if idx is None:
            return None
        return self.schedule[idx]

    def candidate_name(self, step: int) -> str:
        cand = self.active_candidate(step)
        return str(cand["name"]) if cand is not None else ""

    def should_score(self, step: int) -> bool:
        cand = self.active_candidate(step)
        if cand is None:
            return False
        local_step = (step - self.start_step) % self.window_steps
        return local_step >= self.settle_steps

    def record(self, step: int, value: float) -> None:
        if not self.should_score(step):
            return
        cand = self.active_candidate(step)
        if cand is None:
            return
        name = str(cand["name"])
        if np.isfinite(value):
            self.scores.setdefault(name, []).append(float(value))
            idx = self._active_index(step)
            if idx is not None:
                self.window_scores[idx].append(float(value))

    def _mean_for(self, name: str) -> Optional[float]:
        vals = self.scores.get(name, [])
        if not vals:
            return None
        return float(np.mean(vals))

    def _required_improvement(self, baseline_mean: float) -> float:
        return max(
            float(self.min_improvement_abs),
            float(self.min_improvement_rel) * max(abs(float(baseline_mean)), 1.0e-6),
        )

    def _required_consistency(self, baseline_mean: float) -> float:
        return max(
            float(self.consistency_margin_abs),
            float(self.consistency_margin_rel) * max(abs(float(baseline_mean)), 1.0e-6),
        )

    def _window_mean(self, idx: int) -> Optional[float]:
        if idx < 0 or idx >= len(self.window_scores):
            return None
        vals = self.window_scores[idx]
        if not vals:
            return None
        return float(np.mean(vals))

    def _local_candidate_stats(self, name: str) -> Optional[Tuple[float, float, float]]:
        """Return (candidate_mean, local_baseline_mean, local_improvement)."""
        if not self.baseline_each_candidate:
            baseline_mean = self._mean_for("baseline")
            candidate_mean = self._mean_for(name)
            if baseline_mean is None or candidate_mean is None:
                return None
            return candidate_mean, baseline_mean, float(baseline_mean - candidate_mean)
        for idx, cand in enumerate(self.schedule):
            if str(cand["name"]) != name:
                continue
            candidate_mean = self._window_mean(idx)
            baseline_means = [
                value
                for value in (self._window_mean(idx - 1), self._window_mean(idx + 1))
                if value is not None
            ]
            if candidate_mean is None or not baseline_means:
                return None
            baseline_mean = float(np.mean(baseline_means))
            return candidate_mean, baseline_mean, float(baseline_mean - candidate_mean)
        return None

    def _select_legacy_mean(self) -> Optional[Dict[str, object]]:
        best: Optional[Tuple[float, Dict[str, object]]] = None
        for cand in self.candidates:
            mean_val = self._mean_for(str(cand["name"]))
            if mean_val is None:
                continue
            if best is None or mean_val < best[0]:
                best = (mean_val, cand)
        if best is None:
            return None
        selected = dict(best[1])
        selected["score"] = best[0]
        selected["score_reason"] = "legacy_mean"
        return selected

    def _select_relative_improvement(self) -> Optional[Dict[str, object]]:
        best: Optional[Tuple[float, float, Dict[str, object]]] = None
        for cand in self.candidates[1:]:
            name = str(cand["name"])
            stats = self._local_candidate_stats(name)
            if stats is None:
                continue
            mean_val, baseline_mean, improvement = stats
            required = self._required_improvement(baseline_mean)
            if improvement < required:
                continue
            if best is None or improvement > best[0]:
                best = (improvement, mean_val, cand)
        if best is None:
            selected = dict(self.candidates[0])
            selected["score"] = self._mean_for("baseline")
            selected["score_reason"] = "baseline_preferred_no_required_improvement"
            return selected
        selected = dict(best[2])
        selected["score"] = best[1]
        selected["score_improvement"] = best[0]
        selected["score_reason"] = "relative_improvement"
        return selected

    def _select_paired_axis(self) -> Optional[Dict[str, object]]:
        baseline_mean = self._mean_for("baseline")
        if baseline_mean is None:
            return self._select_legacy_mean()
        best: Optional[Tuple[float, float, Dict[str, object], str]] = None
        pair_details: Dict[str, object] = {}
        for axis in self.axes:
            pos_name = f"axis{axis}_pos"
            neg_name = f"axis{axis}_neg"
            pos_stats = self._local_candidate_stats(pos_name)
            neg_stats = self._local_candidate_stats(neg_name)
            if pos_stats is None or neg_stats is None:
                continue
            pos_mean, pos_baseline, pos_improvement = pos_stats
            neg_mean, neg_baseline, neg_improvement = neg_stats
            pair_details[str(axis)] = {
                "pos_mean": pos_mean,
                "neg_mean": neg_mean,
                "pos_baseline_mean": pos_baseline,
                "neg_baseline_mean": neg_baseline,
                "pos_improvement": pos_improvement,
                "neg_improvement": neg_improvement,
            }
            candidates = [
                (pos_improvement, pos_mean, pos_baseline, pos_name, self.candidates[1 + 2 * self.axes.index(axis)]),
                (neg_improvement, neg_mean, neg_baseline, neg_name, self.candidates[2 + 2 * self.axes.index(axis)]),
            ]
            for improvement, mean_val, local_baseline, name, cand in candidates:
                other_improvement = neg_improvement if name.endswith("_pos") else pos_improvement
                separation = float(improvement - other_improvement)
                required_improvement = self._required_improvement(local_baseline)
                required_consistency = self._required_consistency(local_baseline)
                if improvement < required_improvement:
                    continue
                if separation < required_consistency:
                    continue
                if best is None or improvement > best[0]:
                    best = (improvement, mean_val, cand, name)
        self.score_details["paired_axis"] = pair_details
        if best is None:
            selected = dict(self.candidates[0])
            selected["score"] = baseline_mean
            selected["score_reason"] = "baseline_preferred_no_consistent_pair"
            return selected
        selected = dict(best[2])
        selected["score"] = best[1]
        selected["score_improvement"] = best[0]
        selected["score_reason"] = "paired_axis_consistent_improvement"
        return selected

    def maybe_select(self, step: int) -> Optional[Dict[str, object]]:
        if not self.enabled or self.selected is not None or step < self.end_step:
            return self.selected
        if self.score_mode == "legacy_mean":
            selected = self._select_legacy_mean()
        elif self.score_mode == "relative_improvement":
            selected = self._select_relative_improvement()
        elif self.score_mode == "paired_axis":
            selected = self._select_paired_axis()
        else:
            selected = self._select_legacy_mean()
        if selected is None:
            self.selected = dict(self.candidates[0])
            self.selected["score"] = None
        else:
            self.selected = selected
        return self.selected

    def mark_applied(self) -> None:
        self._selection_applied = True

    @property
    def selection_applied(self) -> bool:
        return self._selection_applied

    def summary(self) -> Dict[str, object]:
        score_means = {
            name: (float(np.mean(vals)) if vals else None)
            for name, vals in self.scores.items()
        }
        selected = self.selected or {}
        selected_axes = tuple(selected.get("axes", tuple())) if selected else tuple()
        return {
            "micro_probe_enabled": bool(self.enabled),
            "micro_probe_start_step": int(self.start_step),
            "micro_probe_end_step": int(self.end_step),
            "micro_probe_window_steps": int(self.window_steps),
            "micro_probe_settle_steps": int(self.settle_steps),
            "micro_probe_axes": list(self.axes),
            "micro_probe_magnitude": float(self.magnitude),
            "micro_probe_metric": str(self.metric),
            "micro_probe_score_mode": str(self.score_mode),
            "micro_probe_min_improvement_abs": float(self.min_improvement_abs),
            "micro_probe_min_improvement_rel": float(self.min_improvement_rel),
            "micro_probe_consistency_margin_abs": float(self.consistency_margin_abs),
            "micro_probe_consistency_margin_rel": float(self.consistency_margin_rel),
            "micro_probe_baseline_each_candidate": bool(self.baseline_each_candidate),
            "micro_probe_apply_result": bool(self.apply_result),
            "micro_probe_scores": score_means,
            "micro_probe_score_details": self.score_details,
            "micro_probe_selected_name": str(selected.get("name", "")) if selected else "",
            "micro_probe_selected_axes": list(selected_axes),
            "micro_probe_selected_target": float(selected.get("target", 0.0)) if selected else 0.0,
            "micro_probe_selected_score": selected.get("score") if selected else None,
            "micro_probe_selected_improvement": selected.get("score_improvement") if selected else None,
            "micro_probe_selected_reason": selected.get("score_reason", "") if selected else "",
            "micro_probe_selection_applied": bool(self.selection_applied),
        }


def _force_probe_offset(stdw_wrapper: "EasyUUVStdwWrapper", axes: Tuple[int, ...], target: float) -> None:
    """Temporarily set COM->COB offset for a reversible micro-probe step."""
    try:
        stdw_wrapper._capture_base_offset()
        base = getattr(stdw_wrapper, "_base_offset", None)
        env_offsets = getattr(stdw_wrapper.env.unwrapped, "com_to_cob_offsets", None)
        if base is None or env_offsets is None:
            return
        env_offsets[:] = base.to(env_offsets.device, env_offsets.dtype)
        for axis in axes:
            if 0 <= int(axis) < env_offsets.shape[-1]:
                env_offsets[:, int(axis)] = base[:, int(axis)].to(env_offsets.device, env_offsets.dtype) + float(target)
    except Exception as exc:
        print(f"[WARN] micro-probe offset injection failed: {exc}")


def _save_stdw_checkpoint(
    *,
    ckpt_dir: Path,
    step: int,
    policy,
    obs_normalizer,
    optimizer,
    metadata: Dict[str, object],
    export_deploy_jit: bool,
    dummy_obs_dim: int,
    device: torch.device | str,
) -> Dict[str, Optional[str]]:
    """Save an RSL-RL-compatible adapted checkpoint plus optional deploy JIT."""
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    stem = f"stdw_step_{int(step):06d}"
    ckpt_path = ckpt_dir / f"{stem}.pt"
    payload = {
        "model_state_dict": {k: v.detach().cpu() for k, v in policy.state_dict().items()},
        "optimizer_state_dict": optimizer.state_dict() if optimizer is not None else None,
        "iter": int(step),
        "infos": dict(metadata),
    }
    torch.save(payload, ckpt_path)
    meta_path = ckpt_dir / f"{stem}.json"

    deploy_path: Optional[Path] = None
    if export_deploy_jit:
        try:
            deploy_model = torch.nn.Sequential(copy.deepcopy(obs_normalizer).cpu(), copy.deepcopy(policy.actor).cpu())
            deploy_model.eval()
            dummy = torch.zeros(1, int(dummy_obs_dim), dtype=torch.float32)
            traced = torch.jit.trace(deploy_model, dummy, strict=False)
            deploy_path = ckpt_dir / f"{stem}_deploy.jit"
            traced.save(str(deploy_path))
        except Exception as exc:
            print(f"[WARN] deploy JIT export failed at step {step}: {exc}")
    # Restore caller device/training mode after deepcopy-only export.
    policy.to(device)
    if obs_normalizer is not None:
        obs_normalizer.to(device)
    meta_payload = {
        **metadata,
        "checkpoint_path": str(ckpt_path),
        "deploy_jit_path": str(deploy_path) if deploy_path is not None else None,
    }
    meta_path.write_text(json.dumps(meta_payload, indent=2), encoding="utf-8")
    return {
        "checkpoint_path": str(ckpt_path),
        "metadata_path": str(meta_path),
        "deploy_jit_path": str(deploy_path) if deploy_path is not None else None,
    }


parser = argparse.ArgumentParser(description="STDW online adaptation for EasyUUV (v3).")
parser.add_argument("--cpu", action="store_true", default=False)
parser.add_argument("--disable_fabric", action="store_true", default=False)
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument("--task", type=str, default="EasyUUV-Direct-v1")
parser.add_argument("--seed", type=int, default=None)
parser.add_argument("--total_steps", type=int, default=1400)
parser.add_argument(
    "--deterministic_reference",
    type=_bool_arg,
    default=False,
    help="Decouple the desired/reference trajectory from the global RNG so STDW "
    "on/off share an identical reference under the same seed (for fair overlay).",
)

# STDW gates
parser.add_argument("--use_stdw", type=_bool_arg, default=True)
parser.add_argument("--enable_filter", type=_bool_arg, default=True)
parser.add_argument("--use_quantile_filter", type=_bool_arg, default=True)
parser.add_argument("--discard_ratio", type=float, default=0.1)
parser.add_argument("--enable_pseudo_action", type=_bool_arg, default=True)
parser.add_argument("--pseudo_gain", type=float, default=1.0)
# 改良 2：伪标签饱和门控与自适应衰减
# - pseudo_gate_limit：clip(pseudo_gain * J^-1 Δu, -a_limit, +a_limit)，
#   防止 PID 积分饱和反哺出过激动作（默认 0.5，范围 [0,1]）。
# - pseudo_decay：pseudo_gain(ρ) = pseudo_gain_0 * (1 - pseudo_decay * ρ)，
#   随 drift_frac 增大让策略自己接管控制（建议 0.7）。
parser.add_argument("--pseudo_gate_limit", type=float, default=0.5)
parser.add_argument("--pseudo_decay", type=float, default=0.7)
# 控制策略：默认启用 A-S-Surface 让 easyuuv_env 写出真实的 PID_value_add 到
# _pid_value_add_buf。若仍 self_adapt=False，delta_u≡0，pseudo-action 链路无效。
parser.add_argument(
    "--control_profile",
    type=str,
    default="A-S-Surface",
    choices=["A-S-Surface", "S-Surface", "PID", "direct_pwm"],
)
parser.add_argument("--enable_lyapunov_mask", type=_bool_arg, default=True)
parser.add_argument("--lyapunov_eps", type=float, default=0.0)
parser.add_argument("--lyapunov_p_diag", type=str, default="1.0,1.0,1.0,1.0")
parser.add_argument("--g_C_lr", type=float, default=5e-5)
parser.add_argument("--lambda_reg", type=float, default=1e-3)
# Regularization mode: parameter-space L2 (legacy) vs behavior KL on source
# observations (preferred). behavior_kl 用 frozen ref policy 的输出做 anchor，
# 在输出空间约束策略漂移，比在 5K 个权重参数上加 L2 更鲁棒。
parser.add_argument(
    "--reg_mode", type=str, default="behavior_kl", choices=["l2", "behavior_kl"]
)
parser.add_argument("--target_drift", type=float, default=0.05)
parser.add_argument("--drift_start_step", type=int, default=200)
parser.add_argument("--drift_end_step", type=int, default=1200)
parser.add_argument("--drift_axes", type=str, default="0")
parser.add_argument(
    "--auto_drift_router",
    type=_bool_arg,
    default=False,
    help="Optional deployment probe: route COB drift based on initial com_to_cob xy. Default False keeps legacy behavior.",
)
parser.add_argument(
    "--drift_router_mode",
    type=str,
    default="off",
    choices=["off", "offset_correct"],
    help="off keeps --target_drift/--drift_axes; offset_correct applies a corrective drift for large xy offsets.",
)
parser.add_argument(
    "--drift_router_xy_threshold",
    type=float,
    default=0.04,
    help="Absolute xy offset threshold (m) used by --drift_router_mode offset_correct.",
)
parser.add_argument(
    "--enable_micro_probe",
    type=_bool_arg,
    default=False,
    help="Scheme B: observable-only online micro-probe for deployment drift routing. Default False.",
)
parser.add_argument(
    "--micro_probe_start_step",
    type=int,
    default=40,
    help="First step of the reversible micro-probe window. Keep before --drift_start_step for clean routing.",
)
parser.add_argument(
    "--micro_probe_window_steps",
    type=int,
    default=20,
    help="Number of steps per candidate probe offset.",
)
parser.add_argument(
    "--micro_probe_settle_steps",
    type=int,
    default=5,
    help="Initial steps ignored inside each candidate window.",
)
parser.add_argument(
    "--micro_probe_axes",
    type=str,
    default="0,1",
    help="Comma-separated COM->COB axes probed by scheme B. Uses observable response only.",
)
parser.add_argument(
    "--micro_probe_magnitude",
    type=float,
    default=0.02,
    help="Small reversible probe offset magnitude in meters.",
)
parser.add_argument(
    "--micro_probe_metric",
    type=str,
    default="filtered_error",
    choices=["filtered_error", "compound_error", "raw_error"],
    help="Observable metric used to score probe candidates.",
)
parser.add_argument(
    "--micro_probe_score_mode",
    type=str,
    default="paired_axis",
    choices=["legacy_mean", "relative_improvement", "paired_axis"],
    help=(
        "legacy_mean chooses the lowest raw mean; relative_improvement requires "
        "improvement over baseline; paired_axis also requires pos/neg consistency."
    ),
)
parser.add_argument(
    "--micro_probe_min_improvement_abs",
    type=float,
    default=0.01,
    help="Minimum absolute metric improvement over probe baseline before applying a drift.",
)
parser.add_argument(
    "--micro_probe_min_improvement_rel",
    type=float,
    default=0.03,
    help="Minimum relative improvement over probe baseline before applying a drift.",
)
parser.add_argument(
    "--micro_probe_consistency_margin_abs",
    type=float,
    default=0.005,
    help="Minimum absolute score separation between opposite directions on the same axis.",
)
parser.add_argument(
    "--micro_probe_consistency_margin_rel",
    type=float,
    default=0.03,
    help="Minimum relative score separation between opposite directions on the same axis.",
)
parser.add_argument(
    "--micro_probe_baseline_each_candidate",
    type=_bool_arg,
    default=True,
    help="Interleave a zero-drift baseline window before each candidate to reduce transient bias.",
)
parser.add_argument(
    "--micro_probe_apply_result",
    type=_bool_arg,
    default=True,
    help="If True, selected probe direction replaces target_drift/drift_axes after probing.",
)
parser.add_argument("--ramp_shape", type=str, default="linear", choices=["linear", "cosine"],
                    help="Shape of the drift / disturbance ramp between drift_start_step and "
                         "drift_end_step. 'cosine' produces a smoother S-curve with zero slope "
                         "at both endpoints, useful for sine/oscillatory scenarios that suffer "
                         "from large transient errors during a steep linear ramp.")
parser.add_argument("--filter_window_seconds", type=float, default=5.0)
parser.add_argument("--slow_loop_interval", type=int, default=60)
parser.add_argument("--batch_size", type=int, default=256)
parser.add_argument("--buffer_capacity", type=int, default=50000)
parser.add_argument("--resume_buffer", type=str, default=None)

# Triggered Adaptation Gating (TAG)
parser.add_argument("--enable_trigger_gate", type=_bool_arg, default=True,
                    help="TAG: 仅当低通滤波复合误差 >= trigger_threshold 时才激活慢环梯度更新")
parser.add_argument("--trigger_threshold", type=float, default=0.05,
                    help="TAG 阈值 (rad)；filt_err 低于此值时静默慢环自适应，防止稳态参数漂移")

# Disturbance / noise
parser.add_argument("--noise_std", type=float, default=0.02)
parser.add_argument("--noise_corr", type=float, default=0.8)
parser.add_argument("--wave_mode", type=str, default="sine", choices=["none", "constant", "sine", "jonswap"])
parser.add_argument("--wave_base_vel", nargs=3, type=float, default=[0.06, 0.0, 0.02])
parser.add_argument("--wave_amplitude", nargs=3, type=float, default=[0.08, 0.03, 0.02])
parser.add_argument("--wave_frequency", nargs=3, type=float, default=[0.16, 0.22, 0.3])

# Scenario / embodiment / fault (gradual injection presets)
parser.add_argument(
    "--scenario",
    type=str,
    default=None,
    help="Scenario preset name from scenarios.SCENARIO_PRESETS. "
         "When set, --wave_*/--noise_* are ignored and the schedule "
         "ramps disturbance from baseline to target between drift_start_step "
         "and drift_end_step.",
)
parser.add_argument(
    "--embodiment",
    type=str,
    default="base",
    choices=["base", "long_body", "heavy_moderate", "asymmetric"],
)
parser.add_argument(
    "--pid_multipliers",
    type=str,
    default=None,
    help="Optional JSON string applied via env.apply_pid_multipliers AFTER embodiment apply, "
         "BEFORE the first reset. Example: "
         "'{\"roll_zeta1\": 0.7, \"yaw_zeta3\": 1.2, \"depth_zeta1\": 0.8}'. "
         "Keys must match '<axis>_<param>' where axis ∈ {roll,pitch,yaw,depth} and "
         "param ∈ {zeta1,zeta2,zeta3}. Useful for re-tuning gains on heavy/asymmetric "
         "embodiments without re-training (option 6.2 in REPORT_scenarios_6k).",
)
parser.add_argument(
    "--fault_thrusters",
    type=str,
    default="4,5",
    help="Comma list of thruster indices to fault. Used only when scenario specifies fault_rate.",
)
parser.add_argument(
    "--fault_rate_per_second",
    type=float,
    default=None,
    help="Override scenario fault_rate. Falls back to scenario default if None.",
)
parser.add_argument(
    "--fault_start_offset_steps",
    type=int,
    default=0,
    help="Offset from drift_start_step where fault begins ramping.",
)

# Convergence
parser.add_argument("--stability_threshold", type=float, default=0.05,
                    help="Absolute compound-error threshold below which we count convergence streak. "
                         "Default 0.05 is tight; raise for high-disturbance scenarios.")
parser.add_argument("--stability_threshold_rel", type=float, default=0.0,
                    help="If > 0, also compute a relative threshold = rel * baseline_compound_error_mean "
                         "where baseline window is steps [0, drift_start_step). The effective threshold "
                         "used at runtime is max(stability_threshold, relative). Set to 0 to disable.")
parser.add_argument("--stability_window", type=int, default=10)
parser.add_argument("--final_mse_window", type=int, default=200)

# Paths / config
parser.add_argument("--workflow_config", type=str, default=None)
parser.add_argument("--logs_root", type=str, default=None)
parser.add_argument("--results_root", type=str, default=None)
parser.add_argument("--artifacts_root", type=str, default=None)
parser.add_argument("--experiment_name", type=str, default="easyuuv_direct")
parser.add_argument("--run_name", type=str, default=None)
parser.add_argument("--load_run", type=str, default=None)
parser.add_argument("--checkpoint", type=str, default="model_500.pt")
parser.add_argument(
    "--save_stdw_ckpt",
    type=_bool_arg,
    default=False,
    help="Save intermediate adapted STDW checkpoints. Default False keeps legacy behavior.",
)
parser.add_argument(
    "--stdw_ckpt_interval",
    type=int,
    default=300,
    help="Save adapted checkpoint every N steps when --save_stdw_ckpt is enabled.",
)
parser.add_argument(
    "--stdw_ckpt_keep_last",
    type=int,
    default=0,
    help="If >0, keep only the latest N intermediate checkpoint pairs.",
)
parser.add_argument(
    "--export_deploy_jit",
    type=_bool_arg,
    default=True,
    help="Also export obs_normalizer+actor TorchScript for Isaac-independent deployment.",
)
parser.add_argument("--logger", type=str, default=None, choices={"wandb", "tensorboard", "neptune"})
parser.add_argument("--log_project_name", type=str, default=None)
parser.add_argument("--resume", type=_bool_arg, default=None)


from omni.isaac.lab.app import AppLauncher  # type: ignore

AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

_bootstrap_local_lab_tasks_package()

# ---------------------------------------------------------------------------
# Lab-side imports (after AppLauncher).
# ---------------------------------------------------------------------------

from omni.isaac.lab.utils.math import euler_xyz_from_quat  # noqa: E402
from omni.isaac.lab_tasks.utils import get_checkpoint_path, parse_env_cfg  # noqa: E402
from omni.isaac.lab_tasks.utils.wrappers.rsl_rl import (  # noqa: E402
    RslRlOnPolicyRunnerCfg,
    RslRlVecEnvWrapper,
)
from rsl_rl.runners import OnPolicyRunner  # noqa: E402

import cli_args  # noqa: E402  isort: skip
from custom_workflows.workflow_config import apply_config_overrides, load_workflow_config  # noqa: E402
from custom_workflows.workflow_paths import (  # noqa: E402
    WorkflowPaths,
    ensure_directory,
    resolve_checkpoint_identifiers,
    resolve_checkpoint_path,
)
def _load_module_from_path(mod_name: str, file_path: Path):
    """按绝对路径加载模块，并以 ``mod_name`` 注册到 sys.modules。

    本仓库的本地 ``utils/`` 包名与 ``omni.isaac.lab_tasks.utils`` 同名，普通
    ``import utils.stdw_buffer`` 会被命名空间包截胡；这里走 importlib 显式
    加载，绕开 ``utils`` 名字冲突。``easyuuv_stdw_wrapper`` 在仓库根下并不
    属于任何包，但只要 ``REPO_ROOT`` 在 ``sys.path``，``import xxx`` 是 OK
    的——我们只对真正受影响的 ``utils.*`` 子模块用这种方式。
    """
    spec = importlib.util.spec_from_file_location(mod_name, file_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to load {mod_name} from {file_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[mod_name] = module
    spec.loader.exec_module(module)
    return module


_stdw_buffer_module = _load_module_from_path(
    "_local_stdw_buffer", REPO_ROOT / "utils" / "stdw_buffer.py"
)
StdwReplayBuffer = _stdw_buffer_module.StdwReplayBuffer  # type: ignore[attr-defined]
from easyuuv_stdw_wrapper import EasyUUVStdwWrapper  # noqa: E402
from stdw_integration import (  # noqa: E402
    STDWCSVLogger,
    angle_remap,
    calculate_compound_error,
    calculate_control_effort,
    calculate_domain_bias,
)
from stdw_integration.plots import _plot_stdw_diagnostics  # noqa: E402


# ---------------------------------------------------------------------------
# Helpers (forward paths, low-level reads, pose sniff)
# ---------------------------------------------------------------------------


def _policy_forward_eval(policy, normalized_obs):
    """Fast loop forward: detached, no grad."""
    with torch.no_grad():
        return policy.act_inference(normalized_obs)


def _policy_forward_train(policy, normalized_obs):
    """Slow loop forward: must keep computation graph alive for backward().

    rsl_rl 的 ``ActorCritic.actor`` 是普通 ``nn.Sequential``，最后一层是
    ``nn.Linear(hidden, num_actions)``，输出 shape 直接就是 (B, num_actions)，
    **不是** ``(mean, log_std)`` 的拼接（log_std 单独存在 ``policy.std``/
    ``policy.log_std`` 里）。之前按 2*action_dim 切片会把 (B,4) 误切成 (B,2)，
    导致下游 ``mse_tgt`` shape 不匹配，慢环 19 次全部 except 静默失败。
    """
    if hasattr(policy, "act_inference"):
        # act_inference 不包 no_grad，梯度可正常回传；与 policy.actor(obs) 等价。
        return policy.act_inference(normalized_obs)
    if hasattr(policy, "actor"):
        return policy.actor(normalized_obs)
    if hasattr(policy, "update_distribution"):
        policy.update_distribution(normalized_obs)
        return policy.distribution.loc
    raise RuntimeError("Cannot find a differentiable forward path on the policy.")


def _read_low_level_correction(stdw_wrapper: EasyUUVStdwWrapper, env, action_template: torch.Tensor) -> torch.Tensor:
    """Read PID_value_add from env.unwrapped (cached by the small easyuuv_env hook).

    Falls back to zeros when the field is missing (e.g. self_adapt=False).
    """
    delta_u = stdw_wrapper.last_low_level.get("delta_u") if stdw_wrapper.last_low_level else None
    if delta_u is None:
        delta_u = getattr(env.unwrapped, "_pid_value_add_buf", None)
    if delta_u is None or not isinstance(delta_u, torch.Tensor):
        return torch.zeros_like(action_template)
    delta_u = delta_u.to(device=action_template.device, dtype=action_template.dtype)
    if delta_u.shape != action_template.shape:
        try:
            delta_u = delta_u.reshape(action_template.shape)
        except Exception:
            return torch.zeros_like(action_template)
    return delta_u


def _read_jacobian_inv_diag(action_template: torch.Tensor) -> torch.Tensor:
    """4-channel allocation matrix is fixed; J_inv on this 4D space is identity diag."""
    return torch.ones_like(action_template)


def _get_true_pose(env):
    root_pos = env.unwrapped._robot.data.root_pos_w[0]
    root_quat = env.unwrapped._robot.data.root_quat_w[0]
    true_roll, true_pitch, true_yaw = euler_xyz_from_quat(root_quat.unsqueeze(0))
    return (
        float(root_pos[0].item()),
        float(root_pos[1].item()),
        float(root_pos[2].item()),
        float(angle_remap(true_roll)[0].item()),
        float(angle_remap(true_pitch)[0].item()),
        float(angle_remap(true_yaw)[0].item()),
    )


def _get_desired_pose(env):
    desired_quat = env.unwrapped._goal[0]
    des_roll, des_pitch, des_yaw = euler_xyz_from_quat(desired_quat.unsqueeze(0))
    des_depth = float(env.unwrapped.cfg.starting_depth)
    return (
        float(angle_remap(des_roll)[0].item()),
        float(angle_remap(des_pitch)[0].item()),
        float(angle_remap(des_yaw)[0].item()),
        des_depth,
    )


def _set_initial_disturbance(env, args, yaml_disturbance: dict | None = None) -> None:
    yaml_disturbance = yaml_disturbance or {}
    kwargs = dict(noise_std=args.noise_std, noise_corr=args.noise_corr)
    if "mode" not in yaml_disturbance:
        kwargs["mode"] = args.wave_mode
    if "base_vel" not in yaml_disturbance:
        kwargs["base_vel"] = list(args.wave_base_vel)
    if "amplitude" not in yaml_disturbance:
        kwargs["amplitude"] = list(args.wave_amplitude)
    if "frequency" not in yaml_disturbance:
        kwargs["frequency"] = list(args.wave_frequency)
    env.unwrapped.apply_runtime_domain_shift(**kwargs)


def _reset_wrapper_env(env):
    out = env.reset()
    return out[0] if isinstance(out, tuple) else out


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    workflow_cfg = load_workflow_config(args_cli.workflow_config)
    train_cfg = workflow_cfg.get("train", {}) or {}
    path_cfg = workflow_cfg.get("paths", {}) or {}

    if args_cli.task == parser.get_default("task") and "task" in train_cfg:
        args_cli.task = train_cfg["task"]
    if args_cli.num_envs == parser.get_default("num_envs") and "num_envs" in train_cfg:
        args_cli.num_envs = train_cfg["num_envs"]
    if args_cli.seed is None and "seed" in train_cfg:
        args_cli.seed = train_cfg["seed"]

    paths = WorkflowPaths.from_overrides(
        logs_root=args_cli.logs_root or path_cfg.get("logs_root"),
        results_root=args_cli.results_root or path_cfg.get("results_root"),
        artifacts_root=args_cli.artifacts_root or path_cfg.get("artifacts_root"),
    )

    env_cfg = parse_env_cfg(
        args_cli.task,
        use_gpu=not args_cli.cpu,
        num_envs=args_cli.num_envs,
        use_fabric=not args_cli.disable_fabric,
    )
    if workflow_cfg.get("env"):
        apply_config_overrides(env_cfg, workflow_cfg["env"])
    env_cfg.domain_randomization.use_custom_randomization = False
    env_cfg.noise_cfg.enable_noise = True
    env_cfg.noise_cfg.std_dev = args_cli.noise_std
    env_cfg.noise_cfg.correlation_coeff = args_cli.noise_corr
    # 把 seed 注入到环境与全局 RNG，使 (a) noise / (b) torch.randn / (c) np.random 都跨 seed 不同。
    if args_cli.seed is not None:
        try:
            env_cfg.seed = int(args_cli.seed)
        except Exception:
            pass
        import random as _py_random
        import numpy as _np
        import torch as _torch
        _py_random.seed(int(args_cli.seed))
        _np.random.seed(int(args_cli.seed))
        _torch.manual_seed(int(args_cli.seed))
        if _torch.cuda.is_available():
            _torch.cuda.manual_seed_all(int(args_cli.seed))
    # 公平 overlay 模式：把参考轨迹与全局 RNG 解耦（专用 generator 按 seed 复现），
    # 使 STDW on/off 在相同 seed 下得到逐步一致的 desired 轨迹。
    if args_cli.deterministic_reference:
        env_cfg.deterministic_reference = True
    # 仅当 yaml 未在 disturbance_cfg.mode 上明确写入时，才用 CLI 默认值覆盖；
    # 否则 yaml 的 jonswap_* / base_vel 等会被 CLI 默认值整段抹掉。
    yaml_disturbance = (workflow_cfg.get("env") or {}).get("disturbance_cfg") or {}
    if "mode" not in yaml_disturbance:
        env_cfg.disturbance_cfg.mode = args_cli.wave_mode
    if "base_vel" not in yaml_disturbance:
        env_cfg.disturbance_cfg.base_vel = list(args_cli.wave_base_vel)
    if "amplitude" not in yaml_disturbance:
        env_cfg.disturbance_cfg.amplitude = list(args_cli.wave_amplitude)
    if "frequency" not in yaml_disturbance:
        env_cfg.disturbance_cfg.frequency = list(args_cli.wave_frequency)
    print(f"[DISTURBANCE] mode={env_cfg.disturbance_cfg.mode} "
          f"base_vel={list(getattr(env_cfg.disturbance_cfg, 'base_vel', []))} "
          f"hs={getattr(env_cfg.disturbance_cfg, 'jonswap_hs', None)} "
          f"fp={getattr(env_cfg.disturbance_cfg, 'jonswap_fp', None)}")

    agent_cfg: RslRlOnPolicyRunnerCfg = cli_args.parse_rsl_rl_cfg(args_cli.task, args_cli, workflow_cfg.get("agent"))
    if args_cli.load_run:
        agent_cfg.load_run = args_cli.load_run
    if args_cli.checkpoint:
        agent_cfg.load_checkpoint = args_cli.checkpoint

    raw_env = gym.make(args_cli.task, cfg=env_cfg)

    # 启用控制策略：A-S-Surface 才会触发 self_adapt=True，让 easyuuv_env._pid_value_add_buf
    # 写入真实的低层修正量；否则 _read_low_level_correction 始终返回 0，pseudo-action 无效。
    try:
        raw_env.unwrapped.apply_control_profile(args_cli.control_profile)
    except Exception as exc:
        print(f"[WARN] apply_control_profile({args_cli.control_profile}) failed: {exc}")

    # Cross-embodiment: apply once after gym.make. Embodiment switching is not
    # gradual (PhysX mass / inertia rebuild). The disturbance schedule below
    # remains the only time-varying axis.
    if args_cli.embodiment != "base":
        try:
            raw_env.unwrapped.apply_embodiment_config(args_cli.embodiment)
        except Exception as exc:
            print(f"[WARN] apply_embodiment_config({args_cli.embodiment}) failed: {exc}")

    # 强制把 yaml 里的 disturbance 字段透到 env runtime cfg：parse_env_cfg / gym.make
    # 路径上 disturbance_cfg 这种 inner class 的注入并不可靠，且 apply_control_profile /
    # apply_embodiment_config 可能把 mode/base_vel 改回默认，因此放在它们之后做最终覆盖。
    if yaml_disturbance:
        try:
            shift_kwargs = {}
            for k in ("mode", "base_vel", "amplitude", "frequency",
                      "jonswap_hs", "jonswap_fp", "jonswap_gamma",
                      "jonswap_depth", "jonswap_direction", "jonswap_seed"):
                if k in yaml_disturbance:
                    shift_kwargs[k] = yaml_disturbance[k]
            if shift_kwargs:
                raw_env.unwrapped.apply_runtime_domain_shift(**shift_kwargs)
                print(f"[DISTURBANCE/runtime] applied: {shift_kwargs}")
        except Exception as exc:
            print(f"[WARN] apply_runtime_domain_shift(yaml) failed: {exc}")

    # Optional PID multipliers (option 6.2): apply after embodiment so the
    # rebuilt mass / inertia is the gain reference.
    if args_cli.pid_multipliers:
        try:
            zeta_updates = json.loads(args_cli.pid_multipliers)
            if not isinstance(zeta_updates, dict):
                raise ValueError("pid_multipliers must decode to a JSON object")
            raw_env.unwrapped.apply_pid_multipliers(zeta_updates)
            print(f"[INFO] applied pid_multipliers: {zeta_updates}")
        except Exception as exc:
            print(f"[WARN] apply_pid_multipliers({args_cli.pid_multipliers!r}) failed: {exc}")

    # Compose: EasyUUVStdwWrapper -> RslRlVecEnvWrapper.
    sim_dt = float(getattr(env_cfg.sim, "dt", 1.0 / 120.0))

    def _error_signal_callable(inner_env):
        try:
            des_quat = inner_env.unwrapped._goal[0]
            des_roll, des_pitch, des_yaw = euler_xyz_from_quat(des_quat.unsqueeze(0))
            root_quat = inner_env.unwrapped._robot.data.root_quat_w[0]
            true_roll, true_pitch, true_yaw = euler_xyz_from_quat(root_quat.unsqueeze(0))
            true_z = float(inner_env.unwrapped._robot.data.root_pos_w[0][2].item())
            des_depth = float(inner_env.unwrapped.cfg.starting_depth)
            return calculate_compound_error(
                float(angle_remap(des_roll)[0].item()),
                float(angle_remap(des_pitch)[0].item()),
                float(angle_remap(des_yaw)[0].item()),
                float(angle_remap(true_roll)[0].item()),
                float(angle_remap(true_pitch)[0].item()),
                float(angle_remap(true_yaw)[0].item()),
                des_depth,
                true_z,
            )
        except Exception:
            return 0.0

    effective_target_drift, effective_drift_axes, initial_cob_xy = _resolve_drift_router(raw_env, args_cli)

    stdw_wrapper = EasyUUVStdwWrapper(
        raw_env,
        drift_start_step=args_cli.drift_start_step,
        drift_end_step=args_cli.drift_end_step,
        target_drift=effective_target_drift,
        drift_axes=effective_drift_axes,
        enable_filter=args_cli.enable_filter,
        filter_window_seconds=args_cli.filter_window_seconds,
        sim_dt_seconds=sim_dt,
        ramp_shape=args_cli.ramp_shape,
        error_signal_callable=_error_signal_callable,
        enable_lyapunov_mask=args_cli.enable_lyapunov_mask,
        lyapunov_P_diag=_parse_p_diag(args_cli.lyapunov_p_diag),
        lyapunov_eps=args_cli.lyapunov_eps,
    )
    micro_probe = MicroProbeController(
        enabled=bool(args_cli.enable_micro_probe),
        start_step=int(args_cli.micro_probe_start_step),
        window_steps=int(args_cli.micro_probe_window_steps),
        settle_steps=int(args_cli.micro_probe_settle_steps),
        axes=_parse_axes(args_cli.micro_probe_axes),
        magnitude=float(args_cli.micro_probe_magnitude),
        metric=str(args_cli.micro_probe_metric),
        score_mode=str(args_cli.micro_probe_score_mode),
        min_improvement_abs=float(args_cli.micro_probe_min_improvement_abs),
        min_improvement_rel=float(args_cli.micro_probe_min_improvement_rel),
        consistency_margin_abs=float(args_cli.micro_probe_consistency_margin_abs),
        consistency_margin_rel=float(args_cli.micro_probe_consistency_margin_rel),
        baseline_each_candidate=bool(args_cli.micro_probe_baseline_each_candidate),
        apply_result=bool(args_cli.micro_probe_apply_result),
    )
    if micro_probe.enabled:
        print(
            f"[MICRO-PROBE] enabled start={micro_probe.start_step} end={micro_probe.end_step} "
            f"window={micro_probe.window_steps} axes={micro_probe.axes} "
            f"magnitude={micro_probe.magnitude:.4f} metric={micro_probe.metric} "
            f"score_mode={micro_probe.score_mode}",
            flush=True,
        )
    env = RslRlVecEnvWrapper(stdw_wrapper)

    log_root_path = paths.training_log_root(args_cli.experiment_name)
    print(f"[INFO] Loading checkpoint from directory: {log_root_path}", flush=True)
    resume_path = resolve_checkpoint_path(
        log_root_path,
        agent_cfg.load_run,
        agent_cfg.load_checkpoint,
        get_checkpoint_path,
    )
    print(f"[INFO] Loading model checkpoint from: {resume_path}", flush=True)
    resolved_load_run, resolved_checkpoint = resolve_checkpoint_identifiers(resume_path, agent_cfg.load_run)

    ppo_runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    # 仅加载策略权重；play 下游自建 optimizer（见下方 Adam），且分阶段训练的
    # checkpoint optimizer 仅含可训练子集，会与全参数 optimizer 的 param group 不匹配。
    ppo_runner.load(resume_path, load_optimizer=False)
    runtime_device = env.unwrapped.device

    # Direct fine-tune (no teacher/student copy).
    policy = ppo_runner.alg.actor_critic.to(runtime_device)
    policy.train()
    for parameter in policy.parameters():
        parameter.requires_grad_(True)
    optimizer = torch.optim.Adam(policy.parameters(), lr=args_cli.g_C_lr)

    theta_pre = {n: p.detach().clone() for n, p in policy.named_parameters()}

    # Frozen reference copy of the loaded checkpoint, used as the "source" anchor
    # for the slow loop. Without this, B_src["actions"] (recorded under the
    # fine-tuned policy) would coincide with policy(obs_src) and L_src ≡ 0.
    policy_ref = copy.deepcopy(policy).to(runtime_device)
    policy_ref.eval()
    for parameter in policy_ref.parameters():
        parameter.requires_grad_(False)

    obs_normalizer = ppo_runner.obs_normalizer.to(runtime_device)
    obs_normalizer.eval()
    for parameter in obs_normalizer.parameters():
        parameter.requires_grad_(False)

    # Sanity check: the training forward path must keep grad alive.
    try:
        dummy_obs = torch.zeros(1, env_cfg.num_observations, device=runtime_device)
        sample_out = _policy_forward_train(policy, dummy_obs)
        print(f"[INFO] _policy_forward_train requires_grad={sample_out.requires_grad}")
        assert sample_out.requires_grad, "fatal: policy training forward path is detached"
    except AssertionError:
        raise
    except Exception as exc:
        print(f"[WARN] forward-path sanity check skipped: {exc}")

    buffer = StdwReplayBuffer(capacity=args_cli.buffer_capacity, device=runtime_device)
    if args_cli.resume_buffer:
        try:
            buffer.load(Path(args_cli.resume_buffer))
            print(f"[INFO] Resumed buffer from {args_cli.resume_buffer}; size={len(buffer)}")
        except Exception as exc:
            print(f"[WARN] Failed to load resume buffer: {exc}")

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    result_run_dir = paths.result_run_dir(args_cli.experiment_name, resolved_load_run, resolved_checkpoint)
    stdw_run_dir = ensure_directory(result_run_dir / f"stdw_new_{timestamp}")
    csv_path = stdw_run_dir / "stdw_output.csv"
    artifact_dir = ensure_directory(
        paths.artifacts_root / args_cli.experiment_name / resolved_load_run / Path(resolved_checkpoint).stem / "stdw_new"
    )
    ckpt_dir = ensure_directory(stdw_run_dir / "checkpoints")
    saved_ckpts: List[Dict[str, Optional[str]]] = []

    _set_initial_disturbance(env, args_cli, yaml_disturbance=yaml_disturbance)

    # Optional: scenario-driven gradual injection schedule. When --scenario is
    # passed, the disturbance ramps from baseline to target between
    # drift_start_step and drift_end_step (in lockstep with the wrapper's COB
    # drift). When --scenario is None, the legacy step-injection above stays in
    # effect.
    schedule = None
    if args_cli.scenario is not None:
        try:
            from scenarios import resolve_scenario
            from disturbance_schedule import DisturbanceSchedule
        except ImportError:
            sys.path.insert(0, str(WORKFLOW_DIR))
            from scenarios import resolve_scenario  # type: ignore
            from disturbance_schedule import DisturbanceSchedule  # type: ignore

        fault_thrusters_cli = [int(x.strip()) for x in args_cli.fault_thrusters.split(",") if x.strip()]
        spec = resolve_scenario(
            args_cli.scenario,
            fault_thrusters=fault_thrusters_cli,
            fault_rate_per_second=args_cli.fault_rate_per_second,
            fault_start_offset_steps=args_cli.fault_start_offset_steps,
        )
        schedule = DisturbanceSchedule(
            env.unwrapped,
            spec,
            drift_start_step=args_cli.drift_start_step,
            drift_end_step=args_cli.drift_end_step,
            sim_dt_seconds=sim_dt,
            ramp_shape=args_cli.ramp_shape,
        )
        schedule.reset()
        print(
            f"[INFO] Scenario={spec.name} mode={spec.mode} "
            f"target_amp={spec.target.get('amplitude')} "
            f"target_base_vel={spec.target.get('base_vel')} "
            f"fault_rate={spec.fault_rate_per_second} "
            f"embodiment={args_cli.embodiment} "
            f"drift=[{args_cli.drift_start_step}, {args_cli.drift_end_step}]"
        )

    fieldnames = [
        "step",
        "time_s",
        "rho",
        "drift_fraction",
        "use_stdw",
        "use_filter",
        "use_quantile_filter",
        "raw_error",
        "filtered_error",
        "compound_error",
        "lyapunov_V",
        "lyapunov_dV",
        "stdw_mask",
        "effective_batch_frac",
        "loss",
        "loss_source",
        "loss_target",
        "loss_reg",
        "control_effort",
        "trigger_gate_silenced",
        "episode_reset",
        "domain_bias",
        "fluid_vx",
        "fluid_vy",
        "fluid_vz",
        "volume_mean",
        "des_roll",
        "des_pitch",
        "des_yaw",
        "des_depth",
        "true_x",
        "true_y",
        "true_z",
        "true_roll",
        "true_pitch",
        "true_yaw",
        "true_pose",
        "executed_action",
        "com_to_cob_offset_x",
        "com_to_cob_offset_y",
        "com_to_cob_offset_z",
        "micro_probe_candidate",
        "micro_probe_selected",
        "micro_probe_target",
        "micro_probe_axes",
        "micro_probe_reason",
        "micro_probe_improvement",
        # Scenario / schedule snapshot fields (filled when --scenario is set).
        "scenario",
        "embodiment",
        "disturbance_mode",
        "amp_x",
        "amp_y",
        "amp_z",
        "noise_std_eff",
        "fault_active",
        "fault_efficiency_min",
    ]
    logger = STDWCSVLogger(csv_path, fieldnames)

    print("## EVALUATION LOG ## STDW (new) adaptation started")
    obs, _ = env.get_observations()
    final_mse_window: deque = deque(maxlen=int(args_cli.final_mse_window))
    convergence_step: Optional[int] = None
    convergence_streak = 0
    nonfinite_guard_count = 0
    first_nonfinite_step: Optional[int] = None
    reset_count = 0
    slow_loop_triggers = 0
    gate_silenced_count = 0
    prev_V: Optional[float] = None
    # Baseline compound-error tracker for relative stability threshold (5.2 fix).
    # Collected over the pre-drift window steps in [0, drift_start_step). After the
    # window closes we freeze the mean and use max(abs_thr, rel * baseline) as the
    # effective threshold for convergence detection.
    baseline_err_sum: float = 0.0
    baseline_err_count: int = 0
    baseline_err_mean: Optional[float] = None
    effective_stability_threshold: float = float(args_cli.stability_threshold)

    for step in range(args_cli.total_steps):
        if not simulation_app.is_running():
            break

        # ---- fast loop ----
        with torch.no_grad():
            normalized = obs_normalizer(obs)
            action = _policy_forward_eval(policy, normalized).detach()
        if not torch.isfinite(action).all():
            nonfinite_guard_count += 1
            first_nonfinite_step = step if first_nonfinite_step is None else first_nonfinite_step
            reset_count += 1
            obs = _reset_wrapper_env(env)
            print(f"## EVALUATION LOG ## Step {step}: nonfinite_action_guard_reset")
            continue

        probe_candidate = micro_probe.active_candidate(step)
        if probe_candidate is not None:
            _force_probe_offset(
                stdw_wrapper,
                tuple(probe_candidate.get("axes", tuple())),  # type: ignore[arg-type]
                float(probe_candidate.get("target", 0.0)),
            )
        next_obs, reward, dones, extras = env.step(action)

        # Gradual disturbance injection (only when --scenario was set).
        if schedule is not None:
            try:
                schedule_snapshot = schedule.tick(step)
            except Exception as exc:
                print(f"[WARN] schedule.tick failed at step {step}: {exc}")
                schedule_snapshot = {}
        else:
            schedule_snapshot = {}
        injected = stdw_wrapper.last_extras or {}
        if isinstance(extras, dict):
            for key in ("stdw_raw_error", "stdw_filt_error", "stdw_drift_fraction", "stdw_step", "stdw_V", "stdw_mask"):
                if key in extras:
                    injected[key] = extras[key]
        raw_err = float(injected.get("stdw_raw_error", 0.0))
        filt_err = float(injected.get("stdw_filt_error", raw_err))
        drift_frac = float(injected.get("stdw_drift_fraction", stdw_wrapper.current_drift()))
        V_t = float(injected.get("stdw_V", 0.0))
        stdw_mask_val = float(injected.get("stdw_mask", 1.0))

        # ---- pseudo-action ----
        # 改良 2：在 pseudo_gain 上叠加 drift_frac 自适应衰减 + 修正量 clip 门控。
        #   pseudo_gain(ρ) = pseudo_gain_0 * (1 - pseudo_decay * ρ)
        #   correction      = clip(pseudo_gain(ρ) * J^-1 * Δu, -gate, +gate)
        if args_cli.enable_pseudo_action:
            delta_u = _read_low_level_correction(stdw_wrapper, env, action)
            j_inv_diag = _read_jacobian_inv_diag(action)
            decay = max(0.0, 1.0 - args_cli.pseudo_decay * float(drift_frac))
            scaled_gain = args_cli.pseudo_gain * decay
            raw_correction = scaled_gain * j_inv_diag * delta_u
            gate = float(args_cli.pseudo_gate_limit)
            if gate > 0.0:
                correction = torch.clamp(raw_correction, -gate, gate)
            else:
                correction = raw_correction
            if not torch.isfinite(correction).all():
                a_pseudo = action.clone()
            else:
                a_pseudo = torch.clamp(action + correction, -1.0, 1.0)
        else:
            a_pseudo = action.clone()

        # ---- buffer add (env_id=0) ----
        try:
            obs_row = obs[0] if obs.ndim > 1 else obs
            next_row = next_obs[0] if next_obs.ndim > 1 else next_obs
            action_row = action[0] if action.ndim > 1 else action
            pseudo_row = a_pseudo[0] if a_pseudo.ndim > 1 else a_pseudo
            reward_scalar = reward[0] if isinstance(reward, torch.Tensor) and reward.ndim > 0 else reward
            buffer.add(
                state=obs_row,
                action=action_row,
                pseudo_action=pseudo_row,
                reward=reward_scalar,
                next_state=next_row,
                error=filt_err if args_cli.enable_filter else raw_err,
                stdw_mask=stdw_mask_val,
                lyapunov_V=V_t,
                drift_frac=drift_frac,
                step=step,
            )
        except Exception as exc:
            print(f"[WARN] buffer.add failed at step {step}: {exc}")

        # ---- slow loop ----
        loss_total = 0.0
        loss_src_val = 0.0
        loss_tgt_val = 0.0
        loss_reg_val = 0.0
        effective_batch_frac = 0.0
        triggered_slow = False
        gate_silenced = False
        slow_due = (
            args_cli.use_stdw
            and step % args_cli.slow_loop_interval == 0
            and len(buffer) >= args_cli.batch_size
        )
        # TAG: 仅当滤波复合误差达到阈值（或门限关闭）才允许激活慢环更新。
        is_triggered = True
        if args_cli.enable_trigger_gate:
            is_triggered = (filt_err >= args_cli.trigger_threshold)
        if slow_due and not is_triggered:
            gate_silenced = True
            gate_silenced_count += 1
        if slow_due and is_triggered:
            triggered_slow = True
            slow_loop_triggers += 1
            rho = float(drift_frac)
            try:
                B_src, B_tgt, _ = buffer.sample_pair(
                    args_cli.batch_size,
                    rho,
                    use_quantile_filter=args_cli.use_quantile_filter,
                    discard_ratio=args_cli.discard_ratio,
                )
                a_src_pred = _policy_forward_train(policy, obs_normalizer(B_src["states"]))
                a_tgt_pred = _policy_forward_train(policy, obs_normalizer(B_tgt["states"]))
                if not a_tgt_pred.requires_grad:
                    raise RuntimeError("policy training forward path is detached")

                # Source / target anchors: frozen reference policy on the same observations.
                # Replaces B_src["actions"] which would be ≈ policy(obs_src) and
                # leak L_src ≡ 0 once the fine-tuned policy stays close to itself.
                # 改良 1：同时计算 target obs 上的 anchor，供双向 behavior_kl 使用。
                with torch.no_grad():
                    a_src_anchor = _policy_forward_train(
                        policy_ref, obs_normalizer(B_src["states"])
                    ).detach()
                    a_tgt_anchor = _policy_forward_train(
                        policy_ref, obs_normalizer(B_tgt["states"])
                    ).detach()

                mse_src = ((a_src_pred - a_src_anchor) ** 2).mean(dim=-1)
                mse_tgt = ((a_tgt_pred - B_tgt["pseudo_actions"]) ** 2).mean(dim=-1)

                if args_cli.enable_lyapunov_mask:
                    m_src = B_src["stdw_masks"]
                    m_tgt = B_tgt["stdw_masks"]
                    L_src = (m_src * mse_src).sum() / m_src.sum().clamp_min(1.0)
                    L_tgt = (m_tgt * mse_tgt).sum() / m_tgt.sum().clamp_min(1.0)
                    eff_num = float((m_src.sum() + m_tgt.sum()).item())
                    eff_den = float(len(m_src) + len(m_tgt))
                    effective_batch_frac = eff_num / max(eff_den, 1.0)
                else:
                    L_src = mse_src.mean()
                    L_tgt = mse_tgt.mean()
                    effective_batch_frac = 1.0

                # Regularization: parameter-space L2 vs behavior KL.
                # 改良 1：双向 behavioral anchoring（Bi-directional Behavioral KL）
                #   L_reg = (1-ρ) * MSE(π(s_src), π_ref(s_src))
                #         +     ρ * MSE(π(s_tgt), π_ref(s_tgt))
                # 物理意义：源域时锁住源域行为；目标域时锁住目标域不偏离 baseline。
                # 这给 OOD 状态下的网络退化加了一根"动态拉力弹簧"（解 §8.6 第 2 点）。
                # L2 模式保留作为 legacy 对照。
                if args_cli.reg_mode == "behavior_kl":
                    L_reg_src = ((a_src_pred - a_src_anchor) ** 2).mean()
                    L_reg_tgt = ((a_tgt_pred - a_tgt_anchor) ** 2).mean()
                    L_reg = (1.0 - rho) * L_reg_src + rho * L_reg_tgt
                else:
                    L_reg = sum(
                        ((p - theta_pre[n]) ** 2).sum()
                        for n, p in policy.named_parameters()
                    )
                loss = (1.0 - rho) * L_src + rho * L_tgt + args_cli.lambda_reg * L_reg

                loss_src_val = float(L_src.detach().item())
                loss_tgt_val = float(L_tgt.detach().item())
                loss_reg_val = float(L_reg.detach().item()) if isinstance(L_reg, torch.Tensor) else float(L_reg)
                loss_total = float(loss.detach().item())

                if torch.isfinite(loss) and effective_batch_frac > 0.0:
                    optimizer.zero_grad(set_to_none=True)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(policy.parameters(), max_norm=1.0)
                    optimizer.step()
            except Exception as exc:
                print(f"[WARN] slow-loop failure at step {step}: {exc}")

        # ---- metrics & domain readout ----
        try:
            fluid_velocity = env.unwrapped.get_current_fluid_velocity()[0].detach().cpu().tolist()
        except Exception:
            fluid_velocity = [0.0, 0.0, 0.0]
        try:
            volume_mean = float(env.unwrapped.volumes.mean().item())
            volume_scale = volume_mean / float(env.unwrapped.cfg.volume)
        except Exception:
            volume_mean = 0.0
            volume_scale = 1.0
        domain_bias = calculate_domain_bias(volume_scale, fluid_velocity)
        control_effort = calculate_control_effort(action)
        des_roll, des_pitch, des_yaw, des_depth = _get_desired_pose(env)
        true_x, true_y, true_z, true_roll, true_pitch, true_yaw = _get_true_pose(env)

        compound_error = calculate_compound_error(
            des_roll,
            des_pitch,
            des_yaw,
            true_roll,
            true_pitch,
            true_yaw,
            des_depth,
            true_z,
        )
        probe_metric_value = {
            "filtered_error": float(filt_err),
            "compound_error": float(compound_error),
            "raw_error": float(raw_err),
        }.get(str(args_cli.micro_probe_metric), float(filt_err))
        micro_probe.record(step, probe_metric_value)
        selected_probe = micro_probe.maybe_select(step)
        if (
            selected_probe is not None
            and micro_probe.apply_result
            and not micro_probe.selection_applied
        ):
            selected_axes = tuple(selected_probe.get("axes", tuple()))  # type: ignore[arg-type]
            selected_target = float(selected_probe.get("target", 0.0))
            stdw_wrapper.drift_axes = selected_axes
            stdw_wrapper.target_drift = selected_target
            effective_drift_axes = selected_axes
            effective_target_drift = selected_target
            micro_probe.mark_applied()
            print(
                f"[MICRO-PROBE] selected={selected_probe.get('name')} "
                f"score={selected_probe.get('score')} "
                f"target={selected_target:.4f} axes={selected_axes}",
                flush=True,
            )
        signal_for_eval = filt_err if args_cli.enable_filter else compound_error
        final_mse_window.append(float(signal_for_eval))
        # Accumulate baseline mean before drift starts (5.2: relative threshold).
        if step < int(args_cli.drift_start_step):
            baseline_err_sum += float(compound_error)
            baseline_err_count += 1
        elif baseline_err_mean is None and baseline_err_count > 0:
            baseline_err_mean = baseline_err_sum / float(baseline_err_count)
            if float(args_cli.stability_threshold_rel) > 0.0:
                relative = float(args_cli.stability_threshold_rel) * baseline_err_mean
                effective_stability_threshold = max(
                    float(args_cli.stability_threshold), relative
                )
                print(
                    f"[stability] baseline mean={baseline_err_mean:.4f}, "
                    f"abs_thr={args_cli.stability_threshold:.4f}, "
                    f"rel({args_cli.stability_threshold_rel}x)={relative:.4f}, "
                    f"effective={effective_stability_threshold:.4f}"
                )
        if signal_for_eval < effective_stability_threshold:
            convergence_streak += 1
            if convergence_streak >= args_cli.stability_window and convergence_step is None:
                convergence_step = step
        else:
            convergence_streak = 0

        # com_to_cob_offset readout (env_id=0).
        try:
            offset_row = env.unwrapped.com_to_cob_offsets[0].detach().cpu().tolist()
        except Exception:
            offset_row = [0.0, 0.0, 0.0]

        done_flag = bool(torch.any(dones > 0).item()) if isinstance(dones, torch.Tensor) else bool(dones)
        if done_flag:
            reset_count += 1

        executed_action_list = action.squeeze(0).detach().cpu().tolist() if action.ndim > 1 else action.detach().cpu().tolist()

        row = {
            "step": step,
            "time_s": float(step * sim_dt),
            "rho": float(drift_frac),
            "drift_fraction": float(drift_frac),
            "use_stdw": bool(args_cli.use_stdw),
            "use_filter": bool(args_cli.enable_filter),
            "use_quantile_filter": bool(args_cli.use_quantile_filter),
            "raw_error": float(raw_err),
            "filtered_error": float(filt_err),
            "compound_error": float(compound_error),
            "lyapunov_V": float(V_t),
            "lyapunov_dV": float(V_t - prev_V) if prev_V is not None else float("nan"),
            "stdw_mask": float(stdw_mask_val),
            "effective_batch_frac": float(effective_batch_frac) if triggered_slow else float("nan"),
            "loss": float(loss_total) if triggered_slow else float("nan"),
            "loss_source": float(loss_src_val) if triggered_slow else float("nan"),
            "loss_target": float(loss_tgt_val) if triggered_slow else float("nan"),
            "loss_reg": float(loss_reg_val) if triggered_slow else float("nan"),
            "control_effort": float(control_effort),
            "trigger_gate_silenced": int(gate_silenced),
            "episode_reset": int(done_flag),
            "domain_bias": float(domain_bias),
            "fluid_vx": fluid_velocity[0] if len(fluid_velocity) > 0 else 0.0,
            "fluid_vy": fluid_velocity[1] if len(fluid_velocity) > 1 else 0.0,
            "fluid_vz": fluid_velocity[2] if len(fluid_velocity) > 2 else 0.0,
            "volume_mean": float(volume_mean),
            "des_roll": des_roll,
            "des_pitch": des_pitch,
            "des_yaw": des_yaw,
            "des_depth": des_depth,
            "true_x": true_x,
            "true_y": true_y,
            "true_z": true_z,
            "true_roll": true_roll,
            "true_pitch": true_pitch,
            "true_yaw": true_yaw,
            "true_pose": [true_x, true_y, true_z, true_roll, true_pitch, true_yaw],
            "executed_action": executed_action_list,
            "com_to_cob_offset_x": offset_row[0] if len(offset_row) > 0 else 0.0,
            "com_to_cob_offset_y": offset_row[1] if len(offset_row) > 1 else 0.0,
            "com_to_cob_offset_z": offset_row[2] if len(offset_row) > 2 else 0.0,
            "micro_probe_candidate": micro_probe.candidate_name(step),
            "micro_probe_selected": (micro_probe.selected or {}).get("name", ""),
            "micro_probe_target": float((micro_probe.selected or {}).get("target", 0.0)),
            "micro_probe_axes": _format_axes(tuple((micro_probe.selected or {}).get("axes", tuple()))),
            "micro_probe_reason": (micro_probe.selected or {}).get("score_reason", ""),
            "micro_probe_improvement": (micro_probe.selected or {}).get("score_improvement", ""),
            # Scenario / disturbance-schedule snapshot (NaN/empty when --scenario is None).
            "scenario": schedule_snapshot.get("scenario", args_cli.scenario or "manual"),
            "embodiment": str(args_cli.embodiment),
            "disturbance_mode": schedule_snapshot.get("disturbance_mode", ""),
            "amp_x": schedule_snapshot.get("amp_x", float("nan")),
            "amp_y": schedule_snapshot.get("amp_y", float("nan")),
            "amp_z": schedule_snapshot.get("amp_z", float("nan")),
            "noise_std_eff": schedule_snapshot.get("noise_std_eff", ""),
            "fault_active": bool(schedule_snapshot.get("fault_active", False)),
            "fault_efficiency_min": schedule_snapshot.get("fault_efficiency_min", ""),
        }
        logger.append(row)

        if slow_due:
            print(
                f"## EVALUATION LOG ## [STDW-Slow] Triggered: {is_triggered} "
                f"(step={step} filt_err={filt_err:.6f} thr={args_cli.trigger_threshold:.6f} "
                f"gate={'on' if args_cli.enable_trigger_gate else 'off'})",
                flush=True,
            )
        if triggered_slow:
            print(
                "## EVALUATION LOG ## "
                f"[STDW-Slow] step={step} rho={drift_frac:.3f} loss={loss_total:.6e} "
                f"L_src={loss_src_val:.6e} L_tgt={loss_tgt_val:.6e} L_reg={loss_reg_val:.6e} "
                f"eff_frac={effective_batch_frac:.3f}",
                flush=True,
            )

        if (
            bool(args_cli.save_stdw_ckpt)
            and int(args_cli.stdw_ckpt_interval) > 0
            and (step > 0)
            and (
                step % int(args_cli.stdw_ckpt_interval) == 0
                or step == int(args_cli.total_steps) - 1
            )
        ):
            ckpt_info = _save_stdw_checkpoint(
                ckpt_dir=ckpt_dir,
                step=step,
                policy=policy,
                obs_normalizer=obs_normalizer,
                optimizer=optimizer,
                metadata={
                    "step": int(step),
                    "source_checkpoint": str(resume_path),
                    "task": str(args_cli.task),
                    "embodiment": str(args_cli.embodiment),
                    "target_drift": float(effective_target_drift),
                    "drift_axes": list(effective_drift_axes),
                    "slow_loop_triggers": int(slow_loop_triggers),
                    "final_mse_window_mean": float(np.mean(final_mse_window)) if len(final_mse_window) > 0 else None,
                    **micro_probe.summary(),
                },
                export_deploy_jit=bool(args_cli.export_deploy_jit),
                dummy_obs_dim=int(env_cfg.num_observations),
                device=runtime_device,
            )
            saved_ckpts.append(ckpt_info)
            keep_last = int(args_cli.stdw_ckpt_keep_last)
            if keep_last > 0 and len(saved_ckpts) > keep_last:
                stale = saved_ckpts[:-keep_last]
                saved_ckpts = saved_ckpts[-keep_last:]
                for item in stale:
                    for path_value in item.values():
                        if path_value:
                            try:
                                Path(path_value).unlink(missing_ok=True)
                            except Exception:
                                pass
            print(f"[STDW-CKPT] saved step={step} paths={ckpt_info}", flush=True)

        obs = next_obs
        prev_V = float(V_t)

    logger.close()

    # ---- collect final summary + plots ----
    csv_df = logger.to_frame()

    try:
        diagnostic_plot_paths, tracking_mse_summary = _plot_stdw_diagnostics(
            csv_df,
            stdw_run_dir,
            volume_jump_step=None,
            flow_jump_step=None,
            best_save_step=None,
        )
    except Exception as exc:
        import traceback as _tb
        print(f"[WARN] plotting failed: {exc}")
        _tb.print_exc()
        diagnostic_plot_paths = {}
        tracking_mse_summary = {}

    final_mse = float(np.mean(final_mse_window)) if len(final_mse_window) > 0 else None
    if bool(args_cli.save_stdw_ckpt) and not saved_ckpts:
        final_step = max(int(args_cli.total_steps) - 1, 0)
        ckpt_info = _save_stdw_checkpoint(
            ckpt_dir=ckpt_dir,
            step=final_step,
            policy=policy,
            obs_normalizer=obs_normalizer,
            optimizer=optimizer,
            metadata={
                "step": int(final_step),
                "source_checkpoint": str(resume_path),
                "task": str(args_cli.task),
                "embodiment": str(args_cli.embodiment),
                "target_drift": float(effective_target_drift),
                "drift_axes": list(effective_drift_axes),
                "slow_loop_triggers": int(slow_loop_triggers),
                "final_mse_window_mean": final_mse,
                **micro_probe.summary(),
            },
            export_deploy_jit=bool(args_cli.export_deploy_jit),
            dummy_obs_dim=int(env_cfg.num_observations),
            device=runtime_device,
        )
        saved_ckpts.append(ckpt_info)
        print(f"[STDW-CKPT] saved final paths={ckpt_info}", flush=True)

    # Stationary-window MSE: only steps strictly after drift_end_step (when scenario
    # has fully ramped). Falls back to None if no qualifying rows are present.
    final_mse_after_drift: Optional[float] = None
    try:
        if "compound_error" in csv_df.columns and "step" in csv_df.columns:
            tail = csv_df[csv_df["step"] > int(args_cli.drift_end_step)]
            if len(tail) > 0:
                signal_col = "filtered_error" if (args_cli.enable_filter and "filtered_error" in tail.columns) else "compound_error"
                vals = tail[signal_col].astype(float).dropna()
                if len(vals) > 0:
                    final_mse_after_drift = float(vals.mean())
    except Exception as exc:  # pragma: no cover
        print(f"[WARN] final_mse_after_drift computation failed: {exc}")

    summary = {
        "csv_path": str(csv_path),
        "buffer_path": str(stdw_run_dir / "buffer.pt"),
        "use_stdw": bool(args_cli.use_stdw),
        "enable_filter": bool(args_cli.enable_filter),
        "use_quantile_filter": bool(args_cli.use_quantile_filter),
        "discard_ratio": float(args_cli.discard_ratio),
        "g_C_lr": float(args_cli.g_C_lr),
        "lambda_reg": float(args_cli.lambda_reg),
        "reg_mode": str(args_cli.reg_mode),
        "target_drift": float(effective_target_drift),
        "target_drift_requested": float(args_cli.target_drift),
        "drift_start_step": int(args_cli.drift_start_step),
        "drift_end_step": int(args_cli.drift_end_step),
        "drift_axes": list(effective_drift_axes),
        "drift_axes_requested": list(_parse_axes(args_cli.drift_axes)),
        "auto_drift_router": bool(args_cli.auto_drift_router),
        "drift_router_mode": str(args_cli.drift_router_mode),
        "drift_router_xy_threshold": float(args_cli.drift_router_xy_threshold),
        "initial_com_to_cob_x": float(initial_cob_xy[0]),
        "initial_com_to_cob_y": float(initial_cob_xy[1]),
        "ramp_shape": str(args_cli.ramp_shape),
        "pid_multipliers": args_cli.pid_multipliers,
        "filter_window_seconds": float(args_cli.filter_window_seconds),
        "slow_loop_interval": int(args_cli.slow_loop_interval),
        "batch_size": int(args_cli.batch_size),
        "buffer_capacity": int(args_cli.buffer_capacity),
        "enable_pseudo_action": bool(args_cli.enable_pseudo_action),
        "pseudo_gain": float(args_cli.pseudo_gain),
        "pseudo_gate_limit": float(args_cli.pseudo_gate_limit),
        "pseudo_decay": float(args_cli.pseudo_decay),
        "control_profile": str(args_cli.control_profile),
        "enable_lyapunov_mask": bool(args_cli.enable_lyapunov_mask),
        "lyapunov_eps": float(args_cli.lyapunov_eps),
        "lyapunov_p_diag": list(_parse_p_diag(args_cli.lyapunov_p_diag)),
        "total_steps": int(args_cli.total_steps),
        "final_mse": final_mse,
        "convergence_step": convergence_step,
        "stability_threshold_abs": float(args_cli.stability_threshold),
        "stability_threshold_rel": float(args_cli.stability_threshold_rel),
        "stability_threshold_effective": float(effective_stability_threshold),
        "baseline_compound_error_mean": baseline_err_mean,
        "slow_loop_triggers": int(slow_loop_triggers),
        "enable_trigger_gate": bool(args_cli.enable_trigger_gate),
        "trigger_threshold": float(args_cli.trigger_threshold),
        "gate_silenced_count": int(gate_silenced_count),
        "reset_count": int(reset_count),
        "nonfinite_guard_count": int(nonfinite_guard_count),
        "first_nonfinite_step": first_nonfinite_step,
        "plot_paths": {name: str(path) for name, path in diagnostic_plot_paths.items()},
        # Scenario / embodiment / fault metadata for sweep aggregation.
        "scenario": args_cli.scenario or "manual",
        "embodiment": args_cli.embodiment,
        "fault_rate_per_second": args_cli.fault_rate_per_second,
        "fault_thrusters": args_cli.fault_thrusters,
        "final_mse_after_drift": final_mse_after_drift,
        "save_stdw_ckpt": bool(args_cli.save_stdw_ckpt),
        "stdw_ckpt_interval": int(args_cli.stdw_ckpt_interval),
        "stdw_ckpt_keep_last": int(args_cli.stdw_ckpt_keep_last),
        "export_deploy_jit": bool(args_cli.export_deploy_jit),
        "stdw_ckpt_dir": str(ckpt_dir),
        "saved_stdw_ckpts": saved_ckpts,
        **micro_probe.summary(),
        **tracking_mse_summary,
    }
    summary_path = stdw_run_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    try:
        buffer.save(stdw_run_dir / "buffer.pt")
    except Exception as exc:
        print(f"[WARN] buffer.save failed: {exc}")

    # Mirror plots into artifact_dir for review.
    for name, plot_path in diagnostic_plot_paths.items():
        try:
            shutil.copy2(plot_path, artifact_dir / f"stdw_{name}_{timestamp}.png")
        except Exception:
            pass
    try:
        shutil.copy2(summary_path, artifact_dir / f"summary_{timestamp}.json")
    except Exception:
        pass
    # Mirror raw tracking CSV into artifact_dir so every sweep cell keeps the
    # original target/actual angles + compound error for later re-processing.
    try:
        shutil.copy2(csv_path, artifact_dir / f"stdw_output_{timestamp}.csv")
    except Exception:
        pass

    env.close()


if __name__ == "__main__":
    import traceback as _tb
    try:
        main()
    except SystemExit:
        raise
    except BaseException as exc:  # noqa: BLE001
        print(f"## EVALUATION LOG ## FATAL exception in main(): {type(exc).__name__}: {exc}")
        _tb.print_exc()
        raise
    finally:
        try:
            simulation_app.close()
        except Exception:
            pass

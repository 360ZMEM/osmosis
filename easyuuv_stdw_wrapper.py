"""STDW gym.Wrapper for the EasyUUV direct RL environment.

This wrapper layers two responsibilities on top of the underlying
``EasyUUVEnv`` (registered under ``EasyUUV-Direct-v1``) without touching
its core control logic:

1. Progressive (linear) drift of the ``com_to_cob_offsets`` tensor between a
   ``drift_start_step`` and ``drift_end_step``, on selectable axes (default X).
   This realises the STDW paper's "smooth intermediate domains" assumption.
2. Per-step low-pass (rolling RMS) filtering of an error signal coming from
   the env (defaults to ``calculate_compound_error``).

In addition (v3) the wrapper computes a Lyapunov-style energy
``V_t = 0.5 * e^T P e`` over the (roll, pitch, yaw, depth) channel error and
emits a binary ``stdw_mask = 1`` iff ``ΔV < lyapunov_eps`` (i.e. energy
decreases monotonically), used as a "physical sieve" by the slow-loop loss.

The wrapper exposes ``last_extras`` and ``last_low_level`` so the main
workflow can read STDW-related signals even when an outer
``RslRlVecEnvWrapper`` strips extras fields.
"""

from __future__ import annotations

import collections
import math
from typing import Callable, Dict, Iterable, Optional, Tuple

import gymnasium as gym
import torch


def _wrap_to_pi(x: float) -> float:
    return (x + math.pi) % (2.0 * math.pi) - math.pi


class EasyUUVStdwWrapper(gym.Wrapper):
    """gym.Wrapper that drives progressive domain shift + low-pass filter."""

    def __init__(
        self,
        env: gym.Env,
        *,
        drift_start_step: int = 0,
        drift_end_step: int = 1000,
        target_drift: float = 0.05,
        drift_axes: Iterable[int] = (0,),
        enable_filter: bool = True,
        filter_window_seconds: float = 5.0,
        sim_dt_seconds: float = 1.0 / 120.0,
        ramp_shape: str = "linear",
        error_signal_callable: Optional[Callable[[gym.Env], float]] = None,
        enable_lyapunov_mask: bool = True,
        lyapunov_P_diag: Tuple[float, float, float, float] = (1.0, 1.0, 1.0, 1.0),
        lyapunov_eps: float = 0.0,
        lyapunov_gate_mode: str = "sample_mask",
        lyapunov_abs_margin: float = 0.0,
        lyapunov_rel_margin: float = 0.0,
        lyapunov_window_steps: int = 60,
        lyapunov_min_pass_rate: float = 0.0,
        lyapunov_v_mode: str = "pose_quadratic",
        lyapunov_Q_diag: Tuple[float, float, float, float] = (0.0, 0.0, 0.0, 0.0),
        lyapunov_decay_alpha: float = 0.0,
    ) -> None:
        super().__init__(env)
        self.drift_start_step = int(drift_start_step)
        self.drift_end_step = int(drift_end_step)
        self.target_drift = float(target_drift)
        self.drift_axes = tuple(int(a) for a in drift_axes)
        self.enable_filter = bool(enable_filter)
        self.filter_window_seconds = float(filter_window_seconds)
        self.sim_dt_seconds = float(sim_dt_seconds)
        ramp_shape = str(ramp_shape).lower()
        if ramp_shape not in {"linear", "cosine", "step"}:
            raise ValueError(f"ramp_shape must be linear/cosine/step, got {ramp_shape!r}")
        self.ramp_shape = ramp_shape
        self.error_signal_callable = error_signal_callable
        self.enable_lyapunov_mask = bool(enable_lyapunov_mask)
        self.lyapunov_eps = float(lyapunov_eps)
        lyapunov_gate_mode = str(lyapunov_gate_mode).lower()
        if lyapunov_gate_mode not in {"sample_mask", "strict_sample_mask", "guarded_drift"}:
            raise ValueError(
                "lyapunov_gate_mode must be sample_mask/strict_sample_mask/guarded_drift, "
                f"got {lyapunov_gate_mode!r}"
            )
        self.lyapunov_gate_mode = lyapunov_gate_mode
        self.lyapunov_abs_margin = float(lyapunov_abs_margin)
        self.lyapunov_rel_margin = float(lyapunov_rel_margin)
        self.lyapunov_window_steps = max(int(lyapunov_window_steps), 1)
        self.lyapunov_min_pass_rate = float(lyapunov_min_pass_rate)

        # M1: Lyapunov V redefinition (plug-and-play; default pose_quadratic = legacy).
        lyapunov_v_mode = str(lyapunov_v_mode).lower()
        if lyapunov_v_mode not in {
            "pose_quadratic",
            "so3_consistent",
            "energy_with_rate",
            "control_lyapunov",
        }:
            raise ValueError(
                "lyapunov_v_mode must be pose_quadratic/so3_consistent/energy_with_rate/"
                f"control_lyapunov, got {lyapunov_v_mode!r}"
            )
        self.lyapunov_v_mode = lyapunov_v_mode
        q_tuple = tuple(float(v) for v in lyapunov_Q_diag)
        if len(q_tuple) != 4:
            raise ValueError("lyapunov_Q_diag must contain exactly 4 rate weights (roll, pitch, yaw, depth)")
        self._Q_diag = torch.tensor(q_tuple, dtype=torch.float32)
        self.lyapunov_decay_alpha = float(lyapunov_decay_alpha)
        # Rate state for energy_with_rate / control_lyapunov: previous error vector.
        self._e_prev: Optional[torch.Tensor] = None

        window_steps = max(int(round(self.filter_window_seconds / max(self.sim_dt_seconds, 1.0e-6))), 1)
        self._error_window: collections.deque = collections.deque(maxlen=window_steps)
        self._lyap_pass_window: collections.deque = collections.deque(maxlen=self.lyapunov_window_steps)
        self._baseline_V_window: collections.deque = collections.deque(maxlen=self.lyapunov_window_steps)
        self._baseline_error_window: collections.deque = collections.deque(maxlen=self.lyapunov_window_steps)
        self._step_count: int = 0
        self._base_offset: Optional[torch.Tensor] = None
        self._V_prev: Optional[float] = None
        self._frozen_drift_frac: Optional[float] = None

        # Device-aware P_diag tensor; lazily moved when we first see the env tensor.
        p_tuple = tuple(float(v) for v in lyapunov_P_diag)
        if len(p_tuple) != 4:
            raise ValueError("lyapunov_P_diag must contain exactly 4 channel weights (roll, pitch, yaw, depth)")
        self._P_diag = torch.tensor(p_tuple, dtype=torch.float32)

        # Convenience handles populated each step.
        self.last_extras: Dict[str, float | str] = {}
        self.last_low_level: Dict[str, torch.Tensor] = {}

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _capture_base_offset(self) -> None:
        if self._base_offset is not None:
            return
        base = getattr(self.env.unwrapped, "_base_com_to_cob_offsets", None)
        if base is None:
            base = getattr(self.env.unwrapped, "com_to_cob_offsets", None)
        if base is None:
            return
        self._base_offset = base.detach().clone()

    def _compute_drift_fraction(self, step: int) -> float:
        if step <= self.drift_start_step:
            frac = 0.0
        elif self.ramp_shape == "step" or step >= self.drift_end_step:
            frac = 1.0
        else:
            denom = max(self.drift_end_step - self.drift_start_step, 1)
            linear_frac = float(step - self.drift_start_step) / float(denom)
            if self.ramp_shape == "cosine":
                # Smooth s-curve: 0 -> 1 with zero derivative at both ends.
                # 0.5 * (1 - cos(pi * x)) maps [0,1] -> [0,1] smoothly.
                import math as _m
                frac = 0.5 * (1.0 - _m.cos(_m.pi * linear_frac))
            else:
                frac = linear_frac
        if self._frozen_drift_frac is not None:
            frac = min(frac, float(self._frozen_drift_frac))
        return frac

    def freeze_current_drift(self) -> None:
        self._frozen_drift_frac = self._compute_drift_fraction(self._step_count)

    def clear_drift_freeze(self) -> None:
        self._frozen_drift_frac = None

    def _apply_drift(self, step: int) -> float:
        self._capture_base_offset()
        frac = self._compute_drift_fraction(step)
        if self._base_offset is None:
            return frac
        env_offsets = getattr(self.env.unwrapped, "com_to_cob_offsets", None)
        if env_offsets is None:
            return frac
        env_offsets[:] = self._base_offset.to(env_offsets.device, env_offsets.dtype)
        for axis in self.drift_axes:
            if 0 <= axis < env_offsets.shape[-1]:
                env_offsets[:, axis] = self._base_offset[:, axis].to(env_offsets.device, env_offsets.dtype) + frac * self.target_drift
        return frac

    def _filter_error(self, raw_error: float) -> float:
        if not self.enable_filter:
            return float(raw_error)
        self._error_window.append(float(raw_error))
        if not self._error_window:
            return float(raw_error)
        sq = sum(e * e for e in self._error_window) / float(len(self._error_window))
        return float(math.sqrt(sq))

    def _read_channel_error_vec(self) -> torch.Tensor:
        env = self.env.unwrapped
        cached = getattr(env, "_tracking_error_rpy_depth", None)
        if cached is not None:
            e = cached
            if e.ndim > 1:
                e = e[0]
            return e.detach().to(dtype=torch.float32)
        # Fallback: recompute from desired vs true pose, env_id=0.
        try:
            from omni.isaac.lab.utils.math import euler_xyz_from_quat  # type: ignore
            root_quat = env._robot.data.root_quat_w[0]
            true_roll, true_pitch, true_yaw = euler_xyz_from_quat(root_quat.unsqueeze(0))
            true_z = float(env._robot.data.root_pos_w[0][2].item())
            desired_quat = env._goal[0]
            des_roll, des_pitch, des_yaw = euler_xyz_from_quat(desired_quat.unsqueeze(0))
            des_depth = float(getattr(env.cfg, "starting_depth", 1.5))
            roll_err = _wrap_to_pi(float(des_roll[0].item()) - float(true_roll[0].item()))
            pitch_err = _wrap_to_pi(float(des_pitch[0].item()) - float(true_pitch[0].item()))
            yaw_err = _wrap_to_pi(float(des_yaw[0].item()) - float(true_yaw[0].item()))
            depth_err = des_depth - true_z
            return torch.tensor([roll_err, pitch_err, yaw_err, depth_err], dtype=torch.float32)
        except Exception:
            return torch.zeros(4, dtype=torch.float32)

    def _read_so3_error_vec(self) -> torch.Tensor:
        """M1 V-B: SO(3)-consistent error, aligned with reward/metrics quat_error_magnitude.

        Returns a 4-vector ``[e_att, 0, 0, depth_err]`` so the same diagonal P
        weighting applies: ``P[0]`` weights the SO(3) attitude geodesic error and
        ``P[3]`` weights depth.  Channels 1/2 are zeroed (attitude collapses to a
        single geodesic magnitude), avoiding the Euler wrap discontinuities that
        corrupt V under flip360 large-angle references.
        """
        env = self.env.unwrapped
        try:
            from omni.isaac.lab.utils.math import quat_error_magnitude  # type: ignore

            goal_quat = env._goal[0:1, 0:4]
            root_quat = env._robot.data.root_quat_w[0:1, 0:4]
            e_att = float(quat_error_magnitude(goal_quat, root_quat)[0].item())
            true_z = float(env._robot.data.root_pos_w[0][2].item())
            des_depth = float(getattr(env.cfg, "starting_depth", 1.5))
            depth_err = des_depth - true_z
            return torch.tensor([e_att, 0.0, 0.0, depth_err], dtype=torch.float32)
        except Exception:
            return self._read_channel_error_vec()

    def _error_vec_for_mode(self) -> torch.Tensor:
        if self.lyapunov_v_mode == "so3_consistent":
            e = self._read_so3_error_vec()
        else:
            e = self._read_channel_error_vec()
        if e.ndim > 1:
            e = e[0]
        if e.numel() < 4:
            pad = torch.zeros(4, dtype=torch.float32)
            pad[: e.numel()] = e.to(dtype=torch.float32)
            return pad
        return e[:4].to(dtype=torch.float32)

    def _compute_V(self, e: torch.Tensor) -> Tuple[float, torch.Tensor]:
        """Return (V_t, e_dot) for the active v_mode.

        ``e_dot`` (error rate, finite difference over one step) is only folded
        into V for ``energy_with_rate`` / ``control_lyapunov``; otherwise it is
        returned for bookkeeping but contributes zero energy.
        """
        p = self._P_diag.to(e.device)
        V_pose = 0.5 * float(torch.sum(p * (e ** 2)).item())
        if self._e_prev is not None and self._e_prev.shape == e.shape:
            dt = max(self.sim_dt_seconds, 1.0e-6)
            e_dot = (e - self._e_prev.to(e.device)) / dt
        else:
            e_dot = torch.zeros_like(e)
        if self.lyapunov_v_mode in {"energy_with_rate", "control_lyapunov"}:
            q = self._Q_diag.to(e.device)
            V_rate = 0.5 * float(torch.sum(q * (e_dot ** 2)).item())
            return V_pose + V_rate, e_dot
        return V_pose, e_dot

    def _compute_lyapunov_mask(self) -> Tuple[float, float, float, float]:
        e = self._error_vec_for_mode()
        V_t, e_dot = self._compute_V(e)
        self._e_prev = e.detach().clone()
        if self._V_prev is None:
            dV = float("nan")
        else:
            dV = V_t - self._V_prev
        if not self.enable_lyapunov_mask:
            return V_t, 1.0, dV, 1.0
        if self._V_prev is None:
            return V_t, 0.0, dV, 0.0

        if self.lyapunov_v_mode == "control_lyapunov":
            # CLF-style exponential decay: dV <= -alpha * V_prev (and still < 0).
            alpha = abs(self.lyapunov_decay_alpha)
            passed = dV < -alpha * max(abs(self._V_prev), 0.0)
        elif self.lyapunov_gate_mode in {"strict_sample_mask", "guarded_drift"}:
            abs_ok = dV < -abs(self.lyapunov_abs_margin)
            rel_ok = False
            if abs(self._V_prev) > 1.0e-9:
                rel_ok = (dV / max(abs(self._V_prev), 1.0e-9)) < -abs(self.lyapunov_rel_margin)
            passed = abs_ok or rel_ok
        else:
            passed = dV < self.lyapunov_eps
        mask = 1.0 if passed else 0.0
        return V_t, mask, dV, mask

    @staticmethod
    def _mean_window(values: collections.deque) -> Optional[float]:
        finite = [float(v) for v in values if math.isfinite(float(v))]
        if not finite:
            return None
        return sum(finite) / float(len(finite))

    def _compute_safety_block(self, V_t: float, filt_err: float, lyap_pass: float) -> Tuple[float, float, str]:
        self._lyap_pass_window.append(float(lyap_pass))
        pass_rate = self._mean_window(self._lyap_pass_window)
        pass_rate_f = 1.0 if pass_rate is None else float(pass_rate)

        if self._step_count <= self.drift_start_step:
            self._baseline_V_window.append(float(V_t))
            self._baseline_error_window.append(float(filt_err))
            return 0.0, pass_rate_f, ""

        if self.lyapunov_gate_mode != "guarded_drift":
            return 0.0, pass_rate_f, ""

        baseline_V = self._mean_window(self._baseline_V_window)
        baseline_err = self._mean_window(self._baseline_error_window)
        reasons: list[str] = []
        if (
            self.lyapunov_min_pass_rate > 0.0
            and len(self._lyap_pass_window) >= self.lyapunov_window_steps
            and pass_rate_f < self.lyapunov_min_pass_rate
        ):
            reasons.append("low_pass_rate")
        if baseline_V is not None:
            threshold = baseline_V * (1.0 + max(self.lyapunov_rel_margin, 0.0)) + max(self.lyapunov_abs_margin, 0.0)
            if V_t > threshold:
                reasons.append("V_above_baseline")
        if baseline_err is not None:
            threshold = baseline_err * (1.0 + max(self.lyapunov_rel_margin, 0.0)) + max(self.lyapunov_abs_margin, 0.0)
            if filt_err > threshold:
                reasons.append("error_above_baseline")
        if reasons:
            return 1.0, pass_rate_f, "+".join(reasons)
        return 0.0, pass_rate_f, ""

    def _default_error_signal(self) -> float:
        # Fallback: derive compound error from channel error vec (sum of squares).
        e = self._read_channel_error_vec()
        return float(torch.sum(e ** 2).item())

    def _compute_raw_error(self) -> float:
        if self.error_signal_callable is not None:
            try:
                return float(self.error_signal_callable(self.env))
            except Exception:
                return self._default_error_signal()
        return self._default_error_signal()

    def _capture_low_level(self) -> None:
        env = self.env.unwrapped
        delta_u = getattr(env, "_pid_value_add_buf", None)
        if delta_u is None:
            self.last_low_level = {}
            return
        if not isinstance(delta_u, torch.Tensor):
            self.last_low_level = {}
            return
        # 4-channel diagonal jacobian-inverse approximation (identity).
        j_inv_diag = torch.ones_like(delta_u)
        self.last_low_level = {
            "delta_u": delta_u.detach().clone(),
            "J_inv_diag": j_inv_diag.detach().clone(),
        }

    # ------------------------------------------------------------------
    # gym.Wrapper API
    # ------------------------------------------------------------------

    def step(self, action):
        result = self.env.step(action)
        # Tolerate both Gym/Gymnasium 4-tuple and 5-tuple flavours.
        if len(result) == 5:
            obs, reward, terminated, truncated, extras = result
            dones = terminated
            five_tuple = True
        else:
            obs, reward, dones, extras = result
            terminated = truncated = None
            five_tuple = False

        self._step_count += 1
        frac = self._apply_drift(self._step_count)
        raw_err = self._compute_raw_error()
        filt_err = self._filter_error(raw_err)
        V_t, stdw_mask, dV_t, lyap_pass = self._compute_lyapunov_mask()
        safety_block, lyap_pass_rate, safety_reason = self._compute_safety_block(V_t, filt_err, lyap_pass)
        self._V_prev = V_t
        self._capture_low_level()

        injected = {
            "stdw_raw_error": float(raw_err),
            "stdw_filt_error": float(filt_err),
            "stdw_drift_fraction": float(frac),
            "stdw_step": int(self._step_count),
            "stdw_V": float(V_t),
            "stdw_mask": float(stdw_mask),
            "stdw_dV": float(dV_t),
            "stdw_lyap_pass": float(lyap_pass),
            "stdw_lyap_pass_rate_window": float(lyap_pass_rate),
            "stdw_safety_block": float(safety_block),
            "stdw_safety_reason": str(safety_reason),
        }

        if isinstance(extras, dict):
            new_extras = {**extras, **injected}
        else:
            new_extras = injected

        # Persist for outer-wrapper-safe access.
        self.last_extras = dict(injected)

        if five_tuple:
            return obs, reward, terminated, truncated, new_extras
        return obs, reward, dones, new_extras

    def reset(self, *args, **kwargs):
        result = self.env.reset(*args, **kwargs)
        self._error_window.clear()
        self._V_prev = None
        self._e_prev = None
        # Re-apply current drift after env's _reset_domain may have restored base offsets.
        self._apply_drift(self._step_count)
        return result

    def current_drift(self) -> float:
        return self._compute_drift_fraction(self._step_count)


# ---------------------------------------------------------------------------
# Self-test (mock minimal env, no IsaacLab)
# ---------------------------------------------------------------------------


def _self_test() -> None:
    import numpy as np

    class _DummyUnwrapped:
        def __init__(self):
            self.com_to_cob_offsets = torch.zeros(1, 3)
            self._base_com_to_cob_offsets = torch.zeros(1, 3)
            self._pid_value_add_buf = torch.zeros(1, 4)
            self._tracking_error_rpy_depth = torch.zeros(1, 4)

    class _DummyEnv(gym.Env):
        observation_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(4,), dtype=np.float32)
        action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(4,), dtype=np.float32)

        def __init__(self):
            super().__init__()
            self._inner = _DummyUnwrapped()

        @property
        def unwrapped(self):
            return self._inner

        def reset(self, *args, **kwargs):
            return np.zeros(4, dtype=np.float32), {}

        def step(self, action):
            return np.zeros(4, dtype=np.float32), 0.0, False, False, {"foo": 1}

    env = _DummyEnv()
    wrapped = EasyUUVStdwWrapper(
        env,
        drift_start_step=2,
        drift_end_step=6,
        target_drift=0.1,
        drift_axes=(0,),
        enable_filter=True,
        filter_window_seconds=0.05,
        sim_dt_seconds=0.01,
    )
    wrapped.reset()

    # Drive 8 steps and verify boundary fractions.
    fracs = []
    for _ in range(8):
        _, _, _, _, info = wrapped.step(np.zeros(4, dtype=np.float32))
        fracs.append(info["stdw_drift_fraction"])
        assert math.isfinite(info["stdw_filt_error"]), "filter produced NaN"
    assert fracs[0] == 0.0, fracs
    assert fracs[-1] == 1.0, fracs
    print(f"[easyuuv_stdw_wrapper] self-test OK | fracs={fracs}")


if __name__ == "__main__":
    _self_test()

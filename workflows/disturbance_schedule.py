"""Gradual disturbance schedule for STDW workflow.

The schedule shares its time factor (linear ramp between drift_start_step and
drift_end_step) with ``EasyUUVStdwWrapper``'s ``_compute_drift_fraction`` so
all "external" perturbations (wave amplitude / base flow / observation noise /
thruster fault onset) are slowly time-varying — rather than step-injected — in
synchrony with the COB drift used by the STDW algorithm itself.

The schedule does NOT touch the policy / replay buffer / loss; it only calls
existing env APIs:
- ``env.apply_runtime_domain_shift`` — mode / base_vel / amplitude / frequency /
  noise_std / noise_corr (and jonswap_* via direct attr write since the public
  API does not expose them).
- ``env.set_thruster_fault`` — one-shot enable when crossing the fault threshold.
"""

from __future__ import annotations

import argparse
from typing import Optional

try:  # pragma: no cover - optional during isolated self-test
    from scenarios import ScenarioSpec, baseline_for, resolve_scenario
except ImportError:  # When imported as part of the workflow package
    from .scenarios import ScenarioSpec, baseline_for, resolve_scenario  # type: ignore


def _lerp(a: float, b: float, frac: float) -> float:
    return float(a) + float(frac) * (float(b) - float(a))


def _lerp_vec(a, b, frac: float) -> list:
    a = list(a)
    b = list(b)
    n = max(len(a), len(b))
    while len(a) < n:
        a.append(0.0)
    while len(b) < n:
        b.append(0.0)
    return [_lerp(a[i], b[i], frac) for i in range(n)]


class DisturbanceSchedule:
    """Linear ramp scheduler that pushes target disturbance into the env.

    Parameters
    ----------
    env_unwrapped : EasyUUVEnv
        The unwrapped Isaac Lab env (must expose apply_runtime_domain_shift,
        set_thruster_fault, cfg.disturbance_cfg).
    spec : ScenarioSpec
        Target scenario state.
    drift_start_step / drift_end_step : int
        Same convention as EasyUUVStdwWrapper. Outside this range the schedule
        is clamped (baseline before start, target after end).
    sim_dt_seconds : float
        Used to convert the scheduler's step counter into env-side sim time
        when calling set_thruster_fault.
    """

    def __init__(
        self,
        env_unwrapped,
        spec: ScenarioSpec,
        *,
        drift_start_step: int,
        drift_end_step: int,
        sim_dt_seconds: float = 1.0 / 120.0,
        ramp_shape: str = "linear",
    ) -> None:
        self.env = env_unwrapped
        self.spec = spec
        self.drift_start_step = int(drift_start_step)
        self.drift_end_step = int(drift_end_step)
        self.sim_dt_seconds = float(sim_dt_seconds)
        ramp_shape = str(ramp_shape).lower()
        if ramp_shape not in {"linear", "cosine", "step"}:
            raise ValueError(f"ramp_shape must be linear/cosine/step, got {ramp_shape!r}")
        self.ramp_shape = ramp_shape
        self.baseline = baseline_for(spec)
        self._fault_started = False

    # ------------------------------------------------------------------
    def fraction(self, step: int) -> float:
        if step <= self.drift_start_step:
            return 0.0
        if self.ramp_shape == "step":
            return 1.0
        if step >= self.drift_end_step:
            return 1.0
        denom = max(self.drift_end_step - self.drift_start_step, 1)
        linear_frac = float(step - self.drift_start_step) / float(denom)
        if self.ramp_shape == "cosine":
            import math as _m
            return 0.5 * (1.0 - _m.cos(_m.pi * linear_frac))
        return linear_frac

    # ------------------------------------------------------------------
    def reset(self) -> None:
        """Apply the t=0 baseline parameters once (after env.reset)."""
        bl = self.baseline
        try:
            self.env.apply_runtime_domain_shift(
                mode=bl.get("mode"),
                base_vel=bl.get("base_vel"),
                amplitude=bl.get("amplitude"),
                frequency=bl.get("frequency"),
                noise_std=bl.get("noise_std"),
                noise_corr=bl.get("noise_corr"),
                water_density_scale=1.0,
                torque_pulse_level=0.0,
            )
        except Exception as exc:  # pragma: no cover
            print(f"[DisturbanceSchedule] baseline apply_runtime_domain_shift failed: {exc}")

        # JONSWAP attrs (not in apply_runtime_domain_shift signature; safe to set directly).
        if self.spec.mode == "jonswap":
            disturbance_cfg = self.env.cfg.disturbance_cfg
            disturbance_cfg.jonswap_hs = float(bl.get("jonswap_hs", 0.0))
            disturbance_cfg.jonswap_fp = float(bl.get("jonswap_fp", 0.1))
            disturbance_cfg.jonswap_gamma = float(bl.get("jonswap_gamma", 3.3))
            disturbance_cfg.jonswap_depth = float(bl.get("jonswap_depth", 30.0))

        # Ensure fault is initially disabled.
        try:
            self.env.set_thruster_fault(enabled=False)
        except Exception:
            pass
        try:
            self.env.set_thruster_angle_shift(enabled=False)
        except Exception:
            pass
        self._fault_started = False

    # ------------------------------------------------------------------
    def tick(self, step: int) -> dict:
        """Advance by one step. Returns a snapshot dict for CSV logging."""
        frac = self.fraction(step)
        bl = self.baseline
        tgt = self.spec.target

        snapshot = {
            "scenario": self.spec.name,
            "disturbance_mode": self.spec.mode,
            "amp_x": 0.0,
            "amp_y": 0.0,
            "amp_z": 0.0,
            "noise_std_eff": None,
            "fault_active": False,
            "water_density_scale": 1.0,
            "water_rho": None,
            "fault_profile": self.spec.fault_profile or "",
            "fault_target_efficiency": self.spec.fault_target_efficiency,
            "torque_pulse_level": 0.0,
            "torque_pulse_active": False,
            "torque_pulse_x": 0.0,
            "torque_pulse_y": 0.0,
            "torque_pulse_z": 0.0,
            "thruster_angle_shift_rad": 0.0,
            "thruster_angle_shift_thrusters": "",
            "thruster_angle_shift_axis": "",
        }

        # ---------- ramp wave / current (mode != "none") ----------
        if self.spec.mode in {"sine", "constant", "jonswap"}:
            base_vel = _lerp_vec(bl.get("base_vel", [0.0] * 3), tgt.get("base_vel", [0.0] * 3), frac)
            amplitude = _lerp_vec(bl.get("amplitude", [0.0] * 3), tgt.get("amplitude", [0.0] * 3), frac)
            frequency = list(tgt.get("frequency", bl.get("frequency", [0.0] * 3)))  # frequency is constant
            noise_std = None
            noise_corr = None
            if self.spec.noise_std_target is not None:
                noise_std = _lerp(
                    float(bl.get("noise_std", 0.005)),
                    float(self.spec.noise_std_target),
                    frac,
                )
                noise_corr = float(self.spec.noise_corr) if self.spec.noise_corr is not None else None
                snapshot["noise_std_eff"] = noise_std
            try:
                self.env.apply_runtime_domain_shift(
                    mode=self.spec.mode,
                    base_vel=base_vel,
                    amplitude=amplitude,
                    frequency=frequency,
                    noise_std=noise_std,
                    noise_corr=noise_corr,
                )
            except Exception as exc:  # pragma: no cover
                print(f"[DisturbanceSchedule] tick apply_runtime_domain_shift failed: {exc}")

            snapshot["amp_x"] = float(amplitude[0]) if len(amplitude) > 0 else 0.0
            snapshot["amp_y"] = float(amplitude[1]) if len(amplitude) > 1 else 0.0
            snapshot["amp_z"] = float(amplitude[2]) if len(amplitude) > 2 else 0.0
            snapshot["base_vx"] = float(base_vel[0]) if len(base_vel) > 0 else 0.0
            snapshot["base_vy"] = float(base_vel[1]) if len(base_vel) > 1 else 0.0
            snapshot["base_vz"] = float(base_vel[2]) if len(base_vel) > 2 else 0.0

            # JONSWAP Hs ramp via direct attr write.
            if self.spec.mode == "jonswap":
                hs_target = float(tgt.get("jonswap_hs", 0.0))
                hs_baseline = float(bl.get("jonswap_hs", 0.0))
                hs_now = _lerp(hs_baseline, hs_target, frac)
                disturbance_cfg = self.env.cfg.disturbance_cfg
                disturbance_cfg.jonswap_hs = hs_now
                snapshot["jonswap_hs_eff"] = hs_now

        # ---------- water density scale ----------
        if self.spec.water_density_scale_target is not None:
            density_scale = _lerp(1.0, float(self.spec.water_density_scale_target), frac)
            try:
                self.env.apply_runtime_domain_shift(water_density_scale=density_scale)
            except Exception as exc:  # pragma: no cover
                print(f"[DisturbanceSchedule] water_density_scale failed: {exc}")
            snapshot["water_density_scale"] = float(density_scale)
            try:
                snapshot["water_rho"] = float(self.env.cfg.water_rho)
            except Exception:
                snapshot["water_rho"] = None

        # ---------- runtime torque pulse ----------
        if self.spec.torque_pulse_level is not None:
            try:
                self.env.apply_runtime_domain_shift(torque_pulse_level=float(self.spec.torque_pulse_level))
            except Exception as exc:  # pragma: no cover
                print(f"[DisturbanceSchedule] torque_pulse_level failed: {exc}")
            snapshot["torque_pulse_level"] = float(self.spec.torque_pulse_level)

        # ---------- thruster angle shift ----------
        if self.spec.thruster_angle_shift_rad is not None:
            angle_now = _lerp(0.0, float(self.spec.thruster_angle_shift_rad), frac)
            thrusters = list(self.spec.thruster_angle_shift_thrusters or [4])
            axis = str(self.spec.thruster_angle_shift_axis or "yaw")
            try:
                self.env.apply_runtime_domain_shift(
                    thruster_angle_shift_rad=angle_now,
                    thruster_angle_shift_thrusters=thrusters,
                    thruster_angle_shift_axis=axis,
                )
            except Exception as exc:  # pragma: no cover
                print(f"[DisturbanceSchedule] thruster_angle_shift failed: {exc}")
            snapshot["thruster_angle_shift_rad"] = float(angle_now)
            snapshot["thruster_angle_shift_thrusters"] = ",".join(str(x) for x in thrusters)
            snapshot["thruster_angle_shift_axis"] = axis

        # ---------- thruster fault: one-shot enable at threshold ----------
        if self.spec.fault_rate_per_second is not None:
            threshold = self.drift_start_step + int(self.spec.fault_start_offset_steps)
            if step >= threshold and not self._fault_started:
                try:
                    # NOTE: env-side fault uses episode_length_buf * sim.cfg.dt as the
                    # clock, which resets every episode (~3s). Passing a wall-clock
                    # `start_sim_time = step * sim_dt` would always exceed the
                    # per-episode timer and the threshold `current_sim_time >=
                    # start_sim_time` would never trigger. Instead we anchor the
                    # fault to the start of each episode (0.0) so degradation
                    # accumulates from t=0 within every episode after the
                    # drift threshold. This is the "intermittent thruster fault"
                    # interpretation used in RL ablations.
                    self.env.set_thruster_fault(
                        enabled=True,
                        start_sim_time=0.0,
                        fault_thrusters=list(self.spec.fault_thrusters or [4, 5]),
                        fault_rate_per_second=float(self.spec.fault_rate_per_second),
                        fault_profile=str(self.spec.fault_profile or "rate"),
                        target_efficiency=float(self.spec.fault_target_efficiency if self.spec.fault_target_efficiency is not None else 0.0),
                        ramp_duration_s=float(self.spec.fault_ramp_duration_s if self.spec.fault_ramp_duration_s is not None else 0.0),
                    )
                    self._fault_started = True
                except Exception as exc:  # pragma: no cover
                    print(f"[DisturbanceSchedule] set_thruster_fault failed: {exc}")
            snapshot["fault_active"] = bool(self._fault_started)

        # Probe current min thruster efficiency (best-effort).
        try:
            eff = self.env.thruster_efficiency_factors
            snapshot["fault_efficiency_min"] = float(eff.min().item())
        except Exception:
            snapshot["fault_efficiency_min"] = None
        try:
            active, vec = self.env.get_runtime_torque_pulse_state()
            row = vec[0]
            snapshot["torque_pulse_active"] = bool(active[0].item())
            snapshot["torque_pulse_x"] = float(row[0].item())
            snapshot["torque_pulse_y"] = float(row[1].item())
            snapshot["torque_pulse_z"] = float(row[2].item())
        except Exception:
            pass

        return snapshot


# ---------------------------------------------------------------------------
# Self test (no IsaacLab dependency)
# ---------------------------------------------------------------------------


class _MockEnv:
    """Tiny mock env exposing the four APIs DisturbanceSchedule touches."""

    class _Disturbance:
        mode = "none"
        base_vel = [0.0, 0.0, 0.0]
        amplitude = [0.0, 0.0, 0.0]
        frequency = [0.0, 0.0, 0.0]
        jonswap_hs = 0.0
        jonswap_fp = 0.1
        jonswap_gamma = 3.3
        jonswap_depth = 30.0

    class _Cfg:
        def __init__(self):
            self.disturbance_cfg = _MockEnv._Disturbance()

            class _Noise:
                enable_noise = False
                std_dev = 0.0
                correlation_coeff = 0.8

            self.noise_cfg = _Noise()
            self.water_rho = 997.0

    def __init__(self):
        self.cfg = _MockEnv._Cfg()
        self._torque_pulse_active = False
        self._torque_pulse_vec = [0.0, 0.0, 0.0]
        import math as _m

        class _Eff:
            def min(self):
                class _T:
                    def item(self_inner):
                        return 1.0
                return _T()

        self.thruster_efficiency_factors = _Eff()
        self.fault_state = None

    def apply_runtime_domain_shift(self, **kwargs):
        d = self.cfg.disturbance_cfg
        if kwargs.get("mode") is not None:
            d.mode = kwargs["mode"]
        if kwargs.get("base_vel") is not None:
            d.base_vel = list(kwargs["base_vel"])
        if kwargs.get("amplitude") is not None:
            d.amplitude = list(kwargs["amplitude"])
        if kwargs.get("frequency") is not None:
            d.frequency = list(kwargs["frequency"])
        if kwargs.get("noise_std") is not None:
            self.cfg.noise_cfg.enable_noise = True
            self.cfg.noise_cfg.std_dev = kwargs["noise_std"]
        if kwargs.get("water_density_scale") is not None:
            self.cfg.water_rho = 997.0 * float(kwargs["water_density_scale"])
        if kwargs.get("torque_pulse_level") is not None:
            self._torque_pulse_vec = [float(kwargs["torque_pulse_level"]), 0.0, 0.0]

    def set_thruster_fault(
        self,
        *,
        enabled,
        start_sim_time=0.0,
        fault_thrusters=None,
        fault_rate_per_second=0.02,
        fault_profile="rate",
        target_efficiency=0.0,
        ramp_duration_s=0.0,
    ):
        self.fault_state = {
            "enabled": enabled,
            "start_sim_time": start_sim_time,
            "fault_thrusters": list(fault_thrusters) if fault_thrusters else None,
            "fault_rate_per_second": fault_rate_per_second,
            "fault_profile": fault_profile,
            "target_efficiency": target_efficiency,
            "ramp_duration_s": ramp_duration_s,
        }

    def get_runtime_torque_pulse_state(self):
        import torch
        return torch.tensor([self._torque_pulse_active]), torch.tensor([self._torque_pulse_vec])


def _self_test() -> None:
    env = _MockEnv()
    spec = resolve_scenario("wave_plus_fault")
    sched = DisturbanceSchedule(env, spec, drift_start_step=10, drift_end_step=30, sim_dt_seconds=0.01)
    sched.reset()

    # Boundary 1: at step <= drift_start_step, amplitude must be baseline (0).
    snap = sched.tick(5)
    assert snap["amp_x"] == 0.0, snap
    assert sched.fraction(5) == 0.0
    assert env.fault_state is not None and env.fault_state["enabled"] is False

    # Boundary 2: at step == midpoint, amplitude must be ~half target (0.06).
    snap = sched.tick(20)
    assert abs(snap["amp_x"] - 0.06) < 1e-6, snap
    assert snap["fault_active"] is True  # 20 >= drift_start + offset(0)

    # Boundary 3: at step >= drift_end, amplitude == target (0.12).
    snap = sched.tick(40)
    assert abs(snap["amp_x"] - 0.12) < 1e-6, snap
    assert sched.fraction(40) == 1.0

    # JONSWAP ramp test.
    env2 = _MockEnv()
    spec_j = resolve_scenario("jonswap_strong")
    sched_j = DisturbanceSchedule(env2, spec_j, drift_start_step=0, drift_end_step=10, sim_dt_seconds=0.01)
    sched_j.reset()
    sched_j.tick(0)
    assert env2.cfg.disturbance_cfg.jonswap_hs == 0.0
    sched_j.tick(5)
    assert abs(env2.cfg.disturbance_cfg.jonswap_hs - 1.25) < 1e-6, env2.cfg.disturbance_cfg.jonswap_hs
    sched_j.tick(11)
    assert abs(env2.cfg.disturbance_cfg.jonswap_hs - 2.5) < 1e-6, env2.cfg.disturbance_cfg.jonswap_hs

    print("[disturbance_schedule] self-test OK")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--self-test", action="store_true", default=False)
    args = parser.parse_args()
    if args.self_test:
        _self_test()
    else:
        print("Use --self-test to run the offline mock test.")

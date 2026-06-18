"""Scenario presets for STDW evaluation.

Single source of truth for the disturbance scenarios and embodiment names that
the STDW workflow consumes. The presets are migrated from
``custom_workflows/exp_wave_disturbance.py::DISTURBANCE_SCENARIOS`` and
``custom_workflows/exp_cross_embodiment.py::EMBODIMENT_TYPES``.

The schedule (``disturbance_schedule.DisturbanceSchedule``) treats these as the
**target** parameters; the t=0 baseline is implicit (zero amplitude / zero base
flow / zero noise / fault disabled), so a scenario is "gradually injected" by
linearly ramping from baseline to target between drift_start_step and
drift_end_step.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class ScenarioSpec:
    """Target-state description of a single disturbance scenario."""

    name: str
    mode: str  # "none" | "constant" | "sine" | "jonswap"
    target: dict = field(default_factory=dict)
    noise_std_target: Optional[float] = None
    noise_corr: Optional[float] = None
    fault_thrusters: Optional[list] = None
    fault_rate_per_second: Optional[float] = None
    fault_start_offset_steps: int = 0
    fault_profile: Optional[str] = None
    fault_target_efficiency: Optional[float] = None
    fault_ramp_duration_s: Optional[float] = None
    water_density_scale_target: Optional[float] = None
    torque_pulse_level: Optional[float] = None
    thruster_angle_shift_rad: Optional[float] = None
    thruster_angle_shift_thrusters: Optional[list] = None
    thruster_angle_shift_axis: Optional[str] = None


# Scenario target parameters. Keep aligned with
# ``审稿人回应参考草稿.md`` and ``exp_wave_disturbance.DISTURBANCE_SCENARIOS``.
SCENARIO_PRESETS: dict = {
    "none": ScenarioSpec(
        name="none",
        mode="none",
        target={"base_vel": [0.0, 0.0, 0.0], "amplitude": [0.0, 0.0, 0.0], "frequency": [0.0, 0.0, 0.0]},
    ),
    "sine": ScenarioSpec(
        name="sine",
        mode="sine",
        target={
            "base_vel": [0.06, 0.0, 0.02],
            "amplitude": [0.10, 0.04, 0.03],
            "frequency": [0.16, 0.22, 0.30],
        },
    ),
    "current_bias": ScenarioSpec(
        name="current_bias",
        mode="constant",
        target={
            "base_vel": [0.35, 0.08, 0.0],
            "amplitude": [0.0, 0.0, 0.0],
            "frequency": [0.0, 0.0, 0.0],
        },
    ),
    "jonswap_mild": ScenarioSpec(
        name="jonswap_mild",
        mode="jonswap",
        target={
            "base_vel": [0.0, 0.0, 0.0],
            "amplitude": [0.0, 0.0, 0.0],
            "frequency": [0.0, 0.0, 0.0],
            "jonswap_hs": 1.0,
            "jonswap_fp": 0.12,
            "jonswap_gamma": 3.3,
            "jonswap_depth": 30.0,
        },
    ),
    "jonswap_strong": ScenarioSpec(
        name="jonswap_strong",
        mode="jonswap",
        target={
            "base_vel": [0.0, 0.0, 0.0],
            "amplitude": [0.0, 0.0, 0.0],
            "frequency": [0.0, 0.0, 0.0],
            "jonswap_hs": 2.5,
            "jonswap_fp": 0.10,
            "jonswap_gamma": 3.3,
            "jonswap_depth": 30.0,
        },
    ),
    "current_plus_jonswap": ScenarioSpec(
        name="current_plus_jonswap",
        mode="jonswap",
        target={
            "base_vel": [0.35, 0.08, 0.0],
            "amplitude": [0.0, 0.0, 0.0],
            "frequency": [0.0, 0.0, 0.0],
            "jonswap_hs": 0.8,
            "jonswap_fp": 0.11,
            "jonswap_gamma": 3.3,
            "jonswap_depth": 30.0,
        },
    ),
    "wave_plus_noise": ScenarioSpec(
        name="wave_plus_noise",
        mode="sine",
        target={
            "base_vel": [0.06, 0.0, 0.02],
            "amplitude": [0.10, 0.04, 0.03],
            "frequency": [0.16, 0.22, 0.30],
        },
        noise_std_target=0.04,
        noise_corr=0.9,
    ),
    "wave_plus_fault": ScenarioSpec(
        name="wave_plus_fault",
        mode="sine",
        target={
            "base_vel": [0.08, 0.0, 0.02],
            "amplitude": [0.12, 0.05, 0.02],
            "frequency": [0.16, 0.22, 0.30],
        },
        noise_std_target=0.04,
        noise_corr=0.9,
        fault_thrusters=[4, 5],
        fault_rate_per_second=0.35,
        fault_start_offset_steps=0,
    ),
    "cob_shift_x": ScenarioSpec(
        name="cob_shift_x",
        mode="none",
        target={"base_vel": [0.0, 0.0, 0.0], "amplitude": [0.0, 0.0, 0.0], "frequency": [0.0, 0.0, 0.0]},
    ),
    "thruster_single_fault": ScenarioSpec(
        name="thruster_single_fault",
        mode="none",
        target={"base_vel": [0.0, 0.0, 0.0], "amplitude": [0.0, 0.0, 0.0], "frequency": [0.0, 0.0, 0.0]},
        fault_thrusters=[4],
        fault_rate_per_second=0.0,
        fault_profile="fixed",
        fault_target_efficiency=0.5,
        fault_ramp_duration_s=5.0,
    ),
    "thruster_pair_fault": ScenarioSpec(
        name="thruster_pair_fault",
        mode="none",
        target={"base_vel": [0.0, 0.0, 0.0], "amplitude": [0.0, 0.0, 0.0], "frequency": [0.0, 0.0, 0.0]},
        fault_thrusters=[4, 5],
        fault_rate_per_second=0.0,
        fault_profile="fixed",
        fault_target_efficiency=0.5,
        fault_ramp_duration_s=5.0,
    ),
    "density_095": ScenarioSpec(
        name="density_095",
        mode="none",
        target={"base_vel": [0.0, 0.0, 0.0], "amplitude": [0.0, 0.0, 0.0], "frequency": [0.0, 0.0, 0.0]},
        water_density_scale_target=0.95,
    ),
    "torque_pulse_medium": ScenarioSpec(
        name="torque_pulse_medium",
        mode="none",
        target={"base_vel": [0.0, 0.0, 0.0], "amplitude": [0.0, 0.0, 0.0], "frequency": [0.0, 0.0, 0.0]},
        torque_pulse_level=0.5,
    ),
    "torque_pulse_strong": ScenarioSpec(
        name="torque_pulse_strong",
        mode="none",
        target={"base_vel": [0.0, 0.0, 0.0], "amplitude": [0.0, 0.0, 0.0], "frequency": [0.0, 0.0, 0.0]},
        torque_pulse_level=1.0,
    ),
    "thruster_angle_yaw_p5deg": ScenarioSpec(
        name="thruster_angle_yaw_p5deg",
        mode="none",
        target={"base_vel": [0.0, 0.0, 0.0], "amplitude": [0.0, 0.0, 0.0], "frequency": [0.0, 0.0, 0.0]},
        thruster_angle_shift_rad=0.0872664626,
        thruster_angle_shift_thrusters=[4],
        thruster_angle_shift_axis="yaw",
    ),
    "thruster_angle_yaw_m5deg": ScenarioSpec(
        name="thruster_angle_yaw_m5deg",
        mode="none",
        target={"base_vel": [0.0, 0.0, 0.0], "amplitude": [0.0, 0.0, 0.0], "frequency": [0.0, 0.0, 0.0]},
        thruster_angle_shift_rad=-0.0872664626,
        thruster_angle_shift_thrusters=[4],
        thruster_angle_shift_axis="yaw",
    ),
    "thruster_angle_yaw_p10deg": ScenarioSpec(
        name="thruster_angle_yaw_p10deg",
        mode="none",
        target={"base_vel": [0.0, 0.0, 0.0], "amplitude": [0.0, 0.0, 0.0], "frequency": [0.0, 0.0, 0.0]},
        thruster_angle_shift_rad=0.1745329252,
        thruster_angle_shift_thrusters=[4],
        thruster_angle_shift_axis="yaw",
    ),
}


EMBODIMENT_PRESETS = ("base", "long_body", "heavy_moderate", "asymmetric")


def list_scenarios() -> list:
    return list(SCENARIO_PRESETS.keys())


def list_embodiments() -> list:
    return list(EMBODIMENT_PRESETS)


def resolve_scenario(
    name: str,
    *,
    fault_thrusters: Optional[list] = None,
    fault_rate_per_second: Optional[float] = None,
    fault_start_offset_steps: Optional[int] = None,
    fault_profile: Optional[str] = None,
    fault_target_efficiency: Optional[float] = None,
    fault_ramp_duration_s: Optional[float] = None,
    water_density_scale_target: Optional[float] = None,
    torque_pulse_level: Optional[float] = None,
    thruster_angle_shift_rad: Optional[float] = None,
    thruster_angle_shift_thrusters: Optional[list] = None,
    thruster_angle_shift_axis: Optional[str] = None,
) -> ScenarioSpec:
    """Look up a preset by name and apply optional CLI overrides.

    Returns a *copy* so callers can mutate freely without affecting the registry.
    """
    if name not in SCENARIO_PRESETS:
        raise KeyError(f"Unknown scenario: {name}. Available: {list_scenarios()}")
    base = SCENARIO_PRESETS[name]
    override = ScenarioSpec(
        name=base.name,
        mode=base.mode,
        target=dict(base.target),
        noise_std_target=base.noise_std_target,
        noise_corr=base.noise_corr,
        fault_thrusters=list(base.fault_thrusters) if base.fault_thrusters else None,
        fault_rate_per_second=base.fault_rate_per_second,
        fault_start_offset_steps=base.fault_start_offset_steps,
        fault_profile=base.fault_profile,
        fault_target_efficiency=base.fault_target_efficiency,
        fault_ramp_duration_s=base.fault_ramp_duration_s,
        water_density_scale_target=base.water_density_scale_target,
        torque_pulse_level=base.torque_pulse_level,
        thruster_angle_shift_rad=base.thruster_angle_shift_rad,
        thruster_angle_shift_thrusters=list(base.thruster_angle_shift_thrusters) if base.thruster_angle_shift_thrusters else None,
        thruster_angle_shift_axis=base.thruster_angle_shift_axis,
    )
    if fault_thrusters is not None:
        override.fault_thrusters = list(fault_thrusters)
    if fault_rate_per_second is not None:
        override.fault_rate_per_second = float(fault_rate_per_second)
    if fault_start_offset_steps is not None:
        override.fault_start_offset_steps = int(fault_start_offset_steps)
    if fault_profile is not None:
        override.fault_profile = str(fault_profile)
    if fault_target_efficiency is not None:
        override.fault_target_efficiency = float(fault_target_efficiency)
    if fault_ramp_duration_s is not None:
        override.fault_ramp_duration_s = float(fault_ramp_duration_s)
    if water_density_scale_target is not None:
        override.water_density_scale_target = float(water_density_scale_target)
    if torque_pulse_level is not None:
        override.torque_pulse_level = float(torque_pulse_level)
    if thruster_angle_shift_rad is not None:
        override.thruster_angle_shift_rad = float(thruster_angle_shift_rad)
    if thruster_angle_shift_thrusters is not None:
        override.thruster_angle_shift_thrusters = list(thruster_angle_shift_thrusters)
    if thruster_angle_shift_axis is not None:
        override.thruster_angle_shift_axis = str(thruster_angle_shift_axis)
    return override


def baseline_for(spec: ScenarioSpec) -> dict:
    """Return the t=0 starting parameters for a given scenario.

    All amplitudes / base flows / jonswap_hs start at 0; mode mirrors the target
    so env._compute_dynamics takes the right branch (with zero magnitude).
    Noise std starts at a small floor (0.005) when the scenario asks for noise,
    so the env is always running with some baseline observation noise.
    """
    base = {
        "mode": spec.mode if spec.mode != "none" else "none",
        "base_vel": [0.0, 0.0, 0.0],
        "amplitude": [0.0, 0.0, 0.0],
        "frequency": list(spec.target.get("frequency", [0.0, 0.0, 0.0])),
    }
    if spec.mode == "jonswap":
        base["jonswap_hs"] = 0.0
        base["jonswap_fp"] = float(spec.target.get("jonswap_fp", 0.1))
        base["jonswap_gamma"] = float(spec.target.get("jonswap_gamma", 3.3))
        base["jonswap_depth"] = float(spec.target.get("jonswap_depth", 30.0))
    if spec.noise_std_target is not None:
        base["noise_std"] = 0.005
        base["noise_corr"] = float(spec.noise_corr) if spec.noise_corr is not None else 0.8
    return base


def _self_test() -> None:
    print("Available scenarios:", list_scenarios())
    print("Available embodiments:", list_embodiments())
    for name in list_scenarios():
        spec = resolve_scenario(name)
        bl = baseline_for(spec)
        print(f"  - {name}: mode={spec.mode}, target_keys={sorted(spec.target.keys())}, "
              f"baseline_amp={bl['amplitude']}, fault_rate={spec.fault_rate_per_second}")
    # Override smoke test
    spec = resolve_scenario("wave_plus_fault", fault_rate_per_second=0.5, fault_thrusters=[2, 3])
    assert spec.fault_rate_per_second == 0.5
    assert spec.fault_thrusters == [2, 3]
    print("[scenarios] self-test OK")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--self-test", action="store_true", default=False)
    args = parser.parse_args()
    if args.self_test:
        _self_test()
    else:
        for n in list_scenarios():
            print(n)

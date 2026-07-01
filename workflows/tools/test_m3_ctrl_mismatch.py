"""M3 standalone numeric test: controller-mismatch dynamic range.

Replicates the EasyUUV thrust pipeline math (S-Surface -> allocation -> clip ->
PWM quadratic -> rotor-constant conversion -> efficiency factor) WITHOUT Isaac,
to demonstrate the §1.2 diagnosis:

  - CM-A (pid_gain): scaling the P gain is swallowed by sigmoid saturation +
    the gain-independent additive term (self_adapt) + downstream clip, so the
    final thrust authority barely moves -> the ~67% mismatch ceiling.
  - CM-B (actuator_scale): scaling the thruster efficiency acts on the
    motorValue->thrust *linear* segment and produces a near-proportional change
    in final thrust -> order-of-magnitude dynamic range, breaking the ceiling.
  - CM-C (s_surface_struct): removing the additive floor (add_scale=0) re-couples
    the P-gain mismatch to the output.

Run:  python workflows/tools/test_m3_ctrl_mismatch.py
"""

from __future__ import annotations

import torch


# --- Pipeline constants mirrored from easyuuv_env.py (EasyUUVEnvCfg) ----------
S_RATIO = 4.0
ACTION_LIM = torch.tensor([0.15, 0.15, 0.3, 1.0])
ROTOR_CONSTANT = 0.1 / 100.0
PWM_THRESHOLD = 0.02
# roll-axis nominal S-surface gains (PID_init_args row 0, already /PID_PWM_value).
P_NOM = 1.0   # zeta1
D_NOM = 0.133  # zeta2


def _pwm_to_radps(motor: torch.Tensor) -> torch.Tensor:
    """Replicate the quadratic PWM->rad/s map + deadzone in _compute_dynamics."""
    out = motor.clone()
    pos = out >= PWM_THRESHOLD
    neg = out <= -PWM_THRESHOLD
    dead = out.abs() < PWM_THRESHOLD
    out[dead] = 0.0
    out[pos] = -139.0 * out[pos] ** 2 + 500.0 * out[pos] + 8.28
    out[neg] = 161.0 * out[neg] ** 2 + 517.86 * out[neg] - 5.72
    return out


def final_thrust(
    action: float,
    action_d: float,
    *,
    p_scale: float = 1.0,        # CM-A: scale P gain
    s_ratio_scale: float = 1.0,  # CM-C: scale sigmoid slope
    add_scale: float = 1.0,      # CM-C: scale additive floor
    thrust_scale: float = 1.0,   # CM-B: scale thruster efficiency
    self_adapt: bool = True,
) -> float:
    """Compute steady-state thrust magnitude on one thruster for a roll command.

    We isolate the roll channel (PID_value[:,0]) and route it through a single
    representative thruster (motorValue[:,1] = +roll), then through the static
    PWM->rad/s->thrust map. Thruster first-order lag is at steady state (identity).
    """
    a = action * float(ACTION_LIM[0])
    a_d = action_d * float(ACTION_LIM[0])
    s_ratio_eff = S_RATIO * s_ratio_scale
    p_eff = P_NOM * p_scale

    pid_value = 2.0 / (1.0 + torch.exp(torch.tensor(-s_ratio_eff * p_eff * a - s_ratio_eff * D_NOM * a_d))) - 1.0
    if self_adapt:
        add = 30.0 * (1.0 / 60.0) * (a + 1.0 * a_d) * add_scale
        add = max(-0.35, min(0.35, add))
        pid_value = pid_value + add

    # single thruster sees +PID_value(roll); clip to PWM range
    motor = torch.clip(pid_value.reshape(1), -1.0, 1.0)
    radps = _pwm_to_radps(motor)
    thrust = ROTOR_CONSTANT * torch.abs(radps) * radps  # convert()
    thrust = thrust * thrust_scale                      # efficiency factor (CM-B)
    return float(thrust.item())


def _dynamic_range(values: list[float]) -> float:
    """max/min ratio of |thrust| over a sweep (bigger = more sensitive)."""
    mags = [abs(v) for v in values if abs(v) > 1e-12]
    if not mags:
        return 1.0
    return max(mags) / max(min(mags), 1e-12)


def main() -> None:
    torch.manual_seed(0)
    action, action_d = 1.0, 0.0  # strong roll command, policy saturated

    # Baseline (no mismatch) must be stable across calls.
    base = final_thrust(action, action_d)
    base2 = final_thrust(action, action_d)
    assert abs(base - base2) < 1e-9, "baseline must be deterministic"
    print(f"[OK] baseline thrust = {base:.4f}")

    # --- CM-A: scale P gain over 4 decades (1.0 -> 0.0001) -------------------
    sweep_factors = [1.0, 0.1, 0.01, 0.001]
    cm_a = [final_thrust(action, action_d, p_scale=f, self_adapt=True) for f in sweep_factors]
    range_a = _dynamic_range(cm_a)
    print(f"[CM-A pid_gain]        thrusts={[f'{v:.3f}' for v in cm_a]}  range={range_a:.2f}x")

    # --- CM-B: scale thruster efficiency over same decades ------------------
    cm_b = [final_thrust(action, action_d, thrust_scale=f, self_adapt=True) for f in sweep_factors]
    range_b = _dynamic_range(cm_b)
    print(f"[CM-B actuator_scale]  thrusts={[f'{v:.3f}' for v in cm_b]}  range={range_b:.2f}x")

    # --- CM-C: remove additive floor, then P-gain mismatch bites ------------
    cm_c = [final_thrust(action, action_d, p_scale=f, add_scale=0.0, self_adapt=True) for f in sweep_factors]
    range_c = _dynamic_range(cm_c)
    print(f"[CM-C add_scale=0]     thrusts={[f'{v:.3f}' for v in cm_c]}  range={range_c:.2f}x")

    # --- Assertions encoding the §1.2 diagnosis -----------------------------
    # The core ceiling effect is a *thrust floor*: CM-A cannot drive the output
    # near zero (sigmoid saturation + additive term + PWM offset hold it up),
    # whereas CM-B scales the linear thrust segment all the way down.
    floor_a = min(abs(v) for v in cm_a)
    floor_b = min(abs(v) for v in cm_b)
    # 1) CM-A retains a large residual thrust floor (cannot kill authority).
    assert floor_a > 1.0, f"CM-A should keep a thrust floor >1.0, got {floor_a:.3f}"
    # 2) CM-B drives thrust effectively to zero (no floor).
    assert floor_b < 0.1, f"CM-B should reach near-zero thrust, got {floor_b:.3f}"
    # 3) CM-B is at least an order of magnitude wider in dynamic range.
    assert range_b > range_a * 10.0, "CM-B must be >=10x more sensitive than CM-A"
    assert range_b > 100.0, f"CM-B should give >100x range, got {range_b:.2f}x"
    # 4) Removing the additive floor (CM-C) re-couples the P mismatch and
    #    lowers the floor below CM-A's.
    floor_c = min(abs(v) for v in cm_c)
    assert floor_c < floor_a, "CM-C (add_scale=0) must lower the floor below CM-A"
    print(f"\n[OK] thrust floor: CM-A={floor_a:.3f} (stuck) vs CM-B={floor_b:.3f} (killable); "
          f"CM-B range {range_b:.1f}x >> CM-A {range_a:.1f}x")
    print("     -> mismatch retargeting breaks the saturation ceiling.")
    print("All M3 ctrl-mismatch tests PASSED.")


if __name__ == "__main__":
    main()

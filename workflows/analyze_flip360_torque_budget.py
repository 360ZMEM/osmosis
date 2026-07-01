#!/usr/bin/env python3
"""Flip360 物理边界（力矩预算）分析。

回答"满 ±π 后空翻是否物理可达"：把控制力矩预算（8 推进器经二次推力曲线 + S 面
sigmoid 饱和 + 力臂）与倒置区浮力回正力矩（r_com_to_cob × F_buoy）逐轴对比。

常数全部取自 easyuuv_env.py / thruster_dynamics.py / rigid_body_hydrodynamics.py：
  rho=997, volume=0.022748, g=9.81, rotor_constant=0.001,
  推力曲线 spd= -139 x^2+500 x+8.28 (x>=0.02) / 161 x^2+517.86 x-5.72 (x<=-0.02),
  thrust = rotor_constant*|spd|*spd,
  S 面 PID_value = 2/(1+exp(-s_ratio*P*a - s_ratio*D*ad))-1, s_ratio=4,
  roll/pitch action_lim=0.15, P_roll=0.6/0.6=1.0, gain_beta=0.2,
  垂直推进器 y 力臂=length*0.375=0.21, x 力臂=width*0.3=0.129,
  com_to_cob: base[0,0,0.01], asym[0.05,0.05,0.01]。

用法：python workflows/analyze_flip360_torque_budget.py
"""
import math

# ---- physical constants ----
rho = 997.0
vol = 0.022747843530591776
g = 9.81
mass = 22.701
rotor_constant = 0.1 / 100.0  # 0.001

F_buoy = rho * vol * g
W = mass * g
print(f"F_buoy = {F_buoy:.3f} N   weight = {W:.3f} N  (neutrally buoyant check)")


# ---- thrust curve: PWM(motorValue) -> motor 'speed' -> thrust ----
def pwm_to_thrust(x):
    if abs(x) < 0.02:
        return 0.0
    if x >= 0.02:
        spd = -139.0 * x * x + 500.0 * x + 8.28
    else:
        spd = 161.0 * x * x + 517.86 * x - 5.72
    return rotor_constant * abs(spd) * spd


print("\nPer-thruster thrust vs PWM command:")
for x in [0.1, 0.2, 0.29, 0.5, 0.75, 1.0]:
    print(f"  PWM={x:+.2f} -> thrust = {pwm_to_thrust(x):8.3f} N")

# ---- max PWM the S-surface low-level law can emit (steady, ad~0) ----
s_ratio = 4.0
P_roll = 0.6 / 0.6
a_scaled = 0.15
arg = s_ratio * P_roll * a_scaled
pid_val_max = 2.0 / (1.0 + math.exp(-arg)) - 1.0
print(f"\nS-surface max single-axis PID_value (roll/pitch) = {pid_val_max:.3f}")
print(f"  -> per-thruster thrust at that PWM = {pwm_to_thrust(pid_val_max):.3f} N")

# ---- torque budgets (4 vertical thrusters, body-z force, lever arms) ----
length = 0.56
width = 0.43
y_arm = length * 0.375   # 0.21 m  (roll arm)
x_arm = width * 0.3      # 0.129 m (pitch arm)


def roll_torque(pwm):
    return 4.0 * y_arm * pwm_to_thrust(pwm)


def pitch_torque(pwm):
    return 4.0 * x_arm * pwm_to_thrust(pwm)


print(f"\nRoll torque budget:")
print(f"  at PWM=1.0 (hard clip)        : {roll_torque(1.0):8.2f} N*m")
print(f"  at S-surface max ({pid_val_max:.2f})      : {roll_torque(pid_val_max):8.2f} N*m")
print(f"Pitch torque budget:")
print(f"  at PWM=1.0 (hard clip)        : {pitch_torque(1.0):8.2f} N*m")
print(f"  at S-surface max ({pid_val_max:.2f})      : {pitch_torque(pid_val_max):8.2f} N*m")


# ---- buoyancy restoring torque = |r x F_buoy|, max over tilt (sin=1) ----
def restoring_max(offset):
    r = math.sqrt(sum(c * c for c in offset))
    return r * F_buoy


base_off = [0.0, 0.0, 0.01]
asym_off = [0.05, 0.05, 0.01]
print(f"\nBuoyancy restoring torque (max over tilt, sin(theta)=1 ~ theta=pi/2):")
print(f"  base |r|={math.sqrt(sum(c*c for c in base_off)):.4f} m -> {restoring_max(base_off):.3f} N*m")
print(f"  asym |r|={math.sqrt(sum(c*c for c in asym_off)):.4f} m -> {restoring_max(asym_off):.3f} N*m")
print(f"  (at exact inversion theta=pi: restoring -> 0 but UNSTABLE equilibrium)")

# ---- meta gain-tuner +gain_beta authority ----
gain_beta = 0.2
arg_hi = s_ratio * P_roll * (1.0 + gain_beta) * a_scaled
pid_val_hi = 2.0 / (1.0 + math.exp(-arg_hi)) - 1.0
print(f"\nWith meta gain-tuner at +{int(gain_beta*100)}% authority: PID_value={pid_val_hi:.3f}")
print(f"  roll torque  = {roll_torque(pid_val_hi):6.2f} N*m")
print(f"  pitch torque = {pitch_torque(pid_val_hi):6.2f} N*m")

# ---- per-axis verdict ----
print(f"\n*** VERDICT ***")
print(f"[1] Hard actuator ceiling (PWM=1.0 clip) : roll {roll_torque(1.0):.1f} Nm / pitch {pitch_torque(1.0):.1f} Nm")
print(f"    vs max restoring  base {restoring_max(base_off):.2f} Nm / asym {restoring_max(asym_off):.2f} Nm")
print(f"    -> actuator torque EXCEEDS restoring by {pitch_torque(1.0)/restoring_max(asym_off):.0f}x (asym pitch, worst case).")
print(f"    => Full +/-pi flip is NOT actuator-torque-limited; physically achievable in principle.")
print(f"\n[2] Nominal operating point (S-surface, action_lim=0.15, a~1):")
print(f"    per-axis authority only roll {roll_torque(pid_val_max):.1f} / pitch {pitch_torque(pid_val_max):.1f} Nm.")
print(f"    asym worst-case restoring {restoring_max(asym_off):.2f} Nm approaches nominal pitch authority {pitch_torque(pid_val_max):.1f} Nm.")
print(f"    => To overcome restoring in the inverted region the policy must drive actions deep into")
print(f"       the sigmoid saturation tail (a>>1, coarse/bang-bang) - a hard-to-learn regime.")
print(f"\n[3] Dominant boundary = unstable inverted equilibrium + Euler-axis control singularity")
print(f"    + limited learnable bandwidth at the nominal operating point.")
print(f"    NOT removable by reward shaping / curriculum (F1/F2/F3/F4) - those do not change the")
print(f"    control architecture or the operating-point ceiling. Needs SO(3)/geometric attitude")
print(f"    control or larger action_lim / redesigned allocation to break.")

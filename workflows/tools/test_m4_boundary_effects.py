"""M4 standalone synthetic test: near-boundary effects (B1-B5 + ventilation).

Exercises boundary_effects.BoundaryEffectModels WITHOUT Isaac by feeding
scripted root pose / buoyancy / drag tensors. Validates the §5 plan invariants:

  - off            : any_enabled is False -> wrench is *exactly* zero (zero
                     behaviour change, the plug-and-play default).
  - B1 residual_buoyancy : +ΔB along world-up = frac·m·g.
  - B2 free_surface      : buoyancy+drag scaled by (s-1); submersion ratio s(t)
                           clips to 1 fully submerged, 0 fully breached.
  - B3 thruster_ventilation : per-thruster efficiency = clip((Zs - z_thr)/H, 0, 1);
                              a breached thruster -> 0, an identity run -> all ones.
  - B4 ground_effect     : near-floor suction (world -Z), gated by h_dist<threshold.
  - B5 nonlinear_restoring : explicit COB/COG offset torque τ = r_B×F_B + r_G×F_G.

A stub omni.isaac.lab.utils.math (quat_apply/quat_conjugate, w-x-y-z) is injected
so the module runs offline.

Run:  python workflows/tools/test_m4_boundary_effects.py
"""

from __future__ import annotations

import sys
import types
from pathlib import Path

import torch


REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def _install_math_stub() -> None:
    """Inject a minimal omni.isaac.lab.utils.math with quat ops (w,x,y,z)."""
    try:
        from omni.isaac.lab.utils.math import quat_apply, quat_conjugate  # noqa: F401
        return
    except Exception:
        pass

    def quat_apply(q: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        # rotate vector v by quaternion q = (w, x, y, z)
        w = q[..., 0:1]
        xyz = q[..., 1:]
        t = 2.0 * torch.cross(xyz, v, dim=-1)
        return v + w * t + torch.cross(xyz, t, dim=-1)

    def quat_conjugate(q: torch.Tensor) -> torch.Tensor:
        return torch.cat([q[..., 0:1], -q[..., 1:]], dim=-1)

    mod_chain = [
        "omni", "omni.isaac", "omni.isaac.lab", "omni.isaac.lab.utils",
        "omni.isaac.lab.utils.math",
    ]
    for name in mod_chain:
        if name not in sys.modules:
            sys.modules[name] = types.ModuleType(name)
    m = sys.modules["omni.isaac.lab.utils.math"]
    m.quat_apply = quat_apply
    m.quat_conjugate = quat_conjugate
    sys.modules["omni.isaac.lab.utils"].math = m


_install_math_stub()

from boundary_effects import BoundaryEffectModels, flags_from_mode  # noqa: E402


DEV = torch.device("cpu")
MASS = 22.701
G = 9.81
WEIGHT = MASS * G  # ~222.7 N
IDENT_Q = torch.tensor([[1.0, 0.0, 0.0, 0.0]])           # no rotation (w,x,y,z)
FLIP_X_Q = torch.tensor([[0.0, 1.0, 0.0, 0.0]])          # 180 deg about x


def _model(**kwargs) -> BoundaryEffectModels:
    base = dict(num_envs=1, device=DEV, vehicle_height=0.3, z_surface=3.0, z_bottom=0.0)
    base.update(kwargs)
    return BoundaryEffectModels(**base)


def _wrench(m: BoundaryEffectModels, *, pos, quat=IDENT_Q,
            buoy=None, buoy_tau=None, drag=None, drag_tau=None):
    pos = torch.as_tensor(pos, dtype=torch.float32).reshape(1, 3)
    if buoy is None:
        buoy = torch.tensor([[0.0, 0.0, WEIGHT]])
    if buoy_tau is None:
        buoy_tau = torch.zeros((1, 3))
    if drag is None:
        drag = torch.zeros((1, 3))
    if drag_tau is None:
        drag_tau = torch.zeros((1, 3))
    return m.compute_boundary_wrench(
        root_pos_w=pos,
        root_quat_w=quat,
        masses=torch.tensor([[MASS]]),
        com_to_cob_offsets=torch.tensor([[0.0, 0.0, 0.01]]),
        g_mag=G,
        buoyancy_forces_b=buoy,
        buoyancy_torques_b=buoy_tau,
        drag_forces_b=drag,
        drag_torques_b=drag_tau,
    )


def test_flags_from_mode() -> None:
    assert flags_from_mode("off") == {
        "enable_residual_buoyancy": False, "enable_free_surface": False,
        "enable_ventilation": False, "enable_ground_effect": False,
        "enable_nonlinear_restoring": False,
    }
    f = flags_from_mode("free_surface")
    assert f["enable_free_surface"] and f["enable_ventilation"]
    assert not f["enable_residual_buoyancy"]
    full = flags_from_mode("full")
    assert all(full.values())
    print("[OK] flags_from_mode: off/free_surface/full mapping correct")


def test_off_is_zero() -> None:
    m = _model()
    m.apply_mode("off")
    assert not m.any_enabled
    df, dt, info = _wrench(m, pos=[0, 0, 1.5])
    assert torch.allclose(df, torch.zeros_like(df))
    assert torch.allclose(dt, torch.zeros_like(dt))
    assert info == {}
    print("[OK] off: wrench exactly zero, zero behaviour change")


def test_residual_buoyancy() -> None:
    m = _model(residual_buoyancy_frac=0.015)
    m.apply_mode("residual_buoyancy")
    df, dt, info = _wrench(m, pos=[0, 0, 1.5])
    expected = 0.015 * WEIGHT
    assert abs(float(df[0, 2]) - expected) < 1e-3, df
    assert abs(float(df[0, 0])) < 1e-6 and abs(float(df[0, 1])) < 1e-6
    print(f"[OK] B1 residual_buoyancy: dF_z={float(df[0,2]):.3f} == frac*mg={expected:.3f}")


def test_free_surface_submersion() -> None:
    m = _model()
    m.apply_mode("free_surface")
    # fully submerged -> s=1 -> no force change
    s_sub = float(m.submersion_ratio(torch.tensor([[0.0, 0.0, 1.5]])))
    df_sub, _, info_sub = _wrench(m, pos=[0, 0, 1.5])
    assert abs(s_sub - 1.0) < 1e-6
    assert abs(float(df_sub[0, 2])) < 1e-4, df_sub
    # partially submerged: z - H/2 = 2.85 -> z=3.0 -> s=0.5
    s_half = float(m.submersion_ratio(torch.tensor([[0.0, 0.0, 3.0]])))
    df_half, _, _ = _wrench(m, pos=[0, 0, 3.0])
    assert abs(s_half - 0.5) < 1e-6, s_half
    assert abs(float(df_half[0, 2]) - (-0.5 * WEIGHT)) < 1e-2, df_half
    # fully breached: z - H/2 = 3.15 -> z=3.3 -> s=0 -> full buoyancy removed
    s_out = float(m.submersion_ratio(torch.tensor([[0.0, 0.0, 3.3]])))
    df_out, _, _ = _wrench(m, pos=[0, 0, 3.3])
    assert abs(s_out - 0.0) < 1e-6, s_out
    assert abs(float(df_out[0, 2]) - (-WEIGHT)) < 1e-2, df_out
    print(f"[OK] B2 free_surface: s(submerged)=1, s(half)=0.5, s(breached)=0; "
          f"dF_z half={float(df_half[0,2]):.2f} breached={float(df_out[0,2]):.2f}")


def test_ventilation_factor() -> None:
    m = _model()
    m.apply_mode("free_surface")  # also enables ventilation
    # two thrusters: one high (breaches), one low (submerged)
    offsets = torch.tensor([[[0.0, 0.0, 0.2], [0.0, 0.0, -0.2]]])  # (1,2,3)
    fac = m.compute_ventilation_factor(
        root_pos_w=torch.tensor([[0.0, 0.0, 2.95]]),
        root_quat_w=IDENT_Q,
        thruster_com_offsets=offsets,
    )
    # high thruster z_w = 2.95+0.2=3.15 -> (3.0-3.15)/0.3 = -0.5 -> 0
    # low thruster  z_w = 2.95-0.2=2.75 -> (3.0-2.75)/0.3 = 0.833
    assert abs(float(fac[0, 0]) - 0.0) < 1e-6, fac
    assert abs(float(fac[0, 1]) - 0.8333) < 1e-3, fac
    # disabled -> identity
    m.enable_ventilation = False
    fac_off = m.compute_ventilation_factor(
        root_pos_w=torch.tensor([[0.0, 0.0, 2.95]]),
        root_quat_w=IDENT_Q,
        thruster_com_offsets=offsets,
    )
    assert torch.allclose(fac_off, torch.ones_like(fac_off))
    print(f"[OK] B3 ventilation: breached={float(fac[0,0]):.3f} submerged={float(fac[0,1]):.3f}; "
          f"disabled->ones")


def test_ground_effect() -> None:
    m = _model(ground_effect_coeff=0.15, ground_effect_gamma=2.0,
               ground_effect_threshold=0.5)
    m.apply_mode("ground_effect")
    # near floor h=0.4 < 0.5 -> active; ratio=D/h=0.3/0.4=0.75
    df_near, _, info = _wrench(m, pos=[0, 0, 0.4])
    expected = 0.15 * WEIGHT * (0.3 / 0.4) ** 2
    assert float(df_near[0, 2]) < 0.0, df_near  # suction toward floor
    assert abs(abs(float(df_near[0, 2])) - expected) < 1e-1, (df_near, expected)
    # far h=1.0 > 0.5 -> inactive
    df_far, _, _ = _wrench(m, pos=[0, 0, 1.0])
    assert abs(float(df_far[0, 2])) < 1e-6, df_far
    print(f"[OK] B4 ground_effect: near dF_z={float(df_near[0,2]):.2f} (suction, ~-{expected:.2f}); "
          f"far=0")


def test_nonlinear_restoring() -> None:
    # lateral COB offset -> nonzero pitch torque under vertical buoyancy.
    m = _model(r_cob=(0.05, 0.0, 0.0), r_cog=(0.0, 0.0, 0.0))
    m.apply_mode("nonlinear_restoring")
    _, dt, _ = _wrench(m, pos=[0, 0, 1.5], buoy=torch.tensor([[0.0, 0.0, WEIGHT]]))
    # cross([0.05,0,0],[0,0,W]) = [0, -0.05*W, 0]
    assert abs(float(dt[0, 1]) - (-0.05 * WEIGHT)) < 1e-2, dt
    assert abs(float(dt[0, 0])) < 1e-6 and abs(float(dt[0, 2])) < 1e-6
    print(f"[OK] B5 nonlinear_restoring: tau_y={float(dt[0,1]):.3f} == -r_cob_x*W")


def main() -> None:
    torch.manual_seed(0)
    test_flags_from_mode()
    test_off_is_zero()
    test_residual_buoyancy()
    test_free_surface_submersion()
    test_ventilation_factor()
    test_ground_effect()
    test_nonlinear_restoring()
    print("\nAll M4 boundary-effect tests PASSED")


if __name__ == "__main__":
    main()

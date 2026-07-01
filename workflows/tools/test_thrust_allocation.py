"""TAM standalone synthetic test: config-driven thrust allocation matrix.

Validates thrust_allocation.py against the easyuuv_env runtime force/torque
summation (L1790-1806) and the hard-coded mixing block (L1663-1670) WITHOUT
Isaac, by rebuilding the 8-thruster geometry from thruster_dynamics.py and
reconstructing net body wrenches from the allocation matrix B.

Invariants (plan Step 1):
  (a) build_wrench_matrix(layout) f_i columns == quat_apply(quat_from_rpy(rpy), xhat)
      per thruster, and tau_i == r_i x f_i (matches runtime L1790-1802).
  (b) Config B+ path: unit control command [roll]/[pitch]/[yaw]/[depth] reconstructs
      a net wrench whose target axis is positive (restoring) and off-axes ~0
      (decoupled). The legacy hard-coded mixing block's net-wrench signs equal
      geo_channel_sign=[-1,+1,-1] (roll tau_x neg, pitch tau_y pos, yaw tau_z neg),
      i.e. geo_channel_sign is exactly the compensation that aligns the hard-coded
      output with the config path's natural restoring sign. depth -> +Fz on both.
  (c) uuv4 under-actuated geometry (4 vertical thrusters only) + weighted least
      squares with controllable_dofs=["heave","roll","pitch"]: a yaw command
      reconstructs tau_z ~ 0 (structurally + weight-masked), while roll stays
      controllable (tau_x > 0).

A stub omni.isaac.lab.utils.math (quat_apply/quat_conjugate, w-x-y-z) is injected
so the module runs offline.

Run:  python workflows/tools/test_thrust_allocation.py
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

from omni.isaac.lab.utils.math import quat_apply as omni_quat_apply  # noqa: E402
from thrust_allocation import (  # noqa: E402
    ThrusterLayout,
    allocate,
    build_wrench_matrix,
    control_channels_to_wrench,
    dof_weight_vector,
    quat_from_rpy,
)


# 8-thruster geometry mirrored from thruster_dynamics.get_thruster_com_and_orientations.
# rows: [x, y, z, roll, pitch, yaw]; ordering matches the runtime thruster index 0..7.
_L = 0.56   # length (x)
_W = 0.43   # width  (y)
_P90 = -1.5708  # vertical thruster pitch (-90 deg)
SPECS_8 = [
    [_W * 0.3,  _L * 0.375, 0.03, 0.0, _P90, 0.0],          # 0 front_left_vertical
    [_W * 0.3, -_L * 0.375, 0.03, 0.0, _P90, 0.0],          # 1 front_right_vertical
    [-_W * 0.3,  _L * 0.375, 0.03, 0.0, _P90, 0.0],         # 2 rear_left_vertical
    [-_W * 0.3, -_L * 0.375, 0.03, 0.0, _P90, 0.0],         # 3 rear_right_vertical
    [_W * 0.2,  _L * 0.2, -0.02, 0.0, 0.0, -0.785398],      # 4 front_left_horizontal  (-45)
    [_W * 0.2, -_L * 0.2, -0.02, 0.0, 0.0,  0.785398],      # 5 front_right_horizontal (+45)
    [-_W * 0.2,  _L * 0.2, -0.02, 0.0, 0.0, -2.356194],     # 6 rear_left_horizontal  (-135)
    [-_W * 0.2, -_L * 0.2, -0.02, 0.0, 0.0,  2.356194],     # 7 rear_right_horizontal (+135)
]

# body 6-DOF row indices [Fx,Fy,Fz,Tx,Ty,Tz]; control channel -> wrench row.
ROW = {"roll": 3, "pitch": 4, "yaw": 5, "depth": 2}
GEO_CHANNEL_SIGN = [-1.0, 1.0, -1.0]  # cfg easyuuv_env.py: legacy mix-block net torque signs


def _unit_cmd(channel: str) -> torch.Tensor:
    """[roll, pitch, yaw, depth] unit command on one channel."""
    idx = ("roll", "pitch", "yaw", "depth").index(channel)
    cmd = torch.zeros(4)
    cmd[idx] = 1.0
    return cmd


def _hardcoded_mix(pid: torch.Tensor) -> torch.Tensor:
    """Legacy 8-thruster mixing block (easyuuv_env._pid_control L1663-1670)."""
    roll, pitch, yaw, depth = pid[0], pid[1], pid[2], pid[3]
    mv = torch.zeros(8)
    mv[0] = -roll - pitch + depth
    mv[1] = roll - pitch + depth
    mv[2] = -roll + pitch + depth
    mv[3] = roll + pitch + depth
    mv[4] = yaw
    mv[5] = -yaw
    mv[6] = -yaw
    mv[7] = yaw
    return mv


def test_force_matrix_matches_runtime() -> None:
    """(a) B columns reproduce the runtime per-thruster force/torque exactly."""
    layout = ThrusterLayout.from_specs(SPECS_8)
    B = build_wrench_matrix(layout)
    xhat = torch.tensor([1.0, 0.0, 0.0])
    for i, row in enumerate(SPECS_8):
        x, y, z, rr, rp, ry = row
        q = quat_from_rpy(rr, rp, ry)
        f_runtime = omni_quat_apply(q, xhat)                 # matches L1790-1791
        assert torch.allclose(B[0:3, i], f_runtime, atol=1e-5), (
            f"thruster {i} force column mismatch: {B[0:3, i]} vs {f_runtime}"
        )
        r = torch.tensor([x, y, z])
        tau_runtime = torch.cross(r, f_runtime, dim=-1)      # matches L1802
        assert torch.allclose(B[3:6, i], tau_runtime, atol=1e-5), (
            f"thruster {i} torque column mismatch: {B[3:6, i]} vs {tau_runtime}"
        )
    print("[OK] (a) build_wrench_matrix f_i/tau_i columns == runtime quat_apply + cross")


def test_pinv_signs_and_decoupling() -> None:
    """(b) config B+ gives positive decoupled wrench; legacy signs == geo_channel_sign."""
    layout = ThrusterLayout.from_specs(SPECS_8)
    B = build_wrench_matrix(layout)

    # --- config B+ path: target axis positive, other control axes decoupled ---
    for channel in ("roll", "pitch", "yaw", "depth"):
        w = control_channels_to_wrench(_unit_cmd(channel))
        u = allocate(B, w, mode="pinv")
        net = B @ u
        tgt = ROW[channel]
        assert net[tgt] > 0.9, f"config {channel}: target row {tgt} = {net[tgt]:.4f} not ~+1"
        for other in (set(ROW.values()) - {tgt}):
            assert abs(net[other]) < 1e-3, (
                f"config {channel}: off-axis row {other} = {net[other]:.4e} not ~0"
            )
    print("[OK] (b1) config B+: each channel -> +target axis, decoupled off-axes")

    # --- legacy hard-coded mixing block: net torque signs == geo_channel_sign ---
    legacy_signs = []
    for channel in ("roll", "pitch", "yaw"):
        mv = _hardcoded_mix(_unit_cmd(channel))
        net_hc = B @ mv                                       # runtime net wrench
        tau = net_hc[3:6]
        rot_row = ROW[channel] - 3                            # 0=Tx,1=Ty,2=Tz
        legacy_signs.append(float(torch.sign(tau[rot_row])))
    assert legacy_signs == GEO_CHANNEL_SIGN, (
        f"legacy net-torque signs {legacy_signs} != geo_channel_sign {GEO_CHANNEL_SIGN}"
    )
    # geo_channel_sign * legacy torque -> restoring (positive) on each rotational axis.
    for k, channel in enumerate(("roll", "pitch", "yaw")):
        mv = _hardcoded_mix(_unit_cmd(channel))
        net_hc = B @ mv
        rot_row = ROW[channel] - 3
        assert GEO_CHANNEL_SIGN[k] * float(net_hc[3 + rot_row]) > 0.0
    # depth -> +Fz on the legacy path too.
    net_depth = B @ _hardcoded_mix(_unit_cmd("depth"))
    assert net_depth[ROW["depth"]] > 0.0, "legacy depth did not produce +Fz"
    print("[OK] (b2) legacy mix-block net-torque signs == geo_channel_sign [-1,+1,-1]; depth->+Fz")


def test_underactuated_wls_masks_yaw() -> None:
    """(c) uuv4 (4 vertical) WLS: yaw command -> tau_z ~ 0, roll stays controllable."""
    layout4 = ThrusterLayout.from_specs(SPECS_8[:4])
    B4 = build_wrench_matrix(layout4)
    weight = dof_weight_vector(["heave", "roll", "pitch"])   # no yaw

    w_yaw = control_channels_to_wrench(_unit_cmd("yaw"))
    u_yaw = allocate(B4, w_yaw, mode="wls", weight=weight)
    net_yaw = B4 @ u_yaw
    assert abs(net_yaw[ROW["yaw"]]) < 1e-6, (
        f"uuv4 yaw command leaked tau_z = {net_yaw[ROW['yaw']]:.4e}"
    )

    w_roll = control_channels_to_wrench(_unit_cmd("roll"))
    u_roll = allocate(B4, w_roll, mode="wls", weight=weight)
    net_roll = B4 @ u_roll
    assert net_roll[ROW["roll"]] > 0.0, "uuv4 roll not controllable (tau_x <= 0)"
    # roll thrusts carry a ~4e-6 spurious tau_z purely from the -1.5708 (vs exact -pi/2)
    # vertical-thruster pitch mirrored from thruster_dynamics.py; physically negligible.
    assert abs(net_roll[ROW["yaw"]]) < 1e-4, "uuv4 roll leaked tau_z"
    print("[OK] (c) uuv4 WLS: yaw masked (tau_z~0), roll controllable (tau_x>0)")


def main() -> None:
    torch.manual_seed(0)
    test_force_matrix_matches_runtime()
    test_pinv_signs_and_decoupling()
    test_underactuated_wls_masks_yaw()
    print("\nAll TAM tests PASSED")


if __name__ == "__main__":
    main()

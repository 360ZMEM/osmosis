"""
@file thrust_allocation.py
@brief Config-driven thrust allocation matrix (TAM) for multi-embodiment UUVs.
@details
  Lifts the hard-coded per-axis thruster mixing block (easyuuv_env._pid_control) into an
  explicit allocation matrix B in R^{6xN} built from thruster geometry, supporting
  fully-actuated, redundant, non-orthogonal, and under-actuated layouts.

  Conventions (must match easyuuv_env runtime, thruster_dynamics.get_thruster_com_and_orientations):
    - Each thruster produces a unit body-frame force along its local +x axis, rotated by its
      orientation quaternion: f_i = R(q_i) . xhat.
    - Torque about the body COM: tau_i = r_i x f_i, with r_i the com->thruster offset.
    - Quaternions are (w, x, y, z) order (Isaac Lab quat_apply convention).
    - Body 6-DOF wrench order: [Fx(surge), Fy(sway), Fz(heave), Tx(roll), Ty(pitch), Tz(yaw)].
    - Control channels [roll, pitch, yaw, depth] map to wrench [Tx, Ty, Tz, Fz]; surge/sway = 0.

  This module intentionally imports only torch + stdlib (no omni.isaac.*) so it stays
  importable and unit-testable in a clean Python environment.

  Author: EasyUUV-STDW multi-embodiment extension, 2026-07.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Sequence

import torch


# 控制通道 -> body 6-DOF wrench 的索引映射：[roll, pitch, yaw, depth] -> [Tx, Ty, Tz, Fz]
CONTROL_CHANNEL_NAMES = ("roll", "pitch", "yaw", "depth")
# body 6-DOF: [Fx, Fy, Fz, Tx, Ty, Tz]
_CHANNEL_TO_WRENCH_ROW = {"roll": 3, "pitch": 4, "yaw": 5, "depth": 2}


def quat_apply(quat: torch.Tensor, vec: torch.Tensor) -> torch.Tensor:
    """Rotate ``vec`` by quaternion ``quat`` (w, x, y, z), active rotation.

    Matches omni.isaac.lab.utils.math.quat_apply so the TAM built here is numerically
    consistent with the runtime force/torque summation in easyuuv_env.
    """
    quat = torch.as_tensor(quat, dtype=torch.float32)
    vec = torch.as_tensor(vec, dtype=torch.float32)
    w = quat[..., 0:1]
    xyz = quat[..., 1:4]
    t = 2.0 * torch.cross(xyz, vec, dim=-1)
    return vec + w * t + torch.cross(xyz, t, dim=-1)


def quat_from_rpy(roll: float, pitch: float, yaw: float) -> torch.Tensor:
    """Roll-pitch-yaw (rad) -> quaternion (w, x, y, z).

    Replicates thruster_dynamics.create_tf_rpyquat / quat_from_euler_xyz so layouts declared
    with rpy match the existing hard-coded 8-thruster geometry exactly.
    """
    cy, sy = math.cos(yaw * 0.5), math.sin(yaw * 0.5)
    cp, sp = math.cos(pitch * 0.5), math.sin(pitch * 0.5)
    cr, sr = math.cos(roll * 0.5), math.sin(roll * 0.5)
    w = cr * cp * cy + sr * sp * sy
    x = sr * cp * cy - cr * sp * sy
    y = cr * sp * cy + sr * cp * sy
    z = cr * cp * sy - sr * sp * cy
    return torch.tensor([w, x, y, z], dtype=torch.float32)


@dataclass
class ThrusterLayout:
    """Geometry of a thruster set for one vehicle.

    Attributes:
        positions: (N, 3) com->thruster offsets r_i (meters).
        orientations: (N, 4) thruster orientation quaternions (w, x, y, z).
        num_thrusters: N.
    """

    positions: torch.Tensor
    orientations: torch.Tensor
    num_thrusters: int

    def __post_init__(self) -> None:
        self.positions = torch.as_tensor(self.positions, dtype=torch.float32)
        self.orientations = torch.as_tensor(self.orientations, dtype=torch.float32)
        if self.positions.shape != (self.num_thrusters, 3):
            raise ValueError(
                f"positions must be ({self.num_thrusters}, 3), got {tuple(self.positions.shape)}"
            )
        if self.orientations.shape != (self.num_thrusters, 4):
            raise ValueError(
                f"orientations must be ({self.num_thrusters}, 4), got {tuple(self.orientations.shape)}"
            )

    @classmethod
    def from_specs(cls, specs: Sequence[Sequence[float]]) -> "ThrusterLayout":
        """Build from a list of [x, y, z, roll, pitch, yaw] rows (position + rpy orientation)."""
        positions = []
        orientations = []
        for row in specs:
            x, y, z, rr, rp, ry = row
            positions.append([float(x), float(y), float(z)])
            orientations.append(quat_from_rpy(float(rr), float(rp), float(ry)))
        n = len(specs)
        return cls(
            positions=torch.tensor(positions, dtype=torch.float32),
            orientations=torch.stack(orientations, dim=0) if orientations else torch.zeros((0, 4)),
            num_thrusters=n,
        )


def build_wrench_matrix(layout: ThrusterLayout) -> torch.Tensor:
    """Build the 6xN allocation matrix B mapping thrust magnitudes -> body wrench.

    B[:, i] = [f_i; r_i x f_i], f_i = R(q_i) . xhat. Then wrench = B @ u for u in R^N.
    """
    xhat = torch.tensor([1.0, 0.0, 0.0], dtype=torch.float32).expand(layout.num_thrusters, 3)
    f = quat_apply(layout.orientations, xhat)          # (N, 3)
    tau = torch.cross(layout.positions, f, dim=-1)     # (N, 3)
    B = torch.zeros((6, layout.num_thrusters), dtype=torch.float32)
    B[0:3, :] = f.transpose(0, 1)
    B[3:6, :] = tau.transpose(0, 1)
    return B


def control_channels_to_wrench(cmd: torch.Tensor) -> torch.Tensor:
    """[roll, pitch, yaw, depth] (..., 4) -> body wrench [Fx,Fy,Fz,Tx,Ty,Tz] (..., 6).

    surge/sway commanded to zero; depth -> heave(Fz); roll/pitch/yaw -> Tx/Ty/Tz.
    """
    cmd = torch.as_tensor(cmd, dtype=torch.float32)
    wrench = torch.zeros(cmd.shape[:-1] + (6,), dtype=torch.float32, device=cmd.device)
    for ch_idx, name in enumerate(CONTROL_CHANNEL_NAMES):
        wrench[..., _CHANNEL_TO_WRENCH_ROW[name]] = cmd[..., ch_idx]
    return wrench


def dof_weight_vector(controllable_dofs: Optional[Sequence[str]]) -> torch.Tensor:
    """Diagonal weight (6,) over [Fx,Fy,Fz,Tx,Ty,Tz]; 0 for uncontrollable DOFs.

    ``controllable_dofs`` names use the wrench axes {"surge","sway","heave","roll","pitch","yaw"}.
    None => all controllable (weight 1). Under-actuated layouts (e.g. uuv4 without yaw) pass
    a subset so weighted least squares does not penalize unreachable directions.
    """
    axis_row = {"surge": 0, "sway": 1, "heave": 2, "roll": 3, "pitch": 4, "yaw": 5}
    if controllable_dofs is None:
        return torch.ones(6, dtype=torch.float32)
    w = torch.zeros(6, dtype=torch.float32)
    for name in controllable_dofs:
        if name not in axis_row:
            raise ValueError(f"unknown DOF name {name!r}; expected one of {list(axis_row)}")
        w[axis_row[name]] = 1.0
    return w


def allocate(
    B: torch.Tensor,
    wrench_cmd: torch.Tensor,
    mode: str = "pinv",
    weight: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Allocate a desired body wrench to thruster magnitudes.

    Args:
        B: (6, N) allocation matrix from build_wrench_matrix.
        wrench_cmd: (..., 6) desired body wrench [Fx,Fy,Fz,Tx,Ty,Tz].
        mode: "pinv" (minimum-norm least squares, u = B^+ w) or
              "wls" (weighted least squares, u = (B^T W^2 B)^+ B^T W^2 w) for under-actuated.
        weight: (6,) diagonal weights for "wls"; ignored for "pinv". None => all-ones.

    Returns:
        (..., N) thruster magnitudes.
    """
    B = torch.as_tensor(B, dtype=torch.float32)
    wrench_cmd = torch.as_tensor(wrench_cmd, dtype=torch.float32)
    if mode == "pinv":
        A = B
        w = wrench_cmd
    elif mode == "wls":
        wv = torch.ones(6, dtype=torch.float32) if weight is None else torch.as_tensor(weight, dtype=torch.float32)
        A = wv.unsqueeze(-1) * B                    # (6, N)
        w = wrench_cmd * wv                          # (..., 6)
    else:
        raise ValueError(f"unknown allocate mode {mode!r}; expected 'pinv' or 'wls'")
    Binv = torch.linalg.pinv(A)                       # (N, 6)
    return torch.einsum("nk,...k->...n", Binv, w)

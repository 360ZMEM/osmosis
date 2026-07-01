"""
@file boundary_effects.py
@brief M4 near-boundary hydrodynamic effects for the EasyUUV environment.
@details
  Plug-and-play, *additive* body-frame wrench corrections that model the
  Sim2Real near-boundary physics described in ref/近边界效应.md:

    B1 residual_buoyancy   : slightly-positive trim error (ΔB = frac·m·g, world-up).
    B2 free_surface        : depth-dependent partial-submersion ratio s(t) that
                             linearly attenuates buoyancy + drag near the surface.
    B3 thruster_ventilation: per-thruster efficiency multiplier (a thruster that
                             breaches the surface loses thrust authority).
    B4 ground_effect       : near-floor suction F_ground = F_nom·(D/h)^γ.
    B5 nonlinear_restoring : explicit COG/COB offset torque
                             τ = (R r_B)×F_B + (R r_G)×F_G that sharpens the 180°
                             instability bifurcation during 360° flips.

  Design contract (matches PLAN §5):
    - The module ONLY returns additive corrections; the env adds them to the
      existing force/torque synthesis (easyuuv_env._compute_dynamics).
    - All wrenches are returned in the *body* frame to match the hydro pipeline.
    - Every sub-effect has an independent boolean flag; ``boundary_effect_mode``
      maps to a preset combination (see ``flags_from_mode``).
    - With all flags False the returned wrench is exactly zero -> zero behaviour
      change (default ``off``).

  This module is pure torch (only quat ops from omni.isaac.lab.utils.math) so it
  can be unit-tested offline with a math stub.

  Authors: STDW/meta extensions 2026-06.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Tuple

import torch

from omni.isaac.lab.utils.math import quat_apply, quat_conjugate


_MODE_PRESETS: Dict[str, Tuple[str, ...]] = {
    "off": (),
    "residual_buoyancy": ("residual_buoyancy",),
    "free_surface": ("free_surface", "thruster_ventilation"),
    "ground_effect": ("ground_effect",),
    "nonlinear_restoring": ("nonlinear_restoring",),
    "full": (
        "residual_buoyancy",
        "free_surface",
        "thruster_ventilation",
        "ground_effect",
        "nonlinear_restoring",
    ),
}

_FLAG_NAMES = (
    "enable_residual_buoyancy",
    "enable_free_surface",
    "enable_ventilation",
    "enable_ground_effect",
    "enable_nonlinear_restoring",
)

_EFFECT_TO_FLAG = {
    "residual_buoyancy": "enable_residual_buoyancy",
    "free_surface": "enable_free_surface",
    "thruster_ventilation": "enable_ventilation",
    "ground_effect": "enable_ground_effect",
    "nonlinear_restoring": "enable_nonlinear_restoring",
}


def flags_from_mode(mode: str) -> Dict[str, bool]:
    """Map a ``boundary_effect_mode`` string to the per-effect boolean flags."""
    enabled = _MODE_PRESETS.get(str(mode), ())
    flags = {name: False for name in _FLAG_NAMES}
    for effect in enabled:
        flags[_EFFECT_TO_FLAG[effect]] = True
    return flags


@dataclass
class BoundaryEffectModels:
    """Additive near-boundary wrench generator (body frame).

    All geometry constants default to physically reasonable values but are only
    consulted when the corresponding flag is True. ``z`` is the world Z of the
    vehicle root (root_pos_w[:, 2]); the surface is at ``z_surface`` and the pool
    floor at ``z_bottom`` (both world Z).
    """

    num_envs: int
    device: torch.device

    # --- per-effect switches (default all off -> zero wrench) ---
    enable_residual_buoyancy: bool = False
    enable_free_surface: bool = False
    enable_ventilation: bool = False
    enable_ground_effect: bool = False
    enable_nonlinear_restoring: bool = False

    # --- B1 residual buoyancy ---
    # +frac -> positive (upward) residual buoyancy as a fraction of weight m·g.
    residual_buoyancy_frac: float = 0.015

    # --- B2 free surface / B3 ventilation geometry ---
    z_surface: float = 3.0
    vehicle_height: float = 0.3

    # --- B4 ground effect ---
    z_bottom: float = 0.0
    ground_effect_coeff: float = 0.15      # F_nom = coeff · m·g
    ground_effect_gamma: float = 2.0       # exponent γ
    ground_effect_threshold: float = 0.5   # only active when h_dist < threshold (m)
    ground_effect_cap_mg: float = 2.0      # clamp |F_ground| to cap·m·g (numerical safety)

    # --- B5 nonlinear restoring (body-frame COG/COB offsets) ---
    r_cog: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    r_cob: Tuple[float, float, float] = (0.0, 0.0, 0.0)

    debug: bool = False

    _eps: float = field(default=1.0e-6, init=False, repr=False)

    @property
    def any_enabled(self) -> bool:
        return bool(
            self.enable_residual_buoyancy
            or self.enable_free_surface
            or self.enable_ventilation
            or self.enable_ground_effect
            or self.enable_nonlinear_restoring
        )

    def apply_mode(self, mode: str) -> None:
        """Set the per-effect flags from a ``boundary_effect_mode`` preset."""
        for name, val in flags_from_mode(mode).items():
            setattr(self, name, val)

    # ------------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------------
    def _world_up_in_body(self, root_quat_w: torch.Tensor) -> torch.Tensor:
        """world +Z expressed in the body frame, shape (N, 3)."""
        up_w = torch.zeros((root_quat_w.shape[0], 3), device=root_quat_w.device, dtype=root_quat_w.dtype)
        up_w[..., 2] = 1.0
        return quat_apply(quat_conjugate(root_quat_w), up_w)

    def submersion_ratio(self, root_pos_w: torch.Tensor) -> torch.Tensor:
        """Free-surface submersion ratio s(t) ∈ [0, 1] per env, shape (N,).

        s = clip((Z_surface - (z - H/2)) / H, 0, 1):
          fully submerged (z + H/2 <= Z_surface) -> 1,
          fully breached  (z - H/2 >= Z_surface) -> 0.
        """
        z = root_pos_w[..., 2]
        h = max(float(self.vehicle_height), self._eps)
        s = (self.z_surface - (z - 0.5 * h)) / h
        return torch.clamp(s, 0.0, 1.0)

    # ------------------------------------------------------------------
    # main wrench
    # ------------------------------------------------------------------
    def compute_boundary_wrench(
        self,
        *,
        root_pos_w: torch.Tensor,        # (N, 3) world position
        root_quat_w: torch.Tensor,       # (N, 4) world orientation
        masses: torch.Tensor,            # (N, 1)
        com_to_cob_offsets: torch.Tensor,  # (N, 3) body-frame COM->COB
        g_mag: float,
        buoyancy_forces_b: torch.Tensor,   # (N, 3) nominal buoyancy (body frame)
        buoyancy_torques_b: torch.Tensor,  # (N, 3)
        drag_forces_b: torch.Tensor,       # (N, 3) density+viscosity (body frame)
        drag_torques_b: torch.Tensor,      # (N, 3)
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
        """Return additive (delta_forces_b, delta_torques_b, info).

        The deltas are meant to be *added* to the existing force/torque sum.
        info carries diagnostics (submersion ratio, residual magnitude, ...).
        """
        N = root_pos_w.shape[0]
        dtype = root_pos_w.dtype
        dev = root_pos_w.device
        df = torch.zeros((N, 3), device=dev, dtype=dtype)
        dt = torch.zeros((N, 3), device=dev, dtype=dtype)
        info: Dict[str, torch.Tensor] = {}

        if not self.any_enabled:
            return df, dt, info

        masses = masses.reshape(N, 1).to(device=dev, dtype=dtype)
        weight = masses * float(g_mag)  # (N,1)
        up_b = self._world_up_in_body(root_quat_w)  # (N,3)

        # --- B1 residual buoyancy (ΔB along world up, torque about COM via COB) ---
        if self.enable_residual_buoyancy:
            dB = float(self.residual_buoyancy_frac) * weight  # (N,1)
            f_res = up_b * dB
            df = df + f_res
            dt = dt + torch.cross(com_to_cob_offsets, f_res, dim=-1)
            info["residual_dB"] = dB.squeeze(-1)

        # --- B2 free surface: attenuate buoyancy + drag by (s-1) ---
        if self.enable_free_surface:
            s = self.submersion_ratio(root_pos_w)  # (N,)
            scale = (s - 1.0).reshape(N, 1)         # 0 when submerged, -1 fully out
            df = df + scale * buoyancy_forces_b + scale * drag_forces_b
            dt = dt + scale * buoyancy_torques_b + scale * drag_torques_b
            info["submersion_ratio"] = s

        # --- B4 ground effect: near-floor suction (world-down) ---
        if self.enable_ground_effect:
            z = root_pos_w[..., 2]
            h_dist = (z - self.z_bottom).clamp_min(self._eps)  # (N,)
            active = (h_dist < float(self.ground_effect_threshold)).reshape(N, 1)
            D = max(float(self.vehicle_height), self._eps)
            ratio = (D / (h_dist + self._eps)).reshape(N, 1)
            mag = float(self.ground_effect_coeff) * weight * torch.pow(ratio, float(self.ground_effect_gamma))
            mag = torch.clamp(mag, max=float(self.ground_effect_cap_mg) * weight)
            mag = torch.where(active, mag, torch.zeros_like(mag))
            f_ground = -up_b * mag  # suction toward the floor (world -Z)
            df = df + f_ground
            info["ground_mag"] = mag.squeeze(-1)

        # --- B5 explicit nonlinear restoring torque ---
        if self.enable_nonlinear_restoring:
            r_cob = torch.tensor(self.r_cob, device=dev, dtype=dtype).reshape(1, 3).expand(N, 3)
            r_cog = torch.tensor(self.r_cog, device=dev, dtype=dtype).reshape(1, 3).expand(N, 3)
            # buoyancy force magnitude in body frame == passed nominal buoyancy.
            f_buoy_b = buoyancy_forces_b
            f_grav_b = -up_b * weight
            tau_extra = torch.cross(r_cob, f_buoy_b, dim=-1) + torch.cross(r_cog, f_grav_b, dim=-1)
            dt = dt + tau_extra

        if self.debug:
            print(f"[boundary] df={df}\n dt={dt}\n info={info}")
        return df, dt, info

    # ------------------------------------------------------------------
    # thruster ventilation (per-thruster efficiency multiplier)
    # ------------------------------------------------------------------
    def compute_ventilation_factor(
        self,
        *,
        root_pos_w: torch.Tensor,            # (N, 3)
        root_quat_w: torch.Tensor,           # (N, 4)
        thruster_com_offsets: torch.Tensor,  # (N, K, 3) body-frame thruster positions
    ) -> torch.Tensor:
        """Per-thruster efficiency multiplier ∈ [0, 1], shape (N, K).

        A thruster whose world Z rises toward / above the surface loses authority:
        factor = clip((Z_surface - z_thruster_w) / H, 0, 1).
        Returns all-ones when ventilation is disabled (identity).
        """
        N, K = thruster_com_offsets.shape[0], thruster_com_offsets.shape[1]
        if not self.enable_ventilation:
            return torch.ones((N, K), device=root_pos_w.device, dtype=root_pos_w.dtype)

        q = root_quat_w.unsqueeze(1).expand(N, K, 4).reshape(N * K, 4)
        off_b = thruster_com_offsets.reshape(N * K, 3)
        off_w = quat_apply(q, off_b).reshape(N, K, 3)
        z_thr_w = root_pos_w[..., 2].reshape(N, 1) + off_w[..., 2]  # (N, K)
        h = max(float(self.vehicle_height), self._eps)
        factor = (self.z_surface - z_thr_w) / h
        return torch.clamp(factor, 0.0, 1.0)

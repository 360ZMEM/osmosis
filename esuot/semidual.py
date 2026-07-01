"""Entropic semi-dual objective and dual potential ``w_φ`` (ICML.md Eq.(9)).

This module implements the *entropy-regularized semi-dual* loss that the source
paper derives to remove the unstable inner ``inf`` of the plain semi-dual
formulation (Eq.(7)).  The optimal solution becomes unique and the training
reduces to a single minimisation over the dual potential ``w_φ``.

Reference (``ref/ICML.md``):

    Eq.(9)  L^{E-SemiDual}
        = inf_w   E_{p_T(x)}[ f*(-w(x)) ]
                + ε · E_{p(x_t)}[ log E_{p_T(x)}( exp( (w(x) - 1/(2η)||x - x_t||²) / ε ) ) ]

    Algorithm 1, line 5 (Monte-Carlo, batch B):
        φ ← argmin  (ε/B) Σ_j log( (1/B) Σ_i exp( (w_φ(x_T^i)
                        - 1/(2η)||x_t^j - x_T^i||²) / ε ) )
                  + (1/B) Σ_i f*(-w_φ(x_T^i))

The inner ``log Σ exp`` is evaluated with ``torch.logsumexp`` for numerical
stability.  ``f*`` is the convex conjugate of the chosen f-divergence; KL is the
canonical choice (paper Table 3) and the ablation alternatives (χ², softplus,
identity) are provided for the multi-scheme research requirement.

Unbalanced relaxation: the paper's UOT marginal penalties enter through the
divergence on *each* side (Eq.(18) λ1/λ2).  On the semi-dual side this manifests
as scaling the conjugate term, so we expose ``lambda1`` (target-side conjugate
weight) and ``lambda2`` (source-side transport weight) to let the mass be
created/destroyed -- i.e. tolerate control-energy that is *not* conserved
(thruster saturation / entanglement).
"""

from __future__ import annotations

import math
from typing import Callable

import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# Convex conjugates  f*  of common f-divergences
# ---------------------------------------------------------------------------
#
# For an f-divergence D_f[ρ, p_T] = ∫ p_T f(ρ/p_T), the semi-dual uses the
# convex conjugate  f*(z) = sup_{y≥0} { z·y - f(y) }.  We need f*(-w(x)).
#
#   KL        f(u) = u log u - u + 1      ->  f*(z) = exp(z) - 1
#   chi2      f(u) = (u-1)²               ->  f*(z) = z + z²/4   (z >= -2)
#   softplus            (paper ablation)  ->  f*(z) = softplus(z) = log(1+e^z)
#   identity  f(u) = u (degenerate, OT)   ->  f*(z) = z
#
# KL is the default per Table 3 (best accuracy).  The rest support ablation.


def _f_conj_kl(z: torch.Tensor) -> torch.Tensor:
    # exp(z) - 1 ; clamp the exponent to avoid overflow on extreme potentials.
    return torch.exp(torch.clamp(z, max=30.0)) - 1.0


def _f_conj_chi2(z: torch.Tensor) -> torch.Tensor:
    # f*(z) = z + z²/4, valid region z >= -2; clamp keeps it convex/non-negative slope.
    z = torch.clamp(z, min=-2.0)
    return z + 0.25 * z * z


def _f_conj_softplus(z: torch.Tensor) -> torch.Tensor:
    return torch.nn.functional.softplus(z)


def _f_conj_identity(z: torch.Tensor) -> torch.Tensor:
    return z


_F_CONJUGATES: dict[str, Callable[[torch.Tensor], torch.Tensor]] = {
    "kl": _f_conj_kl,
    "chi2": _f_conj_chi2,
    "softplus": _f_conj_softplus,
    "identity": _f_conj_identity,
}


def f_conjugate(z: torch.Tensor, divergence: str = "kl") -> torch.Tensor:
    """Convex conjugate ``f*(z)`` for the requested f-divergence."""
    try:
        fn = _F_CONJUGATES[divergence]
    except KeyError as exc:
        raise ValueError(
            f"Unknown f-divergence {divergence!r}; choose from {sorted(_F_CONJUGATES)}"
        ) from exc
    return fn(z)


# ---------------------------------------------------------------------------
# Pairwise squared distance
# ---------------------------------------------------------------------------


def pairwise_sq_dist(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Return the (M, N) matrix ``||a_m - b_n||²`` without forming a 3-D tensor.

    ``a`` is (M, D), ``b`` is (N, D).  Uses the expansion
    ``||a-b||² = ||a||² + ||b||² - 2 a·b`` and clamps tiny negatives from
    floating-point cancellation to zero.
    """
    a2 = (a * a).sum(dim=-1, keepdim=True)            # (M, 1)
    b2 = (b * b).sum(dim=-1, keepdim=True).transpose(0, 1)  # (1, N)
    cross = a @ b.transpose(0, 1)                     # (M, N)
    d2 = a2 + b2 - 2.0 * cross
    return d2.clamp_min(0.0)


# ---------------------------------------------------------------------------
# Dual potential network  w_φ : R^D -> R
# ---------------------------------------------------------------------------


class DualPotential(nn.Module):
    """MLP parameterisation of the scalar dual potential ``w_φ(x)``.

    Maps a (B, D) batch of *target-domain* points to a (B,) potential.  Small by
    default since the state-action dimensionality in the STDW buffer is modest.
    """

    def __init__(self, dim: int, hidden: int = 64, depth: int = 2) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        in_dim = dim
        for _ in range(max(depth, 1)):
            layers.append(nn.Linear(in_dim, hidden))
            layers.append(nn.GELU())
            in_dim = hidden
        layers.append(nn.Linear(in_dim, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


# ---------------------------------------------------------------------------
# Entropic semi-dual loss  (Eq.(9) / Algorithm 1 line 5)
# ---------------------------------------------------------------------------


def entropic_semidual_loss(
    w_values_target: torch.Tensor,
    source_pts: torch.Tensor,
    target_pts: torch.Tensor,
    *,
    eps: float,
    eta: float,
    divergence: str = "kl",
    lambda1: float = 1.0,
    lambda2: float = 1.0,
) -> torch.Tensor:
    """Monte-Carlo estimate of ``L^{E-SemiDual}`` minimised over ``w_φ``.

    Args:
        w_values_target: ``w_φ(x_T^i)`` evaluated on the target batch, shape (N,).
        source_pts: source batch ``x_t^j``, shape (M, D).
        target_pts: target batch ``x_T^i``, shape (N, D).
        eps: entropy regularisation strength ε (> 0).
        eta: discretisation step size η (> 0); cost is ``1/(2η)||·||²``.
        divergence: f-divergence key for ``f*`` (default KL).
        lambda1: unbalanced weight on the target-side conjugate term (Eq.18).
        lambda2: unbalanced weight on the source-side transport/LSE term.

    Returns:
        Scalar loss tensor (differentiable w.r.t. ``w_values_target``).

    Shapes are validated so the inner ``logsumexp`` aligns target points (axis 1)
    with the potential vector.
    """
    if eps <= 0.0:
        raise ValueError(f"eps must be > 0 for the entropic objective, got {eps}")
    if eta <= 0.0:
        raise ValueError(f"eta must be > 0, got {eta}")

    M = source_pts.shape[0]
    N = target_pts.shape[0]
    if w_values_target.shape[0] != N:
        raise ValueError(
            f"w_values_target ({w_values_target.shape[0]}) must match target batch ({N})"
        )

    # cost[j, i] = 1/(2η) ||x_t^j - x_T^i||²  ->  (M, N)
    cost = pairwise_sq_dist(source_pts, target_pts) / (2.0 * eta)

    # exponent[j, i] = ( w(x_T^i) - cost[j, i] ) / ε
    exponent = (w_values_target.unsqueeze(0) - cost) / eps  # (M, N)

    # E_{p(x_t)}[ log E_{p_T}( exp(exponent) ) ]
    #   inner   : log (1/N) Σ_i exp(exponent[j,i])  = logsumexp_i - log N
    #   outer   : (1/M) Σ_j inner
    inner = torch.logsumexp(exponent, dim=1) - math.log(N)  # (M,)
    entropic_term = eps * inner.mean()

    # E_{p_T}[ f*(-w(x)) ]
    conj_term = f_conjugate(-w_values_target, divergence=divergence).mean()

    # Unbalanced weighting (Eq.18 λ1/λ2): relax marginal mass conservation.
    return lambda1 * conj_term + lambda2 * entropic_term


__all__ = [
    "DualPotential",
    "entropic_semidual_loss",
    "f_conjugate",
    "pairwise_sq_dist",
]

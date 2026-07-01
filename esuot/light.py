"""Prior-free light backend (ES-B): entropic Sinkhorn + barycentric projection.

This is the lightweight E-SUOT alternative described in PLAN §6 (ES-B) and the
paper's barycentric ablation (``ref/ICML.md`` Algorithm 5 / Table 3 "Barycentric"
row).  It contains **no neural networks** and reads **no physical prior**:

  1. Solve the entropically regularised OT plan ``π`` between the source batch
     ``x_s`` and the target batch ``x_t`` with Sinkhorn iterations on the cost
     ``c_ij = ||x_s^i - x_t^j||²``.
  2. Barycentric projection: ``T(x_s^i) = Σ_j (π_ij / Σ_j π_ij) · x_t^j`` maps each
     source point to the conditional mean of its target mass -- a closed-form,
     unbiased transport that needs no estimation of the target PDF.

The Sinkhorn loop runs in log-space for numerical stability.  Unbalanced mass is
approximated by leaving the marginals soft (a single normalisation of the plan
rows for the barycentric average), which tolerates source/target count or mass
mismatch -- the control-energy non-conservation case.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch

from .semidual import pairwise_sq_dist


def sinkhorn_plan(
    source_pts: torch.Tensor,
    target_pts: torch.Tensor,
    *,
    eps: float = 0.1,
    num_iters: int = 200,
    tol: float = 1.0e-6,
) -> torch.Tensor:
    """Return the entropic OT plan ``π`` (M, N) between two empirical batches.

    Uniform marginals are assumed (``1/M`` and ``1/N``).  Iterations run in
    log-space; early-exits when the dual potentials stop moving.
    """
    if eps <= 0.0:
        raise ValueError(f"eps must be > 0, got {eps}")
    M = source_pts.shape[0]
    N = target_pts.shape[0]
    device = source_pts.device
    dtype = source_pts.dtype

    cost = pairwise_sq_dist(source_pts, target_pts)            # (M, N)
    log_K = -cost / eps                                        # (M, N)
    log_a = torch.full((M,), -torch.log(torch.tensor(float(M))), device=device, dtype=dtype)
    log_b = torch.full((N,), -torch.log(torch.tensor(float(N))), device=device, dtype=dtype)

    f = torch.zeros(M, device=device, dtype=dtype)
    g = torch.zeros(N, device=device, dtype=dtype)
    for _ in range(max(num_iters, 1)):
        f_prev = f
        # f_i = log a_i - logsumexp_j ( log_K_ij + g_j )
        f = log_a - torch.logsumexp(log_K + g.unsqueeze(0), dim=1)
        # g_j = log b_j - logsumexp_i ( log_K_ij + f_i )
        g = log_b - torch.logsumexp(log_K + f.unsqueeze(1), dim=0)
        if torch.max(torch.abs(f - f_prev)) < tol:
            break

    log_plan = f.unsqueeze(1) + log_K + g.unsqueeze(0)         # (M, N)
    return torch.exp(log_plan)


@dataclass
class LightOTConfig:
    """Hyper-parameters for the prior-free light backend (ES-B)."""

    eps: float = 0.1
    num_iters: int = 200
    tol: float = 1.0e-6
    device: str = "cpu"


class LightOTTransport:
    """Barycentric-projection transport from a Sinkhorn plan (no neural nets)."""

    def __init__(self, cfg: Optional[LightOTConfig] = None) -> None:
        self.cfg = cfg or LightOTConfig()
        self.device = torch.device(self.cfg.device)
        self._plan: Optional[torch.Tensor] = None
        self._source: Optional[torch.Tensor] = None
        self._moved: Optional[torch.Tensor] = None

    def fit(self, source_pts: torch.Tensor, target_pts: torch.Tensor) -> "LightOTTransport":
        source_pts = source_pts.to(self.device).float()
        target_pts = target_pts.to(self.device).float()
        if source_pts.ndim != 2 or target_pts.ndim != 2:
            raise ValueError("source_pts and target_pts must be 2-D (B, D)")
        plan = sinkhorn_plan(
            source_pts,
            target_pts,
            eps=self.cfg.eps,
            num_iters=self.cfg.num_iters,
            tol=self.cfg.tol,
        )
        # Barycentric projection: row-normalise the plan, then average targets.
        row_mass = plan.sum(dim=1, keepdim=True).clamp_min(1.0e-12)   # (M, 1)
        weights = plan / row_mass                                     # (M, N)
        self._plan = plan
        self._source = source_pts
        self._moved = weights @ target_pts                            # (M, D)
        return self

    @torch.no_grad()
    def transport(self, pts: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Return the barycentric image of the fitted source batch.

        ``pts`` is accepted for API symmetry with :class:`ESUOTTransport`; when it
        is the same tensor used in :meth:`fit` (the common slow-loop case) the
        cached projection is returned.  For unseen points we fall back to a
        nearest-source barycentric lookup.
        """
        if self._moved is None or self._source is None:
            raise RuntimeError("LightOTTransport.transport called before fit().")
        if pts is None:
            return self._moved
        pts = pts.to(self.device).float()
        if pts.shape == self._source.shape and torch.allclose(pts, self._source):
            return self._moved
        # Unseen query: map each point to the barycentric image of its nearest
        # fitted source point (cheap, prior-free generalisation).
        d2 = pairwise_sq_dist(pts, self._source)        # (Q, M)
        nn_idx = torch.argmin(d2, dim=1)                # (Q,)
        return self._moved[nn_idx]

    @property
    def plan(self) -> Optional[torch.Tensor]:
        return self._plan


__all__ = ["LightOTConfig", "LightOTTransport", "sinkhorn_plan"]

"""Transport map ``T_θ`` and the E-SUOT alternating trainer (Algorithm 1).

Reference (``ref/ICML.md``):

    Eq.(10)  T_θ step:
        argmin_θ  E_{p(x_t)}[ 1/(2η)||x_t - T_θ(x_t)||² - w_φ(T_θ(x_t)) ]

    Algorithm 1:
        for t = 0 .. T-1:
            (lines 3-6) train w_{φ,t} via entropic semi-dual loss (semidual.py)
            (lines 7-10) train T_{θ,t} via Eq.(10)
            x_{t+1} = T_{θ,t}(x_t)            # advance the source batch

This is the prior-aware *full* backend (ES-A): the transported source points are
returned to the STDW slow loop as the target anchor that replaces OPR's
``pseudo_actions``.  It reads **only** state-action samples, never any physical
parameter (COM/COB, J_inv, micro-probe).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import torch
import torch.nn as nn

from .semidual import DualPotential, entropic_semidual_loss, pairwise_sq_dist


# ---------------------------------------------------------------------------
# Transport map  T_θ : R^D -> R^D   (residual parameterisation)
# ---------------------------------------------------------------------------


class TransportMap(nn.Module):
    """Residual MLP ``T_θ(x) = x + g_θ(x)``.

    The residual form keeps ``T_θ`` near identity at initialisation, which is the
    correct prior for a transport that starts at the source distribution and only
    needs to nudge samples toward the target support.
    """

    def __init__(self, dim: int, hidden: int = 64, depth: int = 2) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        in_dim = dim
        for _ in range(max(depth, 1)):
            layers.append(nn.Linear(in_dim, hidden))
            layers.append(nn.GELU())
            in_dim = hidden
        last = nn.Linear(in_dim, dim)
        nn.init.zeros_(last.weight)
        nn.init.zeros_(last.bias)
        layers.append(last)
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.net(x)


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


@dataclass
class ESUOTConfig:
    """Hyper-parameters for the full E-SUOT backend (ES-A)."""

    eps: float = 0.1            # entropy regularisation ε
    eta: float = 1.0            # discretisation step size η (cost 1/(2η)||.||²)
    lambda1: float = 1.0        # unbalanced: target-side conjugate weight
    lambda2: float = 1.0        # unbalanced: source-side transport weight
    divergence: str = "kl"      # f-divergence for f* (kl|chi2|softplus|identity)
    num_steps: int = 1          # T: number of intermediate domains (>=1)
    inner_iters: int = 50       # E: epochs for w_φ and for T_θ per step
    hidden: int = 64
    depth: int = 2
    lr_potential: float = 1.0e-3
    lr_transport: float = 1.0e-3
    grad_clip: float = 1.0
    weight_decay: float = 1.0e-2  # L2 on w_φ; tames off-support extrapolation
    grad_penalty: float = 1.0     # WGAN-GP-style Lipschitz penalty on w_φ
    lipschitz_k: float = 1.0      # target ||∇w|| for the one-sided penalty
    hull_margin: float = 0.25     # clamp T outputs to source∪target hull + margin
    device: str = "cpu"
    seed: Optional[int] = None


# ---------------------------------------------------------------------------
# Full E-SUOT trainer
# ---------------------------------------------------------------------------


class ESUOTTransport:
    """Train ``{T_θ,t}`` and transport source samples toward the target support.

    Typical use inside the STDW slow loop (via :class:`~esuot.adapter`):

        t = ESUOTTransport(dim=D, cfg=cfg)
        t.fit(source_pts, target_pts)
        x_moved = t.transport(source_pts)        # target anchor for the batch
    """

    def __init__(self, dim: int, cfg: ESUOTConfig) -> None:
        self.dim = int(dim)
        self.cfg = cfg
        self.device = torch.device(cfg.device)
        if cfg.seed is not None:
            torch.manual_seed(int(cfg.seed))
        self.maps: List[TransportMap] = []
        self._last_potential_loss: float = float("nan")
        self._last_transport_loss: float = float("nan")
        self._hull_lo: Optional[torch.Tensor] = None
        self._hull_hi: Optional[torch.Tensor] = None

    def _clamp_hull(self, x: torch.Tensor) -> torch.Tensor:
        """Clamp points to the fitted source∪target hull (+margin).

        The dual potential is only trained on the data support; clamping keeps the
        transport map from extrapolating ``w_φ`` into regions where it is
        meaningless and pushing points past the target.
        """
        if self._hull_lo is None or self._hull_hi is None:
            return x
        return torch.max(torch.min(x, self._hull_hi), self._hull_lo)

    # -- training -------------------------------------------------------
    def _lipschitz_penalty(
        self, w: DualPotential, source_pts: torch.Tensor, target_pts: torch.Tensor
    ) -> torch.Tensor:
        """One-sided gradient penalty enforcing ``||∇w|| <= lipschitz_k``.

        The entropic semi-dual objective is unbounded for an arbitrary MLP
        potential: ``w_φ`` can grow a spurious off-support maximum that the
        transport map then ascends, pushing samples past the target.  Sampling
        random interpolations between source and target points and penalising
        ``relu(||∇w|| - k)²`` (WGAN-GP style) restores the Kantorovich potential's
        Lipschitz property and keeps its high region on the target support.
        """
        b = min(source_pts.shape[0], target_pts.shape[0])
        alpha = torch.rand(b, 1, device=source_pts.device, dtype=source_pts.dtype)
        interp = (alpha * source_pts[:b] + (1.0 - alpha) * target_pts[:b]).requires_grad_(True)
        w_interp = w(interp)
        grad = torch.autograd.grad(
            outputs=w_interp.sum(),
            inputs=interp,
            create_graph=True,
            retain_graph=True,
        )[0]
        grad_norm = grad.norm(dim=-1)
        return torch.relu(grad_norm - self.cfg.lipschitz_k).pow(2).mean()

    def _fit_potential(
        self, source_pts: torch.Tensor, target_pts: torch.Tensor
    ) -> DualPotential:
        """Algorithm 1 lines 3-6: minimise the entropic semi-dual loss over w_φ."""
        w = DualPotential(self.dim, hidden=self.cfg.hidden, depth=self.cfg.depth).to(self.device)
        opt = torch.optim.Adam(
            w.parameters(), lr=self.cfg.lr_potential, weight_decay=self.cfg.weight_decay
        )
        last = float("nan")
        for _ in range(max(self.cfg.inner_iters, 1)):
            opt.zero_grad(set_to_none=True)
            w_vals = w(target_pts)
            loss = entropic_semidual_loss(
                w_vals,
                source_pts,
                target_pts,
                eps=self.cfg.eps,
                eta=self.cfg.eta,
                divergence=self.cfg.divergence,
                lambda1=self.cfg.lambda1,
                lambda2=self.cfg.lambda2,
            )
            if self.cfg.grad_penalty > 0.0:
                loss = loss + self.cfg.grad_penalty * self._lipschitz_penalty(
                    w, source_pts, target_pts
                )
            if not torch.isfinite(loss):
                break
            loss.backward()
            if self.cfg.grad_clip > 0.0:
                torch.nn.utils.clip_grad_norm_(w.parameters(), self.cfg.grad_clip)
            opt.step()
            last = float(loss.detach().item())
        self._last_potential_loss = last
        for p in w.parameters():
            p.requires_grad_(False)
        return w

    def _barycentric_proxy(
        self, source_pts: torch.Tensor, target_pts: torch.Tensor, w: DualPotential
    ) -> torch.Tensor:
        """Algorithm 5 barycentric projection of the entropic plan onto the target.

        Directly optimising Eq.(10) over ``T_θ`` is numerically brittle: the cost
        term ``1/(2η)||x_t - T(x_t)||²`` grows quadratically while the dual
        potential ``w_φ`` is only Lipschitz, so for far-apart domains the map
        prefers to stay put or, worse, ascends a spurious off-support maximum of
        ``w_φ`` and overshoots the target.

        Instead we use the frozen ``w_φ`` to reconstruct the entropic conditional
        plan ``π(x_T | x_t) ∝ exp((w_φ(x_T) - 1/(2η)||x_t - x_T||²)/ε)`` and take
        its barycentric projection ``Σ_i π_{ji} x_T^i``.  Because the weights form
        a convex combination of *actual target points*, the proxy always lies in
        the target's convex hull and can never overshoot.
        """
        cost = pairwise_sq_dist(source_pts, target_pts) / (2.0 * self.cfg.eta)  # (M,N)
        w_tgt = w(target_pts).detach()                                          # (N,)
        exponent = (w_tgt.unsqueeze(0) - cost) / self.cfg.eps                   # (M,N)
        plan = torch.softmax(exponent, dim=1)                                   # rows sum to 1
        return plan @ target_pts                                               # (M,D), convex

    def _fit_transport(
        self, source_pts: torch.Tensor, target_pts: torch.Tensor, w: DualPotential
    ) -> TransportMap:
        """Algorithm 1 lines 7-10 (Algorithm 5 variant): regress ``T_θ`` onto the
        barycentric proxy of the frozen entropic plan."""
        proxy = self._barycentric_proxy(source_pts, target_pts, w)
        T = TransportMap(self.dim, hidden=self.cfg.hidden, depth=self.cfg.depth).to(self.device)
        opt = torch.optim.Adam(T.parameters(), lr=self.cfg.lr_transport)
        last = float("nan")
        for _ in range(max(self.cfg.inner_iters, 1)):
            opt.zero_grad(set_to_none=True)
            moved = T(source_pts)
            loss = ((moved - proxy) ** 2).sum(dim=-1).mean()
            if not torch.isfinite(loss):
                break
            loss.backward()
            if self.cfg.grad_clip > 0.0:
                torch.nn.utils.clip_grad_norm_(T.parameters(), self.cfg.grad_clip)
            opt.step()
            last = float(loss.detach().item())
        self._last_transport_loss = last
        for p in T.parameters():
            p.requires_grad_(False)
        return T

    def fit(self, source_pts: torch.Tensor, target_pts: torch.Tensor) -> "ESUOTTransport":
        """Run Algorithm 1 over ``num_steps`` intermediate domains."""
        source_pts = source_pts.to(self.device).float()
        target_pts = target_pts.to(self.device).float()
        if source_pts.ndim != 2 or target_pts.ndim != 2:
            raise ValueError("source_pts and target_pts must be 2-D (B, D)")
        if source_pts.shape[1] != self.dim or target_pts.shape[1] != self.dim:
            raise ValueError(
                f"dim mismatch: cfg.dim={self.dim}, source={source_pts.shape[1]}, "
                f"target={target_pts.shape[1]}"
            )

        self.maps = []
        # Data hull (source ∪ target) + margin, used to clamp the transport map.
        both = torch.cat([source_pts, target_pts], dim=0)
        lo = both.min(dim=0).values
        hi = both.max(dim=0).values
        margin = self.cfg.hull_margin * (hi - lo).clamp_min(1.0e-6)
        self._hull_lo = lo - margin
        self._hull_hi = hi + margin

        x_t = source_pts
        for _ in range(max(self.cfg.num_steps, 1)):
            w = self._fit_potential(x_t, target_pts)
            T = self._fit_transport(x_t, target_pts, w)
            self.maps.append(T)
            with torch.no_grad():
                x_t = self._clamp_hull(T(x_t))     # x_{t+1} = T_{θ,t}(x_t)
        return self

    # -- inference ------------------------------------------------------
    @torch.no_grad()
    def transport(self, pts: torch.Tensor) -> torch.Tensor:
        """Apply the learned composition ``T_{θ,T-1} ∘ … ∘ T_{θ,0}`` to ``pts``."""
        if not self.maps:
            raise RuntimeError("ESUOTTransport.transport called before fit().")
        x = pts.to(self.device).float()
        for T in self.maps:
            x = self._clamp_hull(T(x))
        return x

    @property
    def last_losses(self) -> dict:
        return {
            "potential_loss": self._last_potential_loss,
            "transport_loss": self._last_transport_loss,
        }


__all__ = ["TransportMap", "ESUOTConfig", "ESUOTTransport"]

"""M2: directional hard-constraint guard for the STDW slow loop.

Builds on M1's redefined Lyapunov ``V`` / ``dV``.  M1 emits a per-sample mask
(``stdw_mask = 1`` iff that step was energy-decreasing, i.e. ``dV`` passed the
gate).  The legacy slow loop consumes that mask only for *soft* per-sample loss
weighting (``L = sum(m * mse) / sum(m)``); it never rejects an update and never
constrains the update *direction* -- the three defects recorded in PLAN §1.1
("mask 只做样本加权、不拒绝更新、不约束 drift 方向").

This module converts the soft mask into a *hard*, *directional* batch-level
gate evaluated BEFORE the optimizer step.  Two orthogonal criteria, both keyed
on M1's dV-sign mask:

- ``pass_rate`` (DG-A): the fraction of sampled steps that were
  energy-decreasing must exceed ``min_pass_rate``; otherwise the batch carries
  too little descent evidence and the whole update is rejected.
- ``descent_align`` (DG-B): the *net pull* of the target loss must point along
  the Lyapunov-descent direction.  We split the per-sample target MSE by mask
  and form a signed score ``(w_descent - w_anti) / (w_descent + w_anti)`` in
  ``[-1, 1]``.  A negative score means the gradient is dominated by
  energy-*increasing* (non-descent) samples, so the unmasked update would push
  the policy *against* the Lyapunov flow -- the update is rejected.
- ``both``: require both criteria.

Default mode ``off`` => zero behaviour change.  Pure torch, no Isaac
dependency, so the rejection logic is unit-testable on synthetic batches.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import torch


MODES = ("off", "pass_rate", "descent_align", "both")


@dataclass
class DirGuardConfig:
    """Configuration for the M2 directional hard-constraint guard."""

    mode: str = "off"
    min_pass_rate: float = 0.5
    align_margin: float = 0.0

    def __post_init__(self) -> None:
        if self.mode not in MODES:
            raise ValueError(
                f"stdw_dir_guard mode must be one of {MODES}, got {self.mode!r}"
            )

    @property
    def active(self) -> bool:
        return self.mode != "off"


def _flat_mask(mask: torch.Tensor) -> torch.Tensor:
    return (mask.detach().reshape(-1) > 0.5)


def descent_pass_rate(m_src: torch.Tensor, m_tgt: torch.Tensor) -> float:
    """Fraction of sampled steps whose M1 Lyapunov gate passed (dV descent).

    An empty batch returns ``1.0`` (no evidence against descent), so the guard
    is inert rather than spuriously blocking.
    """
    parts = []
    if m_src is not None and m_src.numel() > 0:
        parts.append(_flat_mask(m_src))
    if m_tgt is not None and m_tgt.numel() > 0:
        parts.append(_flat_mask(m_tgt))
    if not parts:
        return 1.0
    masks = torch.cat(parts)
    return float(masks.float().mean().item())


def descent_alignment_score(mse_tgt: torch.Tensor, m_tgt: torch.Tensor) -> float:
    """Signed net-pull of the target loss along the Lyapunov-descent direction.

    Returns ``(w_descent - w_anti) / (w_descent + w_anti)`` in ``[-1, 1]`` where
    ``w_descent`` / ``w_anti`` are the summed target MSE over the descent
    (mask=1) / anti-descent (mask=0) subsets.  ``> 0`` means the gradient is
    dominated by descent-consistent samples (aligned); ``< 0`` means the update
    is pulled mostly by energy-increasing samples (anti-Lyapunov).

    Degenerate cases (empty batch / all-zero loss) return ``1.0`` so the guard
    stays inert.
    """
    if mse_tgt is None or mse_tgt.numel() == 0:
        return 1.0
    mse = mse_tgt.detach().reshape(-1)
    m = _flat_mask(m_tgt) if (m_tgt is not None and m_tgt.numel() > 0) else torch.ones_like(mse, dtype=torch.bool)
    if m.shape != mse.shape:
        # Defensive: mismatched shapes -> treat as fully descent-consistent.
        return 1.0
    w_descent = float(mse[m].sum().item()) if bool(m.any()) else 0.0
    w_anti = float(mse[~m].sum().item()) if bool((~m).any()) else 0.0
    denom = w_descent + w_anti
    if denom <= 1.0e-12:
        return 1.0
    return (w_descent - w_anti) / denom


def evaluate(
    cfg: DirGuardConfig,
    m_src: torch.Tensor,
    m_tgt: torch.Tensor,
    mse_tgt: torch.Tensor,
) -> Tuple[bool, str, Dict[str, float]]:
    """Decide whether the slow-loop update is directionally admissible.

    Returns ``(accept, reject_reason, metrics)``.  When ``cfg.mode == 'off'`` it
    always accepts (zero behaviour change).  ``reject_reason`` joins the failing
    criteria with ``+`` (e.g. ``"dir_pass_rate+dir_align"``).
    """
    pass_rate = descent_pass_rate(m_src, m_tgt)
    align = descent_alignment_score(mse_tgt, m_tgt)
    metrics = {"dir_guard_pass_rate": pass_rate, "dir_guard_align": align}
    if not cfg.active:
        return True, "", metrics

    reasons = []
    if cfg.mode in ("pass_rate", "both") and pass_rate < cfg.min_pass_rate:
        reasons.append("dir_pass_rate")
    if cfg.mode in ("descent_align", "both") and align < cfg.align_margin:
        reasons.append("dir_align")
    if reasons:
        return False, "+".join(reasons), metrics
    return True, "", metrics

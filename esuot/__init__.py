"""E-SUOT: Entropy-regularized Semi-dual Unbalanced Optimal Transport backend.

A self-research implementation of the ICML-2026 E-SUOT formulation (see
``ref/ICML.md`` Eq.(7)/(9)/(10) and Algorithm 1, strategy in
``ref/ICML补充指引.md``) used as a *plug-and-play* replacement for the OPR
domain-adaptation path inside the STDW slow loop.

Design constraints (mirroring PLAN §6):

- Pure ``torch`` (+ optional ``numpy``); **no Isaac / Omniverse imports** so the
  package is unit-testable offline and decoupled from the simulator.
- **No physical prior.**  The transport is driven *only* by state-action
  samples; it never reads ``_base_com_to_cob_offsets``, never calls
  ``_read_jacobian_inv_diag`` and never runs the micro-probe.  This realises the
  "no explicit estimation, yet unbiased" claim of the paper.
- Two backends, single-select against OPR via ``--domain_adapt_backend``:
    * ``esuot_full``  -> :class:`~esuot.transport.ESUOTTransport`
      (dual potential ``w_φ`` + transport map ``T_θ`` + entropic semi-dual loss),
      faithful to Algorithm 1.
    * ``esuot_light`` -> :class:`~esuot.light.LightOTTransport`
      (entropic Sinkhorn plan + barycentric projection, no neural nets),
      the prior-free lower-complexity backend.

Naming uses generic OT vocabulary (dual_potential / transport_map /
entropic_semidual); no proprietary names from the source paper are reused.
"""

from __future__ import annotations

from .semidual import (
    DualPotential,
    entropic_semidual_loss,
    f_conjugate,
    pairwise_sq_dist,
)
from .transport import ESUOTConfig, ESUOTTransport, TransportMap
from .light import LightOTConfig, LightOTTransport, sinkhorn_plan
from .adapter import DomainAdaptAdapter, AdapterConfig

__all__ = [
    "DualPotential",
    "TransportMap",
    "ESUOTTransport",
    "ESUOTConfig",
    "LightOTTransport",
    "LightOTConfig",
    "DomainAdaptAdapter",
    "AdapterConfig",
    "entropic_semidual_loss",
    "f_conjugate",
    "pairwise_sq_dist",
    "sinkhorn_plan",
]

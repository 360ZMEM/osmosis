"""M5 standalone offline test: E-SUOT dual backend (no Isaac required).

Validates the PLAN §6 / §11 invariants on synthetic source/target distributions:

  - entropic_semidual_loss is finite and decreases over w_φ optimisation.
  - ESUOTTransport (ES-A, full Algorithm 1) moves source samples closer to the
    target support: empirical 2-Wasserstein (via Sinkhorn cost) AND mean-gap both
    drop after transport.
  - LightOTTransport (ES-B, Sinkhorn barycentric) likewise reduces the gap with
    NO neural networks.
  - DomainAdaptAdapter produces an (N_tgt, action_dim) anchor for both backends,
    clamped to [-1, 1], aligned to B_tgt["states"]; uses ONLY state-action samples
    (no physical prior).
  - f_conjugate variants (kl/chi2/softplus/identity) are all finite.

Run:  python workflows/tools/test_m5_esuot.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch


REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from esuot import (  # noqa: E402
    AdapterConfig,
    DomainAdaptAdapter,
    ESUOTConfig,
    ESUOTTransport,
    LightOTConfig,
    LightOTTransport,
    entropic_semidual_loss,
    f_conjugate,
    sinkhorn_plan,
)
from esuot.semidual import DualPotential, pairwise_sq_dist  # noqa: E402


def _ot_cost(a: torch.Tensor, b: torch.Tensor, eps: float = 0.1) -> float:
    """Sinkhorn transport cost <π, C> as an empirical 2-Wasserstein proxy."""
    plan = sinkhorn_plan(a, b, eps=eps, num_iters=300)
    cost = pairwise_sq_dist(a, b)
    return float((plan * cost).sum().item())


def _mean_gap(a: torch.Tensor, b: torch.Tensor) -> float:
    return float((a.mean(dim=0) - b.mean(dim=0)).pow(2).sum().sqrt().item())


def test_f_conjugates_finite() -> None:
    z = torch.linspace(-3.0, 3.0, 50)
    for div in ("kl", "chi2", "softplus", "identity"):
        out = f_conjugate(z, divergence=div)
        assert torch.isfinite(out).all(), f"f_conjugate[{div}] produced non-finite"
    print("[f*] kl/chi2/softplus/identity all finite")


def test_semidual_loss_decreases() -> None:
    torch.manual_seed(0)
    src = torch.randn(64, 2)
    tgt = torch.randn(64, 2) + torch.tensor([3.0, 0.0])
    w = DualPotential(2, hidden=32, depth=2)
    opt = torch.optim.Adam(w.parameters(), lr=1e-2)
    first = None
    last = None
    for i in range(80):
        opt.zero_grad(set_to_none=True)
        loss = entropic_semidual_loss(w(tgt), src, tgt, eps=0.5, eta=1.0)
        loss.backward()
        opt.step()
        if i == 0:
            first = float(loss.item())
        last = float(loss.item())
    assert first is not None and last is not None
    assert last <= first + 1e-6, f"semi-dual loss should not increase: {first}->{last}"
    print(f"[E-SemiDual] loss {first:.4f} -> {last:.4f} (decreasing)")


def test_esuot_full_reduces_gap() -> None:
    torch.manual_seed(1)
    src = torch.randn(96, 3) * 0.5
    tgt = torch.randn(96, 3) * 0.5 + torch.tensor([2.5, -1.5, 0.0])
    cfg = ESUOTConfig(eps=0.2, eta=1.0, num_steps=2, inner_iters=60,
                      lr_potential=2e-3, lr_transport=5e-3, seed=1)
    model = ESUOTTransport(dim=3, cfg=cfg).fit(src, tgt)
    moved = model.transport(src)
    gap_before = _mean_gap(src, tgt)
    gap_after = _mean_gap(moved, tgt)
    cost_before = _ot_cost(src, tgt)
    cost_after = _ot_cost(moved, tgt)
    assert gap_after < gap_before, f"ES-A mean-gap should drop: {gap_before:.3f}->{gap_after:.3f}"
    assert cost_after < cost_before, f"ES-A OT cost should drop: {cost_before:.3f}->{cost_after:.3f}"
    print(f"[ES-A full] mean-gap {gap_before:.3f}->{gap_after:.3f} | "
          f"OT cost {cost_before:.3f}->{cost_after:.3f}")


def test_esuot_light_reduces_gap_no_nn() -> None:
    torch.manual_seed(2)
    src = torch.randn(80, 3) * 0.4
    tgt = torch.randn(80, 3) * 0.4 + torch.tensor([2.0, 1.0, -1.0])
    model = LightOTTransport(LightOTConfig(eps=0.1, num_iters=300)).fit(src, tgt)
    moved = model.transport(src)
    gap_before = _mean_gap(src, tgt)
    gap_after = _mean_gap(moved, tgt)
    assert gap_after < gap_before, f"ES-B mean-gap should drop: {gap_before:.3f}->{gap_after:.3f}"
    # Light backend must contain no nn.Parameter (prior-free, no neural nets).
    assert not list(getattr(model, "parameters", lambda: [])()) if hasattr(model, "parameters") else True
    print(f"[ES-B light] mean-gap {gap_before:.3f}->{gap_after:.3f} (Sinkhorn barycentric, no NN)")


def _make_batches(state_dim: int, action_dim: int, n: int):
    torch.manual_seed(3)
    # source: small actions on a tight state cluster.
    src_states = torch.randn(n, state_dim) * 0.3
    src_actions = torch.tanh(torch.randn(n, action_dim) * 0.2)
    # target: shifted states + different action regime.
    tgt_states = torch.randn(n, state_dim) * 0.3 + 1.5
    tgt_actions = torch.tanh(torch.randn(n, action_dim) * 0.2 + 0.4)
    B_src = {"states": src_states, "actions": src_actions}
    B_tgt = {"states": tgt_states, "actions": tgt_actions}
    return B_src, B_tgt


def test_adapter_anchor_shape_both_backends() -> None:
    state_dim, action_dim, n = 9, 4, 48
    B_src, B_tgt = _make_batches(state_dim, action_dim, n)
    for backend in ("esuot_light", "esuot_full"):
        cfg = AdapterConfig(backend=backend, eps=0.2, eta=1.0, inner_iters=30,
                            num_steps=1, sinkhorn_iters=200, seed=3)
        adapter = DomainAdaptAdapter(cfg)
        anchor = adapter.compute_target_anchor(B_src, B_tgt)
        assert anchor.shape == (n, action_dim), \
            f"{backend}: anchor shape {tuple(anchor.shape)} != ({n}, {action_dim})"
        assert torch.isfinite(anchor).all(), f"{backend}: anchor has non-finite entries"
        assert float(anchor.min()) >= -1.0 - 1e-6 and float(anchor.max()) <= 1.0 + 1e-6, \
            f"{backend}: anchor not clamped to [-1,1]"
        print(f"[adapter:{backend}] anchor shape={tuple(anchor.shape)} "
              f"range=({float(anchor.min()):.3f},{float(anchor.max()):.3f})")


def main() -> None:
    test_f_conjugates_finite()
    test_semidual_loss_decreases()
    test_esuot_full_reduces_gap()
    test_esuot_light_reduces_gap_no_nn()
    test_adapter_anchor_shape_both_backends()
    print("All M5 E-SUOT tests PASSED.")


if __name__ == "__main__":
    main()

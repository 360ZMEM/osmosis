"""M2 standalone synthetic test: directional hard-constraint guard.

Exercises ``stdw_dir_guard`` (the M2 batch-level rejection logic that builds on
M1's per-sample dV-sign mask) WITHOUT Isaac by feeding scripted masks and
target-MSE tensors.  Validates the PLAN §1.1 invariants:

  - off            : always accepts (zero behaviour change), regardless of mask.
  - pass_rate (DG-A): rejects when the descent-sample fraction < min_pass_rate.
  - descent_align (DG-B): rejects when the target-loss net pull is dominated by
    energy-increasing (mask=0) samples (signed score < align_margin).
  - both           : requires both criteria; reject_reason joins failures with "+".
  - degenerate     : empty batch / all-zero loss stays inert (accept).

Run:  python workflows/tools/test_m2_dir_guard.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch


REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import stdw_dir_guard  # noqa: E402


def _cfg(**kw) -> "stdw_dir_guard.DirGuardConfig":
    return stdw_dir_guard.DirGuardConfig(**kw)


def test_off_always_accepts() -> None:
    # Even with a pathological batch (all anti-descent, huge anti loss), off accepts.
    m = torch.zeros(8)
    mse = torch.ones(8) * 100.0
    accept, reason, metrics = stdw_dir_guard.evaluate(_cfg(mode="off"), m, m, mse)
    assert accept is True and reason == "", "off mode must always accept"
    # Metrics still computed for logging.
    assert "dir_guard_pass_rate" in metrics and "dir_guard_align" in metrics
    print(f"[off] accept={accept} pass_rate={metrics['dir_guard_pass_rate']:.3f} "
          f"align={metrics['dir_guard_align']:.3f}")


def test_pass_rate_rejects_low_descent() -> None:
    cfg = _cfg(mode="pass_rate", min_pass_rate=0.5)
    # 1/4 descent -> pass_rate=0.25 < 0.5 -> reject.
    m_low = torch.tensor([1.0, 0.0, 0.0, 0.0])
    mse = torch.ones(4)
    accept, reason, metrics = stdw_dir_guard.evaluate(cfg, m_low, m_low, mse)
    assert accept is False and reason == "dir_pass_rate", reason
    assert abs(metrics["dir_guard_pass_rate"] - 0.25) < 1e-6
    # 3/4 descent -> pass_rate=0.75 >= 0.5 -> accept.
    m_hi = torch.tensor([1.0, 1.0, 1.0, 0.0])
    accept2, reason2, m2 = stdw_dir_guard.evaluate(cfg, m_hi, m_hi, mse)
    assert accept2 is True and reason2 == "", reason2
    print(f"[pass_rate] low(0.25)->reject  hi(0.75)->accept  ok")


def test_descent_align_rejects_anti_lyapunov() -> None:
    cfg = _cfg(mode="descent_align", align_margin=0.0)
    # mask=1 on tiny-loss samples, mask=0 on huge-loss samples => net pull is
    # dominated by anti-descent samples => signed score < 0 => reject.
    m = torch.tensor([1.0, 1.0, 0.0, 0.0])
    mse_anti = torch.tensor([0.1, 0.1, 5.0, 5.0])
    accept, reason, metrics = stdw_dir_guard.evaluate(cfg, m, m, mse_anti)
    assert accept is False and reason == "dir_align", reason
    assert metrics["dir_guard_align"] < 0.0, metrics["dir_guard_align"]
    # Flip: descent samples carry the heavy loss => aligned => accept.
    mse_aligned = torch.tensor([5.0, 5.0, 0.1, 0.1])
    accept2, reason2, m2 = stdw_dir_guard.evaluate(cfg, m, m, mse_aligned)
    assert accept2 is True and reason2 == "", reason2
    assert m2["dir_guard_align"] > 0.0
    print(f"[descent_align] anti(score<0)->reject  aligned(score>0)->accept  ok")


def test_both_requires_both() -> None:
    cfg = _cfg(mode="both", min_pass_rate=0.5, align_margin=0.0)
    # Fails both: low pass_rate AND anti-aligned net pull.
    m = torch.tensor([1.0, 0.0, 0.0, 0.0])
    mse = torch.tensor([0.1, 5.0, 5.0, 5.0])
    accept, reason, _ = stdw_dir_guard.evaluate(cfg, m, m, mse)
    assert accept is False, "both should reject when both fail"
    assert "dir_pass_rate" in reason and "dir_align" in reason, reason
    # Passes both: high pass_rate AND aligned.
    m_ok = torch.tensor([1.0, 1.0, 1.0, 0.0])
    mse_ok = torch.tensor([5.0, 5.0, 5.0, 0.1])
    accept2, reason2, _ = stdw_dir_guard.evaluate(cfg, m_ok, m_ok, mse_ok)
    assert accept2 is True and reason2 == "", reason2
    print(f"[both] reason on double-fail = {reason!r}  ok")


def test_degenerate_batches_stay_inert() -> None:
    cfg = _cfg(mode="both", min_pass_rate=0.5, align_margin=0.0)
    empty = torch.zeros(0)
    accept, reason, metrics = stdw_dir_guard.evaluate(cfg, empty, empty, empty)
    assert accept is True and reason == "", "empty batch must stay inert"
    assert metrics["dir_guard_pass_rate"] == 1.0 and metrics["dir_guard_align"] == 1.0
    # All-zero loss with all-descent mask: align degenerate -> 1.0, pass_rate 1.0.
    m = torch.ones(4)
    accept2, _, m2 = stdw_dir_guard.evaluate(cfg, m, m, torch.zeros(4))
    assert accept2 is True and m2["dir_guard_align"] == 1.0
    print(f"[degenerate] empty & zero-loss stay inert  ok")


def test_invalid_mode_raises() -> None:
    try:
        _cfg(mode="bogus")
    except ValueError:
        print("[validate] invalid mode raises ValueError  ok")
        return
    raise AssertionError("invalid mode should raise ValueError")


def main() -> None:
    torch.manual_seed(0)
    test_off_always_accepts()
    test_pass_rate_rejects_low_descent()
    test_descent_align_rejects_anti_lyapunov()
    test_both_requires_both()
    test_degenerate_batches_stay_inert()
    test_invalid_mode_raises()
    print("All M2 dir-guard tests PASSED.")


if __name__ == "__main__":
    main()

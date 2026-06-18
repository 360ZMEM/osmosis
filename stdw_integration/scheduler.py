from __future__ import annotations


def linear_rho_schedule(step: int, total_steps: int, start: float = 0.0, end: float = 1.0) -> float:
    if total_steps <= 1:
        return float(end)

    progress = min(max(step, 0), total_steps - 1) / float(total_steps - 1)
    return float(start + (end - start) * progress)

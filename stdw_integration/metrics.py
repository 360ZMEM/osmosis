from __future__ import annotations

import math
from typing import Iterable

import numpy as np
import torch


def angle_remap(angle: torch.Tensor) -> torch.Tensor:
    return (angle + torch.pi) % (2 * torch.pi) - torch.pi


def calculate_compound_error(
    des_roll: float,
    des_pitch: float,
    des_yaw: float,
    true_roll: float,
    true_pitch: float,
    true_yaw: float,
    des_depth: float,
    true_depth: float,
) -> float:
    roll_error = float(angle_remap(torch.tensor(true_roll - des_roll)).item())
    pitch_error = float(angle_remap(torch.tensor(true_pitch - des_pitch)).item())
    yaw_error = float(angle_remap(torch.tensor(true_yaw - des_yaw)).item())
    depth_error = true_depth - des_depth
    return float(roll_error**2 + pitch_error**2 + yaw_error**2 + depth_error**2)


def calculate_control_effort(actions: torch.Tensor) -> float:
    if actions.ndim == 1:
        actions = actions.unsqueeze(0)
    return float(torch.linalg.norm(actions, dim=-1).mean().item())


def calculate_domain_bias(volume_scale: float, flow_velocity: Iterable[float]) -> float:
    flow_mag = float(np.linalg.norm(np.asarray(list(flow_velocity), dtype=np.float32)))
    return abs(volume_scale - 1.0) + flow_mag

"""@file eval/wrappers.py
@brief State <-> observation / reward wrappers (Isaac-independent).

State dict contract (must be produced by your hardware bridge):
  position           : np.ndarray (3,)   [m]    world frame
  orientation_quat   : np.ndarray (4,)   [w,x,y,z]
  linear_velocity_b  : np.ndarray (3,)   [m/s]  body frame
  angular_velocity_b : np.ndarray (3,)   [rad/s] body frame
  goal_position      : np.ndarray (3,)   [m]
  goal_yaw           : float             [rad]

Two observation layouts are supported:
  a3_12d      : current parametric EasyUUV A3 policy
                [goal_quat(4), depth_z(1), root_quat(4), angular_velocity_b(3)]
  legacy_10d  : older Isaac-independent demo layout
                [pos_err(3), yaw_err(1), linear_velocity_b(3), angular_velocity_b(3)]
"""

from __future__ import annotations

import math
from typing import Dict

import numpy as np


# Observation layout (must match training-time feature order).
_OBS_KEYS_LEGACY_10D = (
    "pos_err_x", "pos_err_y", "pos_err_z",
    "yaw_err",
    "lin_vel_bx", "lin_vel_by", "lin_vel_bz",
    "ang_vel_bx", "ang_vel_by", "ang_vel_bz",
)
_OBS_KEYS_A3_12D = (
    "goal_qw", "goal_qx", "goal_qy", "goal_qz",
    "depth_z",
    "root_qw", "root_qx", "root_qy", "root_qz",
    "ang_vel_bx", "ang_vel_by", "ang_vel_bz",
)
OBS_DIM_LEGACY_10D = len(_OBS_KEYS_LEGACY_10D)
OBS_DIM_A3_12D = len(_OBS_KEYS_A3_12D)
OBS_DIM = OBS_DIM_A3_12D
ACT_DIM_BASELINE = 4
ACT_DIM_PARAMETRIC = 8


def _quat_to_yaw(quat_wxyz: np.ndarray) -> float:
    """Extract yaw (rotation about world Z) from a w-x-y-z quaternion."""
    w, x, y, z = float(quat_wxyz[0]), float(quat_wxyz[1]), float(quat_wxyz[2]), float(quat_wxyz[3])
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    return math.atan2(siny_cosp, cosy_cosp)


def _wrap_pi(angle: float) -> float:
    return (angle + math.pi) % (2.0 * math.pi) - math.pi


def _yaw_to_quat_wxyz(yaw: float) -> np.ndarray:
    half = 0.5 * float(yaw)
    return np.array([math.cos(half), 0.0, 0.0, math.sin(half)], dtype=np.float32)


def obs_from_state(state: Dict[str, np.ndarray], *, layout: str = "a3_12d") -> np.ndarray:
    """@brief Build an observation from a hardware state dict.
    @param layout "a3_12d" for current A3 parametric policy or "legacy_10d".
    @return np.ndarray shape (12,) or (10,), dtype float32.
    """
    layout = str(layout).lower()
    if layout in {"a3", "a3_12d", "parametric_12d"}:
        goal_quat = state.get("goal_orientation_quat")
        if goal_quat is None:
            goal_quat = _yaw_to_quat_wxyz(float(state["goal_yaw"]))
        goal_quat = np.asarray(goal_quat, dtype=np.float32)
        root_quat = np.asarray(state["orientation_quat"], dtype=np.float32)
        pos = np.asarray(state["position"], dtype=np.float32)
        ang_vel_b = np.asarray(state["angular_velocity_b"], dtype=np.float32)
        return np.concatenate([goal_quat, [float(pos[2])], root_quat, ang_vel_b]).astype(np.float32)

    if layout not in {"legacy", "legacy_10d", "poserr_10d"}:
        raise ValueError(f"unsupported obs layout {layout!r}")

    pos = np.asarray(state["position"], dtype=np.float32)
    goal = np.asarray(state["goal_position"], dtype=np.float32)
    pos_err = goal - pos

    yaw = _quat_to_yaw(np.asarray(state["orientation_quat"], dtype=np.float32))
    yaw_err = _wrap_pi(float(state["goal_yaw"]) - yaw)

    lin_vel_b = np.asarray(state["linear_velocity_b"], dtype=np.float32)
    ang_vel_b = np.asarray(state["angular_velocity_b"], dtype=np.float32)

    return np.concatenate([pos_err, [yaw_err], lin_vel_b, ang_vel_b]).astype(np.float32)


def reward_from_state(state: Dict[str, np.ndarray], action: np.ndarray) -> float:
    """@brief Reference reward shaping (must match training cfg).
    @param state Same dict as obs_from_state.
    @param action Last action np.ndarray (4 or 8,).
    @return Scalar reward.
    @details
      r = -|pos_err|^2 - 0.5*|yaw_err| - 0.1*|action[:4]|^2 - 0.05*|ang_vel_b|^2
      The 4 a_gain channels (action[4:8]) are NOT penalised.
    """
    pos = np.asarray(state["position"], dtype=np.float32)
    goal = np.asarray(state["goal_position"], dtype=np.float32)
    pos_err = goal - pos
    yaw_err = _wrap_pi(
        float(state["goal_yaw"]) - _quat_to_yaw(np.asarray(state["orientation_quat"], dtype=np.float32))
    )
    ang_vel_b = np.asarray(state["angular_velocity_b"], dtype=np.float32)

    ctrl = np.asarray(action, dtype=np.float32)[:4]

    r = (
        -float(np.dot(pos_err, pos_err))
        - 0.5 * abs(yaw_err)
        - 0.1 * float(np.dot(ctrl, ctrl))
        - 0.05 * float(np.dot(ang_vel_b, ang_vel_b))
    )
    return float(r)

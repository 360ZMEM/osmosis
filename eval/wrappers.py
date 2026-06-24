"""@file eval/wrappers.py
@brief 状态 <-> 观测 / reward 转换 wrapper（Isaac 独立）。

状态字典契约（必须由硬件 bridge 产出）：
  position           : np.ndarray (3,)   [m]    世界坐标系
  orientation_quat   : np.ndarray (4,)   [w,x,y,z]
  linear_velocity_b  : np.ndarray (3,)   [m/s]  机体坐标系
  angular_velocity_b : np.ndarray (3,)   [rad/s] 机体坐标系
  goal_position      : np.ndarray (3,)   [m]
  goal_yaw           : float             [rad]

支持两种观测布局：
  a3_12d      : 当前 EasyUUV A3 参数化策略
                [goal_quat(4), depth_z(1), root_quat(4), angular_velocity_b(3)]
  legacy_10d  : 较早的 Isaac 独立 demo 布局
                [pos_err(3), yaw_err(1), linear_velocity_b(3), angular_velocity_b(3)]
"""

from __future__ import annotations

import math
from typing import Dict

import numpy as np


# 观测布局必须与训练时特征顺序一致。
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
    """从 w-x-y-z 四元数中提取 yaw（绕世界 Z 轴旋转）。"""
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
    """@brief 从硬件 state dict 构造策略观测。
    @param layout 当前 A3 参数化策略使用 "a3_12d"，旧 demo 使用 "legacy_10d"。
    @return np.ndarray，shape 为 (12,) 或 (10,)，dtype 为 float32。
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
    """@brief 参考 reward shaping（应与训练配置保持一致）。
    @param state 与 obs_from_state 相同的 state dict。
    @param action 上一步动作，np.ndarray shape 为 (4,) 或 (8,)。
    @return 标量 reward。
    @details
      r = -|pos_err|^2 - 0.5*|yaw_err| - 0.1*|action[:4]|^2 - 0.05*|ang_vel_b|^2
      4 个 a_gain 通道（action[4:8]）不参与惩罚。
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

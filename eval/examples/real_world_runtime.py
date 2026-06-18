"""@file eval/examples/real_world_runtime.py
@brief Isaac-independent real-world runtime skeleton for STDW/A3 deployment.

This example follows the practical structure of ``real_world_ref/example_LLM.py``
but removes Isaac/OpenAI assumptions.  It can run in dry-run mode on a laptop,
or open a serial port when ``serial.enabled: true`` in deploy_config.yaml.
"""

from __future__ import annotations

import argparse
import math
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np

from easyuuv_stdw.eval import Policy, obs_from_state
from easyuuv_stdw.eval.deploy_config import DEFAULT_CONFIG_PATH, DeployConfig, load_deploy_config


def _wrap_pi(angle: float) -> float:
    return (angle + math.pi) % (2.0 * math.pi) - math.pi


def _quat_from_euler_xyz(roll: float, pitch: float, yaw: float) -> np.ndarray:
    """Return w-x-y-z quaternion from roll/pitch/yaw radians."""
    cr, sr = math.cos(roll * 0.5), math.sin(roll * 0.5)
    cp, sp = math.cos(pitch * 0.5), math.sin(pitch * 0.5)
    cy, sy = math.cos(yaw * 0.5), math.sin(yaw * 0.5)
    return np.array([
        cr * cp * cy + sr * sp * sy,
        sr * cp * cy - cr * sp * sy,
        cr * sp * cy + sr * cp * sy,
        cr * cp * sy - sr * sp * cy,
    ], dtype=np.float32)


def _euler_xyz_from_quat(q: np.ndarray) -> tuple[float, float, float]:
    """Return roll/pitch/yaw radians from w-x-y-z quaternion."""
    w, x, y, z = [float(v) for v in q]
    sinr_cosp = 2.0 * (w * x + y * z)
    cosr_cosp = 1.0 - 2.0 * (x * x + y * y)
    roll = math.atan2(sinr_cosp, cosr_cosp)
    sinp = 2.0 * (w * y - z * x)
    pitch = math.copysign(math.pi / 2.0, sinp) if abs(sinp) >= 1.0 else math.asin(sinp)
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    yaw = math.atan2(siny_cosp, cosy_cosp)
    return roll, pitch, yaw


def parse_part(part_str: str) -> dict[str, Optional[float]]:
    """Parse one ESP32 IMU part containing P/R/Y tokens."""
    result: dict[str, Optional[float]] = {"P": None, "R": None, "Y": None}
    for elem in part_str.strip().split():
        if not re.match(r"^[PRY]:", elem):
            continue
        key, raw = elem.split(":", 1)
        try:
            result[key] = float(raw)
        except ValueError:
            result[key] = None
    return result


def parse_all_parts(data_str: str) -> list[dict[str, Optional[float]]]:
    """Parse the existing ESP32 format: ``P:.. R:.. Y:.. | ... | ...``."""
    parts = data_str.strip().split("|")[:3]
    parsed = [parse_part(part) for part in parts]
    while len(parsed) < 3:
        parsed.append({"P": None, "R": None, "Y": None})
    return parsed


@dataclass
class RuntimeState:
    position: np.ndarray
    orientation_quat: np.ndarray
    linear_velocity_b: np.ndarray
    angular_velocity_b: np.ndarray
    goal_position: np.ndarray
    goal_yaw: float


class RealWorldEnv:
    """Minimal serial bridge that produces the eval/wrappers.py state contract."""

    def __init__(self, cfg: DeployConfig):
        self.cfg = cfg
        self.ser = None
        self.last_rpy = np.zeros(3, dtype=np.float32)
        self.last_update_t: Optional[float] = None
        self.state = RuntimeState(
            position=np.array([0.0, 0.0, -1.0], dtype=np.float32),
            orientation_quat=_quat_from_euler_xyz(0.0, 0.0, 0.0),
            linear_velocity_b=np.zeros(3, dtype=np.float32),
            angular_velocity_b=np.zeros(3, dtype=np.float32),
            goal_position=np.array([0.0, 0.0, -1.0], dtype=np.float32),
            goal_yaw=0.0,
        )
        if cfg.serial.enabled:
            try:
                import serial  # type: ignore
            except ImportError as exc:  # pragma: no cover - hardware host only
                raise ImportError("pyserial is required when serial.enabled=true; install with `pip install pyserial`.") from exc
            self.ser = serial.Serial(
                cfg.serial.port,
                cfg.serial.baudrate,
                timeout=cfg.serial.timeout,
                write_timeout=cfg.serial.write_timeout,
            )
            self.ser.reset_input_buffer()

    def reset(self) -> np.ndarray:
        """Activate hardware if serial is enabled and return initial obs."""
        if self.ser is not None:
            # TODO(deploy): Confirm 'a' is the activation command for your ESP32 firmware.
            self.ser.write(b"a")
            time.sleep(1.0)
        self.position_update()
        return self.get_obs()

    def position_update(self) -> bool:
        """Read one IMU line and update quaternion + angular velocity estimate."""
        if self.ser is None:
            return True
        raw_line = self.ser.readline()
        if not raw_line:
            return False
        try:
            line = raw_line.decode("utf-8").strip()
            vals = parse_all_parts(line)[0]
            if vals["R"] is None or vals["P"] is None or vals["Y"] is None:
                return False
            now = time.time()
            rpy = np.array([vals["R"], vals["P"], vals["Y"]], dtype=np.float32) * math.pi / 180.0
            if self.last_update_t is not None:
                dt = max(now - self.last_update_t, 1e-6)
                self.state.angular_velocity_b = (rpy - self.last_rpy) / dt
            self.last_update_t = now
            self.last_rpy = rpy
            self.state.orientation_quat = _quat_from_euler_xyz(float(rpy[0]), float(rpy[1]), float(rpy[2]))
            return True
        except Exception:
            # TODO(deploy): Replace with structured logging on the vehicle.
            return False

    def get_obs(self) -> np.ndarray:
        """Return the current A3 12-D policy observation."""
        self.position_update()
        return obs_from_state(self.state.__dict__, layout=self.cfg.policy.obs_layout)

    def step(self, action: np.ndarray) -> np.ndarray:
        """Apply one policy action through the serial command protocol."""
        ctrl = np.clip(np.asarray(action, dtype=np.float32)[:3], -1.0, 1.0)
        action_limit = np.asarray(self.cfg.control.action_limit_rpy, dtype=np.float32)
        roll, pitch, yaw = _euler_xyz_from_quat(self.state.orientation_quat)
        desired = np.array([roll, pitch, yaw], dtype=np.float32) + ctrl * action_limit
        desired = np.array([_wrap_pi(float(v)) for v in desired], dtype=np.float32)
        if self.ser is not None:
            # TODO(deploy): Confirm command units/sign conventions with the ESP32 firmware.
            deg = desired * 180.0 / math.pi
            self.ser.write(f"r{deg[0]:.2f}p{deg[1]:.2f}y{deg[2]:.2f}".encode("utf-8"))
        else:
            self.state.orientation_quat = _quat_from_euler_xyz(float(desired[0]), float(desired[1]), float(desired[2]))
        return self.get_obs()

    def halt(self) -> None:
        """Send emergency stop to the low-level board when available."""
        if self.ser is not None:
            # TODO(deploy): Confirm 'e' is the emergency stop command.
            self.ser.write(b"e")


class ZeroPolicy:
    """Dry-run fallback when no deploy policy has been copied yet."""

    backend = "zero-dry-run"

    def act(self, obs: np.ndarray) -> np.ndarray:
        del obs
        return np.zeros(8, dtype=np.float32)


def main() -> None:
    parser = argparse.ArgumentParser(description="Isaac-independent real-world deployment skeleton.")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG_PATH))
    parser.add_argument("--policy", default=None, help="Override policy.model_path from YAML")
    parser.add_argument("--steps", type=int, default=10, help="Short dry-run horizon for smoke testing")
    args = parser.parse_args()

    cfg = load_deploy_config(args.config)
    if args.policy:
        cfg.policy.model_path = str(Path(args.policy).expanduser())

    env = RealWorldEnv(cfg)
    policy_path = Path(cfg.policy.model_path).expanduser()
    if policy_path.is_file():
        policy = Policy(policy_path, device=cfg.policy.device)
    elif not cfg.serial.enabled:
        # Dry-run should work before a deploy JIT is copied to the vehicle host.
        # TODO(deploy): Replace the YAML model_path with a real *_deploy.jit before wet testing.
        print(f"[WARN] policy not found: {policy_path}; using zero-action dry-run policy.")
        policy = ZeroPolicy()
    else:
        raise FileNotFoundError(f"policy file not found: {policy_path}")
    obs = env.reset()
    print(f"runtime config={args.config}, serial_enabled={cfg.serial.enabled}, backend={policy.backend}, obs_shape={obs.shape}")

    try:
        for step in range(args.steps):
            action = policy.act(obs)
            obs = env.step(action)
            print(f"step={step:04d} action={action[:4].tolist()} obs_tail_ang_vel={obs[-3:].tolist()}")
            time.sleep(float(cfg.control.control_dt))
    except KeyboardInterrupt:
        env.halt()
        raise
    finally:
        env.halt()


if __name__ == "__main__":
    main()

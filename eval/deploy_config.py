"""@file eval/deploy_config.py
@brief Isaac 独立部署配置加载器。

本模块把实物部署参数集中放在配置文件中，避免散落到命令行示例里。
它刻意不导入 Isaac / omni / rsl_rl 模块，因此只依赖 Python、numpy、torch
以及可选的 pyyaml 就能在板载主机运行。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping


DEFAULT_CONFIG_PATH = Path(__file__).with_name("deploy_config.yaml")


@dataclass
class SerialConfig:
    port: str = "/dev/ttyUSB0"
    baudrate: int = 115200
    timeout: float = 0.05
    write_timeout: float = 0.05
    enabled: bool = False


@dataclass
class ControlConfig:
    control_dt: float = 1.0 / 160.0
    steps_per_action: int = 240
    control_mode: list[bool] = field(default_factory=lambda: [False, False, True])
    action_limit_rpy: list[float] = field(default_factory=lambda: [0.0, 0.0, -0.34])


@dataclass
class ControllerConfig:
    roll_zeta: list[float] = field(default_factory=lambda: [0.25, 0.0])
    pitch_zeta: list[float] = field(default_factory=lambda: [0.40, 0.2])
    yaw_zeta: list[float] = field(default_factory=lambda: [0.10, 0.0])
    s_ratio: float = 1.0


@dataclass
class PolicyConfig:
    model_path: str = "./logs/deploy/stdw_step_001499_deploy.jit"
    obs_layout: str = "a3_12d"
    device: str = "cpu"


@dataclass
class MicroProbeConfig:
    enable: bool = False
    start_step: int = 200
    window_steps: int = 60
    settle_steps: int = 20
    axes: list[int] = field(default_factory=lambda: [0, 1])
    magnitude: float = 0.02
    score_mode: str = "paired_axis"


@dataclass
class StdwDeployConfig:
    enable: bool = False
    min_real_samples: int = 64
    slow_loop_interval: int = 120
    g_C_lr: float = 5.0e-5
    micro_probe: MicroProbeConfig = field(default_factory=MicroProbeConfig)


@dataclass
class SineGoalConfig:
    enable: bool = False
    amplitude: float = 1.0472
    period: float = 10.0


@dataclass
class GoalConfig:
    sequence: list[list[float]] = field(
        default_factory=lambda: [
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 1.2566],
            [0.0, 0.0, 0.0],
            [0.0, 0.0, -1.2566],
            [0.0, 0.0, 0.0],
        ]
    )
    sine: SineGoalConfig = field(default_factory=SineGoalConfig)


@dataclass
class DeployConfig:
    serial: SerialConfig = field(default_factory=SerialConfig)
    control: ControlConfig = field(default_factory=ControlConfig)
    controller: ControllerConfig = field(default_factory=ControllerConfig)
    policy: PolicyConfig = field(default_factory=PolicyConfig)
    stdw: StdwDeployConfig = field(default_factory=StdwDeployConfig)
    goal: GoalConfig = field(default_factory=GoalConfig)


def _merge_dataclass(obj: Any, data: Mapping[str, Any]) -> Any:
    """将嵌套 mapping 原地合并到 dataclass 对象中。"""
    for key, value in data.items():
        if not hasattr(obj, key):
            raise KeyError(f"unknown deploy config key: {key}")
        current = getattr(obj, key)
        if hasattr(current, "__dataclass_fields__") and isinstance(value, Mapping):
            _merge_dataclass(current, value)
        else:
            setattr(obj, key, value)
    return obj


def load_deploy_config(path: str | Path | None = None) -> DeployConfig:
    """加载部署 YAML，并覆盖 dataclass 默认值。

    YAML 中缺失字段会保留默认值；未知字段会立即报错，避免部署参数拼写错误
    静默改变真实硬件行为。
    """
    cfg = DeployConfig()
    cfg_path = Path(path).expanduser() if path else DEFAULT_CONFIG_PATH
    if not cfg_path.is_file():
        return cfg

    try:
        import yaml
    except ImportError as exc:  # pragma: no cover - 取决于板载/主机镜像
        raise ImportError("pyyaml is required to read deploy_config.yaml; install with `pip install pyyaml`.") from exc

    with cfg_path.open("r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}
    if not isinstance(raw, Mapping):
        raise ValueError(f"deploy config must be a YAML mapping, got {type(raw).__name__}")
    return _merge_dataclass(cfg, raw)


__all__ = [
    "DEFAULT_CONFIG_PATH",
    "DeployConfig",
    "load_deploy_config",
]

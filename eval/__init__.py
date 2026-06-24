"""@file eval/__init__.py
@brief EasyUUV STDW 策略的 Isaac Lab 独立评估工具。
@details
  本子包包含在 Isaac Sim 之外运行已训练 EasyUUV 策略所需的最小组件：
    - wrappers.py       state<->obs / reward 转换（纯 numpy/torch）
    - policy_loader.py  pt / jit / onnx 加载器（按后缀自动分派）
    - train_loop.py     参考 PPO 循环（不依赖 rsl_rl 和 Isaac）
    - deploy_eval.py    CLI：加载 ckpt + replay -> 指标
    - examples/         最小示例脚本

  本模块刻意不导入 omni.isaac.*，因此可在板载计算机或 CI 容器中运行。
"""

from .wrappers import obs_from_state, reward_from_state  # noqa: F401
from .policy_loader import Policy  # noqa: F401
from .deploy_config import DeployConfig, load_deploy_config  # noqa: F401

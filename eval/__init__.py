"""@file eval/__init__.py
@brief Isaac-Lab-independent evaluation utilities for EasyUUV STDW policies.
@details
  This subpackage holds everything needed to run a trained EasyUUV policy
  outside Isaac Sim:
    - wrappers.py       state<->obs / reward (pure numpy/torch)
    - policy_loader.py  pt / jit / onnx loader (auto-dispatch by suffix)
    - train_loop.py     reference PPO loop (no rsl_rl, no Isaac)
    - deploy_eval.py    CLI: load ckpt + replay -> metrics
    - examples/         minimal demo scripts

  This module deliberately does NOT import omni.isaac.* so it can run on a
  vehicle's onboard computer or in a CI container.
"""

from .wrappers import obs_from_state, reward_from_state  # noqa: F401
from .policy_loader import Policy  # noqa: F401
from .deploy_config import DeployConfig, load_deploy_config  # noqa: F401

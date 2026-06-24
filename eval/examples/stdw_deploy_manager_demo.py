"""Scheme-B STDW checkpoint 的部署管理 demo。

本示例不依赖 Isaac。它读取 STDW 运行的 ``summary.json`` 或单个 checkpoint
metadata JSON，选出导出的 ``*_deploy.jit`` 策略，构造当前 A3 12D 观测，
并将 8D 策略输出映射为参考推进器命令。
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict

import numpy as np

from easyuuv_stdw.eval import Policy, obs_from_state
from easyuuv_stdw.eval.deploy_config import DEFAULT_CONFIG_PATH, load_deploy_config
from easyuuv_stdw.eval.examples.thruster_io_demo import to_thruster_cmds


def _latest_deploy_jit_from_summary(summary: Dict[str, Any]) -> str:
    saved = summary.get("saved_stdw_ckpts") or []
    for item in reversed(saved):
        deploy = item.get("deploy_jit_path") if isinstance(item, dict) else None
        if deploy:
            return str(deploy)
    raise ValueError("summary does not contain any saved deploy_jit_path entries")


def _resolve_policy_path(policy: str | None, metadata: str | None, summary: str | None) -> str:
    if policy:
        return str(Path(policy).expanduser())
    if metadata:
        data = json.loads(Path(metadata).read_text(encoding="utf-8"))
        deploy = data.get("deploy_jit_path")
        if deploy:
            return str(deploy)
        raise ValueError("metadata JSON does not contain deploy_jit_path")
    if summary:
        data = json.loads(Path(summary).read_text(encoding="utf-8"))
        return _latest_deploy_jit_from_summary(data)
    raise ValueError("provide one of --policy, --metadata-json, or --summary-json")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run one managed STDW deployment inference step.")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG_PATH), help="Deployment YAML config path")
    parser.add_argument("--policy", default=None, help="Direct path to exported *_deploy.jit/.onnx/.pt module")
    parser.add_argument("--metadata-json", default=None, help="Per-checkpoint stdw_step_*.json metadata")
    parser.add_argument("--summary-json", default=None, help="STDW run summary.json with saved_stdw_ckpts")
    args = parser.parse_args()

    cfg = load_deploy_config(args.config)
    # TODO(deploy): 实物上应让 deploy_config.yaml 的 model_path 指向
    # 从自适应运行中拷贝过来的最新 *_deploy.jit。
    policy_arg = args.policy or cfg.policy.model_path
    policy_path = _resolve_policy_path(policy_arg, args.metadata_json, args.summary_json)
    policy = Policy(policy_path, device=cfg.policy.device)

    # TODO(deploy): 将该 dict 替换为真实硬件 bridge 输出。
    # 字段必须匹配 eval/wrappers.py；这里不涉及 Isaac。
    state = {
        "position": np.array([0.0, 0.0, -1.0], dtype=np.float32),
        "orientation_quat": np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32),
        "linear_velocity_b": np.zeros(3, dtype=np.float32),
        "angular_velocity_b": np.zeros(3, dtype=np.float32),
        "goal_position": np.array([0.0, 0.0, -1.0], dtype=np.float32),
        "goal_yaw": 0.0,
    }

    obs = obs_from_state(state, layout=cfg.policy.obs_layout)
    action = policy.act(obs)
    thruster_cmds = to_thruster_cmds(action)

    print(f"policy={policy_path}")
    print(f"config={args.config}, obs_layout={cfg.policy.obs_layout}, stdw_enable={cfg.stdw.enable}")
    print(f"backend={policy.backend}, obs_shape={obs.shape}, action_shape={action.shape}")
    print(f"control_intent={action[:4].tolist()}")
    if action.shape[0] > 4:
        print(f"a_gain_channels={action[4:].tolist()}")
    print(f"thruster_cmds={thruster_cmds.tolist()}")


if __name__ == "__main__":
    main()

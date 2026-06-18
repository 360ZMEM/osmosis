# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Single-env eval for the 8-dim meta-control policy + ζ_runtime timeline.

核心目的：在 COB drift（重心-浮心偏移）注入下，记录 ``env._zeta_runtime`` /
``env._zeta_nominal`` 的逐 step 时间轴，验证策略确实在线"调大/调小"S-Surface
增益以抵抗倾覆力矩——这是实施计划 §3.6 / §5 中明文写明的可观测性要求。

输出：
    <save_path>/meta_eval_timeline.csv         逐 step 一行
    <save_path>/meta_eval_summary.json         统计指标（drift 区间 ζ 比、PE/死区占比、姿态 MSE）
    <save_path>/meta_eval_zeta_timeline.png    matplotlib 4 子图（roll / pitch / yaw / depth）
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

import matplotlib

matplotlib.use("Agg")

from omni.isaac.lab.app import AppLauncher


# ---------------------------------------------------------------------------
# Bootstrap：与 play_stdw_adapt.py / train_meta.py 同一套路。
# ---------------------------------------------------------------------------

WORKFLOW_DIR = Path(__file__).resolve().parent
REPO_ROOT = WORKFLOW_DIR.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
CUSTOM_WORKFLOWS_DIR = REPO_ROOT / "custom_workflows"
if str(CUSTOM_WORKFLOWS_DIR) not in sys.path:
    sys.path.insert(0, str(CUSTOM_WORKFLOWS_DIR))


def _bootstrap_local_lab_tasks_package() -> None:
    package_name = "omni.isaac.lab_tasks"
    if package_name in sys.modules and getattr(
        sys.modules[package_name], "__file__", None
    ) == str(REPO_ROOT / "__init__.py"):
        return
    try:
        import omni.isaac.lab_tasks  # noqa: F401
        import omni.isaac.lab_tasks.utils  # noqa: F401
        import omni.isaac.lab_tasks.utils.wrappers  # noqa: F401
        import omni.isaac.lab_tasks.utils.wrappers.rsl_rl  # noqa: F401
    except Exception:
        pass
    package_init = REPO_ROOT / "__init__.py"
    spec = importlib.util.spec_from_file_location(
        package_name, package_init, submodule_search_locations=[str(REPO_ROOT)]
    )
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to bootstrap {package_name} from {package_init}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[package_name] = module
    spec.loader.exec_module(module)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _bool_arg(value):
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "on", "y", "t"}


parser = argparse.ArgumentParser(description="8-dim meta-control eval (single env + COB drift).")
parser.add_argument("--cpu", action="store_true", default=False)
parser.add_argument("--disable_fabric", action="store_true", default=False)
parser.add_argument("--task", type=str, default="EasyUUV-Direct-Parametric-v1")
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument("--seed", type=int, default=None)
parser.add_argument("--steps", type=int, default=2000, help="评估步数")

# COB drift injection（实施计划 §3.6 必备）
parser.add_argument("--cob_drift_axis", type=str, default="y", choices=["x", "y", "z"],
                    help="重心-浮心偏移注入方向（仅 1 维）")
parser.add_argument("--cob_drift_magnitude", type=float, default=0.0,
                    help="终态偏移量 (m)。0 表示不注入")
parser.add_argument("--cob_drift_start_step", type=int, default=200)
parser.add_argument("--cob_drift_end_step", type=int, default=800)
parser.add_argument("--embodiment", type=str, default=None,
                    help="切换机型 (base/long_body/heavy_duty/heavy_moderate/asymmetric)；走 env.apply_embodiment_config()")

# 元控制 CLI（评估期临时改 ζ 限幅 / 死区等）
parser.add_argument("--tune_gains", type=_bool_arg, default=None)
parser.add_argument("--gain_beta", type=float, default=None)
parser.add_argument("--enable_pe", type=_bool_arg, default=None)
parser.add_argument("--pe_freq", type=float, default=None)
parser.add_argument("--pe_amp", type=float, default=None)
parser.add_argument("--pe_decay_gamma", type=float, default=None)
parser.add_argument("--enable_deadzone", type=_bool_arg, default=None)
parser.add_argument("--deadzone_threshold", type=float, default=None)
parser.add_argument("--enable_param_lpf", type=_bool_arg, default=None)
parser.add_argument("--param_lpf_cutoff", type=float, default=None)
parser.add_argument("--identity_init", type=_bool_arg, default=None)

# 输出与 checkpoint 选择
parser.add_argument("--csv_path", type=str, default=None,
                    help="若不指定，则写到 results/<exp>/<run>/<ckpt>_play/meta_eval_timeline.csv")
parser.add_argument("--save_dir", type=str, default=None,
                    help="可选：直接指定整个 save_path 目录")
parser.add_argument("--policy_path", type=str, default=None,
                    help="直接指定一个 .pt checkpoint 路径；不指定则走标准 logs 路径解析")

# 与 cli_args 兼容（experiment_name / load_run / load_checkpoint 等都来自这里）
import cli_args  # noqa: E402  isort: skip
cli_args.add_rsl_rl_args(parser)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()


app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

_bootstrap_local_lab_tasks_package()


# ---------------------------------------------------------------------------
# Lab-side imports.
# ---------------------------------------------------------------------------

import csv  # noqa: E402
import gymnasium as gym  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

from rsl_rl.runners import OnPolicyRunner  # noqa: E402

import omni.isaac.lab_tasks  # noqa: F401, E402
from omni.isaac.lab_tasks.utils import get_checkpoint_path, parse_env_cfg  # noqa: E402
from omni.isaac.lab_tasks.utils.wrappers.rsl_rl import (  # noqa: E402
    RslRlOnPolicyRunnerCfg,
    RslRlVecEnvWrapper,
)

from custom_workflows.workflow_config import apply_config_overrides, load_workflow_config  # noqa: E402
from custom_workflows.workflow_paths import WorkflowPaths, ensure_directory  # noqa: E402


META_CFG_KEYS = (
    "tune_gains",
    "gain_beta",
    "enable_pe",
    "pe_freq",
    "pe_amp",
    "pe_decay_gamma",
    "enable_deadzone",
    "deadzone_threshold",
    "enable_param_lpf",
    "param_lpf_cutoff",
    "identity_init",
)
AXIS_NAMES = ("roll", "pitch", "yaw", "depth")
AXIS_TO_IDX = {"x": 0, "y": 1, "z": 2}


def _apply_meta_cfg_overrides(env_cfg, args) -> None:
    for key in META_CFG_KEYS:
        value = getattr(args, key, None)
        if value is None:
            continue
        if not hasattr(env_cfg, key):
            print(f"[WARN] env_cfg has no attribute '{key}'; skipping --{key}={value}")
            continue
        setattr(env_cfg, key, value)
        print(f"[INFO] Override env_cfg.{key} = {value}")


def _drift_value(step: int, args) -> float:
    """线性 ramp：[start, end] 区间从 0 渐进到 magnitude。"""
    if args.cob_drift_magnitude == 0.0:
        return 0.0
    if step < args.cob_drift_start_step:
        return 0.0
    if step >= args.cob_drift_end_step:
        return args.cob_drift_magnitude
    span = max(1, args.cob_drift_end_step - args.cob_drift_start_step)
    frac = (step - args.cob_drift_start_step) / span
    return float(args.cob_drift_magnitude * frac)


def _to_np(t: torch.Tensor) -> np.ndarray:
    return t.detach().cpu().numpy()


def main():
    workflow_cfg = load_workflow_config(args_cli.workflow_config)
    path_cfg = workflow_cfg.get("paths", {}) or {}

    paths = WorkflowPaths.from_overrides(
        logs_root=args_cli.logs_root or path_cfg.get("logs_root"),
        results_root=args_cli.results_root or path_cfg.get("results_root"),
        artifacts_root=args_cli.artifacts_root or path_cfg.get("artifacts_root"),
    )

    # parse configuration（强制 num_envs=1 单环境评估）
    env_cfg = parse_env_cfg(
        args_cli.task,
        use_gpu=not args_cli.cpu,
        num_envs=max(1, args_cli.num_envs),
        use_fabric=not args_cli.disable_fabric,
    )
    if workflow_cfg.get("env"):
        apply_config_overrides(env_cfg, workflow_cfg["env"])
    _apply_meta_cfg_overrides(env_cfg, args_cli)

    agent_cfg: RslRlOnPolicyRunnerCfg = cli_args.parse_rsl_rl_cfg(
        args_cli.task, args_cli, workflow_cfg.get("agent")
    )

    env = gym.make(args_cli.task, cfg=env_cfg)
    env = RslRlVecEnvWrapper(env)

    # checkpoint 解析
    if args_cli.policy_path:
        resume_path = args_cli.policy_path
    else:
        log_root_path = paths.training_log_root(agent_cfg.experiment_name)
        print(f"[INFO] Loading experiment from directory: {log_root_path}")
        resume_path = get_checkpoint_path(str(log_root_path), agent_cfg.load_run, agent_cfg.load_checkpoint)
    print(f"[INFO]: Loading model checkpoint from: {resume_path}")

    ppo_runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    ppo_runner.load(resume_path)
    policy = ppo_runner.get_inference_policy(device=env.unwrapped.device)

    # 输出目录
    if args_cli.save_dir:
        save_path = Path(args_cli.save_dir)
    else:
        save_path = paths.result_run_dir(agent_cfg.experiment_name, agent_cfg.load_run, Path(resume_path).name)
    ensure_directory(save_path)
    csv_path = Path(args_cli.csv_path) if args_cli.csv_path else (save_path / "meta_eval_timeline.csv")
    ensure_directory(csv_path.parent)
    summary_path = save_path / "meta_eval_summary.json"
    png_path = save_path / "meta_eval_zeta_timeline.png"
    print(f"[INFO]: Writing CSV -> {csv_path}")
    print(f"[INFO]: Writing summary -> {summary_path}")
    print(f"[INFO]: Writing plot -> {png_path}")

    # CSV 表头
    fieldnames = [
        "step", "sim_time",
        "cob_offset_x", "cob_offset_y", "cob_offset_z",
        "ang_vel_x", "ang_vel_y", "ang_vel_z", "ang_vel_norm",
    ]
    fieldnames += [f"a_ctrl_{i}" for i in range(4)]
    fieldnames += [f"a_gain_{i}" for i in range(4)]
    fieldnames += [f"a_gain_lpf_{i}" for i in range(4)]
    fieldnames += [f"zeta1_{name}" for name in AXIS_NAMES]
    fieldnames += [f"zeta1_{name}_nominal" for name in AXIS_NAMES]
    fieldnames += ["pe_active", "deadzone_active"]

    csv_file = open(csv_path, "w", newline="")
    csv_writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
    csv_writer.writeheader()

    # 时间轴缓存（用于绘图与统计）
    timeline = {name: {"runtime": [], "nominal": []} for name in AXIS_NAMES}
    pe_flags = []
    deadzone_flags = []
    cob_offset_axis_history = []
    ang_vel_norm_history = []

    # reset
    obs, _ = env.get_observations()
    underlying = env.unwrapped
    if args_cli.embodiment is not None and args_cli.embodiment != "base":
        if hasattr(underlying, "apply_embodiment_config"):
            underlying.apply_embodiment_config(args_cli.embodiment)
            obs, _ = env.reset()
        else:
            print(f"[WARN] env has no apply_embodiment_config; ignoring --embodiment={args_cli.embodiment}")
    drift_axis_idx = AXIS_TO_IDX[args_cli.cob_drift_axis]

    # 缓存 reset 后的 cob 基线（避免 DR 改写导致漂移叠加错误）
    if hasattr(underlying, "_base_com_to_cob_offsets"):
        cob_baseline = underlying._base_com_to_cob_offsets.clone()
    else:
        cob_baseline = underlying.com_to_cob_offsets.clone()

    sim_dt = float(underlying.sim.cfg.dt) * int(underlying.cfg.decimation)
    n_total = int(args_cli.steps)
    drift_in_window = []  # (step, ratio_vector) 用于 summary

    for step in range(n_total):
        # 注入 COB drift（直接写 env.com_to_cob_offsets，buoyancy 在每个 _compute_dynamics 调用）
        delta = _drift_value(step, args_cli)
        underlying.com_to_cob_offsets[:] = cob_baseline
        underlying.com_to_cob_offsets[:, drift_axis_idx] += delta

        with torch.no_grad():
            actions = policy(obs)

        # step
        obs, rew, dones, _ = env.step(actions)

        # 提取 8 维拆分（前 4 维 a_ctrl，后 4 维 a_gain）
        actions_np = _to_np(actions)[0]
        if actions_np.shape[0] >= 8:
            a_ctrl = actions_np[:4]
            a_gain = actions_np[4:8]
        else:
            a_ctrl = actions_np[:4]
            a_gain = np.zeros(4, dtype=actions_np.dtype)

        # 取 ζ 时间轴（仅在 tune_gains=True 时存在）
        if getattr(underlying, "_tune_gains_enabled", False):
            zeta_runtime = _to_np(underlying._zeta_runtime[0])  # (4,)
            zeta_nominal = _to_np(underlying._zeta_nominal[0])  # (4,)
            tuner = getattr(underlying, "_gain_tuner", None)
            if tuner is not None and hasattr(tuner, "_a_gain_lpf"):
                a_gain_lpf = _to_np(tuner._a_gain_lpf[0])
            else:
                a_gain_lpf = a_gain.copy()
            pe_active = bool(_to_np(underlying._last_pe_active[0]))
            deadzone_active = bool(_to_np(underlying._last_deadzone_active[0]))
        else:
            zeta_runtime = _to_np(underlying.PID_args[0, :, 0])
            zeta_nominal = zeta_runtime.copy()
            a_gain_lpf = a_gain.copy()
            pe_active = False
            deadzone_active = False

        ang_vel = _to_np(underlying._robot.data.root_ang_vel_b[0])
        ang_vel_norm = float(np.linalg.norm(ang_vel))
        cob = _to_np(underlying.com_to_cob_offsets[0])

        row = {
            "step": step,
            "sim_time": step * sim_dt,
            "cob_offset_x": float(cob[0]),
            "cob_offset_y": float(cob[1]),
            "cob_offset_z": float(cob[2]),
            "ang_vel_x": float(ang_vel[0]),
            "ang_vel_y": float(ang_vel[1]),
            "ang_vel_z": float(ang_vel[2]),
            "ang_vel_norm": ang_vel_norm,
            "pe_active": int(pe_active),
            "deadzone_active": int(deadzone_active),
        }
        for i in range(4):
            row[f"a_ctrl_{i}"] = float(a_ctrl[i])
            row[f"a_gain_{i}"] = float(a_gain[i])
            row[f"a_gain_lpf_{i}"] = float(a_gain_lpf[i])
        for i, name in enumerate(AXIS_NAMES):
            row[f"zeta1_{name}"] = float(zeta_runtime[i])
            row[f"zeta1_{name}_nominal"] = float(zeta_nominal[i])
        csv_writer.writerow(row)

        for i, name in enumerate(AXIS_NAMES):
            timeline[name]["runtime"].append(float(zeta_runtime[i]))
            timeline[name]["nominal"].append(float(zeta_nominal[i]))
        pe_flags.append(int(pe_active))
        deadzone_flags.append(int(deadzone_active))
        cob_offset_axis_history.append(float(cob[drift_axis_idx]))
        ang_vel_norm_history.append(ang_vel_norm)

        # 漂移区间内的 ζ 比率（用于"必须可观察到"判据）
        if args_cli.cob_drift_start_step <= step < args_cli.cob_drift_end_step and args_cli.cob_drift_magnitude != 0.0:
            ratio = np.where(np.abs(zeta_nominal) > 1e-9, zeta_runtime / zeta_nominal, 1.0)
            drift_in_window.append(ratio)

    csv_file.close()

    # ----- summary.json -----
    summary = {
        "task": args_cli.task,
        "checkpoint": str(resume_path),
        "steps": n_total,
        "sim_dt": sim_dt,
        "cob_drift_axis": args_cli.cob_drift_axis,
        "cob_drift_magnitude": args_cli.cob_drift_magnitude,
        "cob_drift_window": [args_cli.cob_drift_start_step, args_cli.cob_drift_end_step],
        "pe_active_ratio": float(np.mean(pe_flags)) if pe_flags else 0.0,
        "deadzone_active_ratio": float(np.mean(deadzone_flags)) if deadzone_flags else 0.0,
        "ang_vel_norm_mean": float(np.mean(ang_vel_norm_history)) if ang_vel_norm_history else 0.0,
        "ang_vel_norm_max": float(np.max(ang_vel_norm_history)) if ang_vel_norm_history else 0.0,
        "axis_names": list(AXIS_NAMES),
    }
    if drift_in_window:
        ratios = np.stack(drift_in_window, axis=0)  # (T, 4)
        summary["zeta_runtime_over_nominal_mean"] = ratios.mean(axis=0).tolist()
        summary["zeta_runtime_over_nominal_max"] = ratios.max(axis=0).tolist()
        summary["zeta_runtime_over_nominal_min"] = ratios.min(axis=0).tolist()
    with open(summary_path, "w") as fp:
        json.dump(summary, fp, indent=2)
    print("[INFO] Summary:", json.dumps(summary, indent=2))

    # ----- ζ 时间轴 PNG -----
    try:
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(4, 1, figsize=(10, 8), sharex=True)
        steps_axis = np.arange(n_total) * sim_dt
        for i, name in enumerate(AXIS_NAMES):
            ax = axes[i]
            ax.plot(steps_axis, timeline[name]["runtime"], label=f"ζ1_{name} runtime", linewidth=1.0)
            ax.plot(steps_axis, timeline[name]["nominal"], label=f"ζ1_{name} nominal", linestyle="--", linewidth=0.8)
            ax.set_ylabel(name)
            ax.legend(loc="upper right", fontsize=8)
            ax.grid(True, linestyle=":", linewidth=0.5)
            if args_cli.cob_drift_magnitude != 0.0:
                ax.axvspan(
                    args_cli.cob_drift_start_step * sim_dt,
                    args_cli.cob_drift_end_step * sim_dt,
                    color="orange", alpha=0.1, label="drift window",
                )
        axes[-1].set_xlabel("sim_time (s)")
        fig.suptitle(
            f"ζ1 timeline | drift={args_cli.cob_drift_axis}+{args_cli.cob_drift_magnitude}m",
            fontsize=10,
        )
        fig.tight_layout()
        fig.savefig(png_path, dpi=120)
        plt.close(fig)
        print(f"[INFO] Saved ζ timeline plot to {png_path}")
    except Exception as exc:
        # 绘图失败不影响 CSV/JSON 输出
        print(f"[WARN] Failed to render ζ timeline plot: {exc}")

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()

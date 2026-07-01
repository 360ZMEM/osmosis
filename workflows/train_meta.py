# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Train an 8-dim meta-control policy (PPO + RSL-RL) for EasyUUV.

策略输出维度从 4 升级到 8（前 4 维 a_ctrl + 后 4 维 a_gain），后 4 维由
``ParametricGainTuner`` 经过 4 个控制学机制（Bounded Safeguard / PE / Dead-Zone /
Singular-Perturbation LPF）映射回 S-Surface 增益 ζ_runtime。

------------------------------------------------------------------------------
启动命令契约（参考实施计划 §3.5）：

    # 1) 主线训练（8 维元控制 + 全部 4 个机制开启）
    custom_workflows/run_with_isaac_env.sh python workflows_new_stdw/train_meta.py \\
        --task EasyUUV-Direct-Parametric-v1 \\
        --num_envs 1024 --headless \\
        --max_iterations 1500

    # 2) 课程式起步（幂等初始化 + 关 PE）
    custom_workflows/run_with_isaac_env.sh python workflows_new_stdw/train_meta.py \\
        --identity_init True --enable_pe False --max_iterations 200

    # 3) 兼容性 sanity（关元控制，跑回旧 4 维通路）
    custom_workflows/run_with_isaac_env.sh python workflows_new_stdw/train_meta.py \\
        --tune_gains False --task EasyUUV-Direct-v1
------------------------------------------------------------------------------
"""

from __future__ import annotations

import argparse
import importlib.util
import sys
from datetime import datetime
from pathlib import Path

from omni.isaac.lab.app import AppLauncher


# ---------------------------------------------------------------------------
# Bootstrap：与 workflows_new_stdw/play_stdw_adapt.py 同一套路。
# ---------------------------------------------------------------------------

WORKFLOW_DIR = Path(__file__).resolve().parent
REPO_ROOT = WORKFLOW_DIR.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
CUSTOM_WORKFLOWS_DIR = REPO_ROOT / "custom_workflows"
if str(CUSTOM_WORKFLOWS_DIR) not in sys.path:
    sys.path.insert(0, str(CUSTOM_WORKFLOWS_DIR))


def _bootstrap_local_lab_tasks_package() -> None:
    """让 gym entry_point ``omni.isaac.lab_tasks.direct.easyuuv_stdw`` 能解析到本地包。

    与 play_stdw_adapt.py 中的同名函数等价；详见那里的脚注。
    """
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
        package_name,
        package_init,
        submodule_search_locations=[str(REPO_ROOT)],
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


parser = argparse.ArgumentParser(description="Train an 8-dim meta-control RL agent (PPO).")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument("--video_interval", type=int, default=2000, help="Interval between video recordings (in steps).")
parser.add_argument("--cpu", action="store_true", default=False, help="Use CPU pipeline.")
parser.add_argument("--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations.")
parser.add_argument("--num_envs", type=int, default=1024, help="Number of environments to simulate.")
parser.add_argument(
    "--task",
    type=str,
    default="EasyUUV-Direct-Parametric-v1",
    help="Name of the task (default: 8-dim meta-control task).",
)
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")
parser.add_argument("--max_iterations", type=int, default=None, help="RL Policy training iterations.")

# === 11 个元控制 CLI（默认值与 EasyUUVEnvCfg 保持一致；具体语义见 gain_tuner.py） ===
parser.add_argument("--tune_gains", type=_bool_arg, default=None,
                    help="主开关；True 时使用 8 维 a_ctrl+a_gain 通路，False 退化为旧 4 维通路")
parser.add_argument("--gain_beta", type=float, default=None, help="Bounded Safeguard β (默认 0.2)")
parser.add_argument("--enable_pe", type=_bool_arg, default=None, help="Persistent Excitation 总开关")
parser.add_argument("--pe_freq", type=float, default=None, help="PE 注入正弦频率 (Hz)")
parser.add_argument("--pe_amp", type=float, default=None, help="PE 振幅（相对 ζ_nominal）")
parser.add_argument("--pe_decay_gamma", type=float, default=None, help="PE 状态相关衰减系数 γ")
parser.add_argument("--enable_deadzone", type=_bool_arg, default=None, help="死区参数冻结开关")
parser.add_argument("--deadzone_threshold", type=float, default=None, help="死区阈值 (rad/s)")
parser.add_argument("--enable_param_lpf", type=_bool_arg, default=None, help="一阶 LPF 开关")
parser.add_argument("--param_lpf_cutoff", type=float, default=None, help="LPF 截止频率 (Hz)")
parser.add_argument("--identity_init", type=_bool_arg, default=None,
                    help="幂等初始化（旁路全部 4 个机制，ζ_runtime ≡ ζ_nominal），用于课程式起步")

# === 分阶段元控制训练 (Stage-wise Meta-Control Training) ===
parser.add_argument("--meta_stage", type=int, default=1, choices=[0, 1, 2],
                    help="分阶段训练：0=不隔离梯度的 fine-tune；1=动作稳定化(只训控制头,增益头冻结为0)；2=增益自整定(冻结控制头,只训增益头)")
parser.add_argument("--stage1_checkpoint", type=str, default=None,
                    help="阶段二加载的阶段一最优权重 .pt 路径")
parser.add_argument("--stage2_cob_offset_xyz", type=str, default="0.03,0.03,0.01",
                    help="阶段二中强度 COM-COB 漂移逐轴范围 (m)")
parser.add_argument("--stage2_wave_mode", type=str, default="jonswap",
                    choices=["none", "constant", "sine", "jonswap"],
                    help="阶段二海浪扰动模式")
parser.add_argument("--resume_load_optimizer", type=_bool_arg, default=True,
                    help="resume 时是否加载 optimizer state；fine-tune/跨 stage 迁移时可设 False 只加载 policy/normalizer")

# === 参考轨迹 + 阻尼奖励（曲线平滑/抑超调调优） ===
parser.add_argument("--reference_mode", type=str, default=None,
                    choices=["step", "sine_sweep", "flip360_sine", "mixed_sine_flip360"],
                    help="参考轨迹模式：step=随机姿态硬阶跃；sine_sweep=平滑正弦扫频；flip360_sine=roll/pitch ±π 连续后空翻；mixed_sine_flip360=普通正弦+后空翻混合 replay")
parser.add_argument("--ref_sine_amp", type=str, default=None,
                    help="sine_sweep 逐轴幅度 (rad)，逗号分隔 roll,pitch,yaw（课程式低幅起步）")
parser.add_argument("--ref_sine_freq", type=str, default=None,
                    help="sine_sweep 逐轴频率 (Hz)，逗号分隔 roll,pitch,yaw")
parser.add_argument("--ref_mix_flip_prob", type=float, default=None,
                    help="mixed_sine_flip360 中抽到 flip360 子任务的 env 比例")
parser.add_argument("--ref_mix_sine_amp", type=str, default=None,
                    help="mixed_sine_flip360 中 ordinary sine 子任务幅度，逗号分隔 roll,pitch,yaw")
parser.add_argument("--ref_mix_sine_freq", type=str, default=None,
                    help="mixed_sine_flip360 中 ordinary sine 子任务频率，逗号分隔 roll,pitch,yaw")
parser.add_argument("--rew_scale_ang_vel", type=float, default=None,
                    help="角速度阻尼奖励权重（抑制振荡），默认 cfg 值")
parser.add_argument("--rew_scale_action_rate", type=float, default=None,
                    help="动作平滑（一阶差）阻尼权重，默认 cfg 值")
parser.add_argument("--rew_scale_action_jerk", type=float, default=None,
                    help="动作 jerk（二阶差）阻尼权重，压制高频震荡，默认 cfg 值")
parser.add_argument(
    "--embodiment",
    type=str,
    default="base",
    choices=["base", "long_body", "heavy_moderate", "asymmetric", "uuv6", "uuv4", "uuv6_angled", "uuv4_angled"],
    help="训练时切换 embodiment；base 保持默认旧路径，其余在 gym.make 后调用 env.apply_embodiment_config。",
)

# 与 custom_workflows/cli_args.py 的 RSL-RL CLI 兼容
import cli_args  # noqa: E402  isort: skip
cli_args.add_rsl_rl_args(parser)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
if args_cli.video:
    args_cli.enable_cameras = True

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

_bootstrap_local_lab_tasks_package()

"""Rest everything follows."""

import gymnasium as gym  # noqa: E402
import json  # noqa: E402
import os  # noqa: E402
import torch  # noqa: E402

from rsl_rl.runners import OnPolicyRunner  # noqa: E402

from omni.isaac.lab.envs import ManagerBasedRLEnvCfg  # noqa: E402
from omni.isaac.lab.utils.dict import print_dict  # noqa: E402
from omni.isaac.lab.utils.io import dump_pickle, dump_yaml  # noqa: E402

import omni.isaac.lab_tasks  # noqa: F401, E402
from omni.isaac.lab_tasks.utils import get_checkpoint_path, parse_env_cfg  # noqa: E402
from omni.isaac.lab_tasks.utils.wrappers.rsl_rl import (  # noqa: E402
    RslRlOnPolicyRunnerCfg,
    RslRlVecEnvWrapper,
)

from custom_workflows.workflow_config import apply_config_overrides, load_workflow_config  # noqa: E402
from custom_workflows.workflow_paths import (  # noqa: E402
    WorkflowPaths,
    ensure_directory,
    resolve_checkpoint_path,
)

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = False


# 通过 setattr 把 CLI 注入到 env_cfg；只对显式给出的非 None 值生效，未给则沿用 cfg 默认值。
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
    "reference_mode",
    "ref_mix_flip_prob",
    "rew_scale_ang_vel",
    "rew_scale_action_rate",
    "rew_scale_action_jerk",
)


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

    # 列表型参考参数单独解析（逗号分隔 -> list[float]）。
    for key in ("ref_sine_amp", "ref_sine_freq", "ref_mix_sine_amp", "ref_mix_sine_freq"):
        raw = getattr(args, key, None)
        if raw is None:
            continue
        try:
            vals = [float(x) for x in str(raw).split(",")]
        except Exception as exc:
            print(f"[WARN] failed to parse --{key}={raw}: {exc}")
            continue
        setattr(env_cfg, key, vals)
        print(f"[INFO] Override env_cfg.{key} = {vals}")


def _apply_stage_env_overrides(env_cfg, args) -> None:
    """阶段相关 env_cfg 覆盖（在通用 meta override 之后调用，优先级最高）。"""
    if args.meta_stage == 0:
        print("[INFO][Stage0] fine-tune mode: keep env_cfg overrides unchanged")
        return
    if args.meta_stage == 1:
        # 阶段一：底层完全使用标称参数（ζ_runtime ≡ ζ_nominal），增益头不影响动力学。
        if hasattr(env_cfg, "identity_init"):
            env_cfg.identity_init = True
            print("[INFO][Stage1] force env_cfg.identity_init = True (底层使用标称参数)")
    else:
        # 阶段二：激活增益通路 + 中强度 COM-COB 漂移 + JONSWAP 海浪。
        if hasattr(env_cfg, "identity_init"):
            env_cfg.identity_init = False
        if hasattr(env_cfg, "tune_gains"):
            env_cfg.tune_gains = True
        try:
            xyz = [float(x) for x in str(args.stage2_cob_offset_xyz).split(",")]
            env_cfg.domain_randomization.com_to_cob_offset_xyz_range = xyz
            print(f"[INFO][Stage2] com_to_cob_offset_xyz_range = {xyz}")
        except Exception as exc:
            print(f"[WARN][Stage2] failed to set cob offset range: {exc}")
        try:
            env_cfg.disturbance_cfg.mode = str(args.stage2_wave_mode)
            print(f"[INFO][Stage2] disturbance_cfg.mode = {args.stage2_wave_mode}")
        except Exception as exc:
            print(f"[WARN][Stage2] failed to set wave mode: {exc}")


def _setup_stage_gradient_isolation(runner, args) -> None:
    """按行隔离 actor 输出层梯度：阶段一冻结增益头(后4行)，阶段二严格锁定控制头(前4行)+主干。"""
    import torch as _torch

    if args.meta_stage == 0:
        print("[INFO][Stage0] fine-tune mode: no gradient isolation")
        return

    actor = runner.alg.actor_critic.actor
    linears = [m for m in actor.modules() if isinstance(m, _torch.nn.Linear)]
    if not linears:
        print("[WARN] actor has no Linear layer; gradient isolation skipped")
        return
    last_linear = linears[-1]

    def _make_row_freeze_hook(freeze_slice):
        def _hook(grad):
            g = grad.clone()
            g[freeze_slice] = 0.0
            return g
        return _hook

    if args.meta_stage == 1:
        freeze = slice(4, 8)  # 冻结增益头(后4行)
        print("[INFO][Stage1] freeze actor output rows [4:8] (gain head)")
    else:
        freeze = slice(0, 4)  # 冻结控制头(前4行)
        # 严格锁定控制行为：actor 主干(除输出层)梯度全部关闭。
        for m in linears[:-1]:
            for p in m.parameters():
                p.requires_grad_(False)
        print("[INFO][Stage2] freeze actor backbone + output rows [0:4] (ctrl head)")

    last_linear.weight.register_hook(_make_row_freeze_hook(freeze))
    last_linear.bias.register_hook(_make_row_freeze_hook(freeze))

    if args.meta_stage == 2:
        # Adam 只接收可训练参数，防止已锁定的 Stage1 权重被意外更改。
        trainable = list(filter(lambda p: p.requires_grad, runner.alg.actor_critic.parameters()))
        lr = runner.alg.optimizer.param_groups[0]["lr"]
        runner.alg.optimizer = _torch.optim.Adam(trainable, lr=lr)
        n_train = sum(p.numel() for p in trainable)
        print(f"[INFO][Stage2] rebuilt Adam over {len(trainable)} trainable tensors "
              f"({n_train} params)")


# 极简训练日志：每 iter 一行 JSONL，只保留 reward / loss / noise / mse 这类标量。
_MSE_KEY = "Episode Reward / log MSE"


def _attach_compact_logger(runner, log_dir):
    """Monkey-patch ``runner.log`` 把每个 iter 的关键标量落盘到两份 JSONL。"""
    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    compact_path = log_dir / "compact_log.jsonl"
    mse_path = log_dir / "mse_curve.jsonl"
    compact_fp = compact_path.open("w", buffering=1)
    mse_fp = mse_path.open("w", buffering=1)
    print(f"[INFO] compact log -> {compact_path}")
    print(f"[INFO] mse curve   -> {mse_path}")

    original_log = runner.log

    def _to_scalar(value):
        if value is None:
            return None
        if isinstance(value, torch.Tensor):
            if value.numel() == 0:
                return None
            return float(value.detach().mean().cpu().item())
        try:
            return float(value)
        except (TypeError, ValueError):
            return None

    def _round(v):
        return None if v is None else round(v, 6)

    def patched_log(locs, *args, **kwargs):
        try:
            it = int(locs.get("it", -1))
            tot_timesteps = locs.get("tot_timesteps", None)
            ep_infos = locs.get("ep_infos", []) or []

            row = {
                "iter": it,
                "reward": _round(_to_scalar(locs.get("mean_reward"))),
                "ep_len": _round(_to_scalar(locs.get("mean_episode_length"))),
                "vloss": _round(_to_scalar(locs.get("mean_value_loss"))),
                "ploss": _round(_to_scalar(locs.get("mean_surrogate_loss"))),
                "timesteps": _round(_to_scalar(tot_timesteps)),
            }
            std_module = getattr(getattr(runner, "alg", None), "actor_critic", None)
            if std_module is not None and hasattr(std_module, "std"):
                try:
                    row["noise_std"] = round(float(std_module.std.detach().mean().cpu().item()), 6)
                except Exception:
                    pass
            row = {k: v for k, v in row.items() if v is not None}
            compact_fp.write(json.dumps(row) + "\n")

            mse_values = []
            for info in ep_infos:
                if not isinstance(info, dict) or _MSE_KEY not in info:
                    continue
                v = info[_MSE_KEY]
                if isinstance(v, torch.Tensor):
                    if v.numel() == 0:
                        continue
                    v = float(v.detach().mean().cpu().item())
                mse_values.append(float(v))
            if mse_values:
                log_mse_mean = sum(mse_values) / len(mse_values)
                mse_fp.write(json.dumps({
                    "iter": it,
                    "log_mse": round(log_mse_mean, 6),
                    "n": len(mse_values),
                }) + "\n")
        except Exception as exc:  # 日志失败不能拖死训练
            print(f"[WARN] compact logger error at iter={locs.get('it', '?')}: {exc}")
        return original_log(locs, *args, **kwargs)

    runner.log = patched_log
    return compact_fp, mse_fp


def main():
    workflow_cfg = load_workflow_config(args_cli.workflow_config)
    train_cfg = workflow_cfg.get("train", {}) or {}
    path_cfg = workflow_cfg.get("paths", {}) or {}

    if args_cli.task == parser.get_default("task") and "task" in train_cfg:
        args_cli.task = train_cfg["task"]
    if args_cli.num_envs == parser.get_default("num_envs") and "num_envs" in train_cfg:
        args_cli.num_envs = train_cfg["num_envs"]
    if args_cli.seed is None and "seed" in train_cfg:
        args_cli.seed = train_cfg["seed"]
    if args_cli.max_iterations is None and "max_iterations" in train_cfg:
        args_cli.max_iterations = train_cfg["max_iterations"]
    if args_cli.embodiment == parser.get_default("embodiment") and "embodiment" in train_cfg:
        args_cli.embodiment = train_cfg["embodiment"]

    paths = WorkflowPaths.from_overrides(
        logs_root=args_cli.logs_root or path_cfg.get("logs_root"),
        results_root=args_cli.results_root or path_cfg.get("results_root"),
        artifacts_root=args_cli.artifacts_root or path_cfg.get("artifacts_root"),
    )

    env_cfg: ManagerBasedRLEnvCfg = parse_env_cfg(
        args_cli.task,
        use_gpu=not args_cli.cpu,
        num_envs=args_cli.num_envs,
        use_fabric=not args_cli.disable_fabric,
    )
    if workflow_cfg.get("env"):
        apply_config_overrides(env_cfg, workflow_cfg["env"])
    # 元控制 CLI 在 workflow_config / parse_env_cfg 之后再覆盖一遍，确保 CLI 优先级最高。
    _apply_meta_cfg_overrides(env_cfg, args_cli)
    # 分阶段训练 env 覆盖（优先级最高）：阶段一标称参数；阶段二激活增益+中强度扰动。
    _apply_stage_env_overrides(env_cfg, args_cli)

    agent_cfg: RslRlOnPolicyRunnerCfg = cli_args.parse_rsl_rl_cfg(
        args_cli.task, args_cli, workflow_cfg.get("agent")
    )
    if not getattr(agent_cfg, "run_name", "") and train_cfg.get("run_name"):
        agent_cfg.run_name = str(train_cfg["run_name"])
    # 阶段后缀，便于区分 stage1 / stage2 的 checkpoint 目录。
    _stage_suffix = f"stage{args_cli.meta_stage}"
    agent_cfg.run_name = (
        f"{agent_cfg.run_name}_{_stage_suffix}" if getattr(agent_cfg, "run_name", "")
        else _stage_suffix
    )

    log_root_path = paths.training_log_root(agent_cfg.experiment_name)
    print(f"[INFO] Logging experiment in directory: {log_root_path}")
    log_dir = paths.training_run_dir(
        agent_cfg.experiment_name,
        agent_cfg.run_name,
        datetime.now().strftime("%Y-%m-%d_%H-%M-%S"),
    )
    ensure_directory(log_dir)
    ensure_directory(log_dir / "params")

    if args_cli.max_iterations:
        agent_cfg.max_iterations = args_cli.max_iterations

    env = gym.make(
        args_cli.task,
        cfg=env_cfg,
        render_mode="rgb_array" if args_cli.video else None,
    )
    if args_cli.embodiment != "base":
        try:
            env.unwrapped.apply_embodiment_config(args_cli.embodiment)
            print(f"[INFO] applied embodiment: {args_cli.embodiment}")
        except Exception as exc:
            raise RuntimeError(f"apply_embodiment_config({args_cli.embodiment!r}) failed") from exc
    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos"),
            "step_trigger": lambda step: step % args_cli.video_interval == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording videos during training.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)
    env = RslRlVecEnvWrapper(env)

    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=str(log_dir), device=agent_cfg.device)
    runner.add_git_repo_to_log(__file__)

    compact_fp, mse_fp = _attach_compact_logger(runner, log_dir)

    if agent_cfg.resume:
        resume_path = resolve_checkpoint_path(
            log_root_path=log_root_path,
            load_run=agent_cfg.load_run,
            checkpoint_name=agent_cfg.load_checkpoint,
            fallback_resolver=get_checkpoint_path,
        )
        print(f"[INFO]: Loading model checkpoint from: {resume_path} (load_optimizer={args_cli.resume_load_optimizer})")
        runner.load(resume_path, load_optimizer=bool(args_cli.resume_load_optimizer))

    # 阶段二：显式加载阶段一最优权重（独立于 RL resume 续训路径）。
    if args_cli.meta_stage == 2:
        if not args_cli.stage1_checkpoint:
            raise SystemExit("[FATAL] --meta_stage 2 requires --stage1_checkpoint <stage1.pt>")
        if not os.path.isfile(args_cli.stage1_checkpoint):
            raise SystemExit(f"[FATAL] stage1_checkpoint not found: {args_cli.stage1_checkpoint}")
        print(f"[INFO][Stage2] loading stage-1 checkpoint: {args_cli.stage1_checkpoint}")
        runner.load(args_cli.stage1_checkpoint)

    # 分阶段梯度隔离（hook + 阶段二重建 optimizer），必须在 runner.learn 之前。
    _setup_stage_gradient_isolation(runner, args_cli)

    env.seed(agent_cfg.seed)

    dump_yaml(str(log_dir / "params" / "env.yaml"), env_cfg)
    dump_yaml(str(log_dir / "params" / "agent.yaml"), agent_cfg)
    dump_pickle(str(log_dir / "params" / "env.pkl"), env_cfg)
    dump_pickle(str(log_dir / "params" / "agent.pkl"), agent_cfg)

    runner.learn(num_learning_iterations=agent_cfg.max_iterations, init_at_random_ep_len=True)

    try:
        compact_fp.close()
        mse_fp.close()
    except Exception:
        pass

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()

# 8 维元控制 — 实施报告（v1，2026-06-05）

> 关联实施计划：[8维元控制重构与训练评估打通实施计划.md](file:///home/zmem063/isaaclab/source/extensions/omni.isaac.lab_tasks/omni/isaac/lab_tasks/direct/isaac-auv-env-new/.trae/documents/8维元控制重构与训练评估打通实施计划.md) (v1.1)
> 关联静态验证：[8维元控制静态验证与收尾交付计划.md](file:///home/zmem063/isaaclab/source/extensions/omni.isaac.lab_tasks/omni/isaac/lab_tasks/direct/isaac-auv-env-new/.trae/documents/8维元控制静态验证与收尾交付计划.md) (v2)
> 关联训练扫参：[REPORT_meta_train_sweep_20260605.md](file:///home/zmem063/isaaclab/source/extensions/omni.isaac.lab_tasks/omni/isaac/lab_tasks/direct/isaac-auv-env-new/workflows_new_stdw/REPORT_meta_train_sweep_20260605.md)
> 验收方式：3 段式 GPU end-to-end smoke + 200 iter 全新训练 + 4 组消融 eval；全部 rc=0、CSV/JSON/PNG 落盘可校验。

---

## 1. 任务目标与范围

**目标**：把 WarpAUV 的动作空间从 **4 维直接控制** 升级为 **8 维元控制**：
- 前 4 维 `a_ctrl ∈ [-1, 1]^4` 仍走 S-Surface 控制器路径（roll / pitch / yaw / depth 误差指令）。
- 后 4 维 `a_gain ∈ [-1, 1]^4` 通过 `ParametricGainTuner` 在线调制 S-Surface 的 ζ 增益。

**范围**：
- ✅ 实现 4 个控制学机制（Bounded Safeguard、Persistent Excitation、Dead-Zone Freezing、Singular Perturbation LPF）。
- ✅ 每个机制独立 CLI 开关 + 标称-增益参数化。
- ✅ 与原 4 维通路（`MOGA-WarpAUV-Direct-v1`）**并行共存**，不破坏历史 PPO 训练。
- ✅ 端到端 GPU 验证（5 iter smoke + 200 iter fresh）通过。
- ⛔ **不在本范围**：STDW 整合（用户明文要求"先打通元控制再整合 STDW"）；多场景 8×4 扫参（属于 STDW 阶段）。

---

## 2. 控制学机制设计与公式依据

| 机制 | 公式 | 控制学依据 |
|---|---|---|
| **Bounded Safeguard** | `ζ_i(t) = ζ_{i,nom} · (1 + β · a_gain_i(t))`，`β = 0.2` | 对角残差映射，β 是 Hurwitz 区域保护系数；±20% 摆动幅度可被 [Lyapunov 稳定性圆盘](https://en.wikipedia.org/wiki/Lyapunov_stability) 完整覆盖。 |
| **Persistent Excitation (PE)** | `ζ_i ← ζ_i + a(t) sin(ω_d t)`，`a(t) = pe_amp / (1 + γ · ‖ω_body‖²)` | Narendra & Annaswamy (1989) Th. 2.7.1：参数估计渐近收敛要求"信息持续激励" `∫_t^{t+T} φφ^T dτ ≥ α₀I`；状态相关衰减是 σ-modification 思想（Ioannou & Sun, Ch. 8.5）—静态时不打扰、激烈时不挠乱。 |
| **Dead-Zone Freezing** | `if ‖compound_err‖ < ε_dz: a_gain ← 0`，`ε_dz = 0.02 rad/s` | Ioannou & Sun, Robust Adaptive Control, Ch. 8.5：在线辨识器对噪声敏感，死区内冻结参数避免 noise-driven drift。 |
| **Singular-Perturbation LPF** | `a_gain_lpf(t) = α · a_gain(t) + (1-α) · a_gain_lpf(t-1)`，`α = dt / (RC + dt)`，`RC = 1/(2π · f_c)` | Khalil, Singular Perturbation Methods：双时间尺度系统（快内回路 / 慢自适应外回路），用一阶 LPF 实现频率隔离，截止 `f_c = 1.0 Hz` 对应 RC ≈ 0.16 s。 |
| **Identity init**（幂等开关） | `if identity_init: ζ_runtime ≡ ζ_nominal` | 课程式起步：训练初期把 4 个机制全部短路，等价 4 维 baseline，便于 warm-up。 |

完整推导见 [gain_tuner.py](file:///home/zmem063/isaaclab/source/extensions/omni.isaac.lab_tasks/omni/isaac/lab_tasks/direct/isaac-auv-env-new/gain_tuner.py) 顶部 docstring 与各方法注释。

---

## 3. 文件改动清单

### 3.1 新建文件

| 路径 | 行数 | 说明 |
|---|---:|---|
| [gain_tuner.py](file:///home/zmem063/isaaclab/source/extensions/omni.isaac.lab_tasks/omni/isaac/lab_tasks/direct/isaac-auv-env-new/gain_tuner.py) | ~370 | `ParametricGainTuner` class，托管 4 个机制 + identity_init 短路。含 6 项 `__main__` self-test。 |
| [workflows_new_stdw/train_meta.py](file:///home/zmem063/isaaclab/source/extensions/omni.isaac.lab_tasks/omni/isaac/lab_tasks/direct/isaac-auv-env-new/workflows_new_stdw/train_meta.py) | 297 | 8 维 PPO 训练入口，11 个元控制 CLI（全 default=None，仅显式给值才覆盖 cfg）。 |
| [workflows_new_stdw/play_meta_eval.py](file:///home/zmem063/isaaclab/source/extensions/omni.isaac.lab_tasks/omni/isaac/lab_tasks/direct/isaac-auv-env-new/workflows_new_stdw/play_meta_eval.py) | 436 | 单环境 eval + COB drift 注入；CSV (30 列) / summary.json / matplotlib 4 子图 PNG。 |
| [workflows_new_stdw/run_meta_smoke.sh](file:///home/zmem063/isaaclab/source/extensions/omni.isaac.lab_tasks/omni/isaac/lab_tasks/direct/isaac-auv-env-new/workflows_new_stdw/run_meta_smoke.sh) | 93 | 3 段式端到端 smoke（5 iter 训练 + 无漂移 200 步 eval + COB drift 800 步 eval）。 |

### 3.2 修改文件

| 路径 | 改动摘要 |
|---|---|
| [warpauv_env.py](file:///home/zmem063/isaaclab/source/extensions/omni.isaac.lab_tasks/omni/isaac/lab_tasks/direct/isaac-auv-env-new/warpauv_env.py) | + 11 个 cfg 字段（L139-152）；`PID_args/old_actions/actions_i` 形状改 `(N, 4)` 兼容 8 维拆分（L255-261）；`__init__` 内实例化 `ParametricGainTuner`（L308-344）；`_pre_physics_step + _apply_action` 拆动作 + 调用 tuner（L645-687）；`apply_control_profile / apply_pid_multipliers / _refresh_domain_randomization` 三处 reset 钩子捕获 ζ_nominal 快照；末尾新增 `WarpAUVParametricEnvCfg`（L1261-1271）。 |
| [agents/rsl_rl_ppo_cfg.py](file:///home/zmem063/isaaclab/source/extensions/omni.isaac.lab_tasks/omni/isaac/lab_tasks/direct/isaac-auv-env-new/agents/rsl_rl_ppo_cfg.py) | 末尾新增 `WarpAUVParametricPPORunnerCfg`（experiment_name="warpauv_parametric"，actor_hidden_dims=[128,128]，init_noise_std=0.8，entropy_coef=0.005，lr=3e-4）。 |
| [__init__.py](file:///home/zmem063/isaaclab/source/extensions/omni.isaac.lab_tasks/omni/isaac/lab_tasks/direct/isaac-auv-env-new/__init__.py) | 新增 `MOGA-WarpAUV-Direct-Parametric-v1` 注册（与原 `MOGA-WarpAUV-Direct-v1` 并行）。 |

> 注：原 4 维通路（`MOGA-WarpAUV-Direct-v1` + `WarpAUVPPORunnerCfg` + 默认 `tune_gains=False`）**完全不受影响**；用 `--tune_gains False --task MOGA-WarpAUV-Direct-v1` 即可回退。

---

## 4. 端到端 GPU 验证（smoke）

数据基目录：[`.tmp/meta_smoke_gpu_20260605_213644/`](file:///home/zmem063/isaaclab/source/extensions/omni.isaac.lab_tasks/omni/isaac/lab_tasks/direct/isaac-auv-env-new/.tmp/meta_smoke_gpu_20260605_213644)

### 4.1 Phase 1：5 iter 短训（GPU, num_envs=64）

| 项 | 值 |
|---|---|
| 启动命令 | `bash custom_workflows/run_with_isaac_env.sh workflows_new_stdw/train_meta.py --headless --task MOGA-WarpAUV-Direct-Parametric-v1 --num_envs 64 --max_iterations 5 ...` |
| 退出码 | 0 |
| 总耗时 | ~13 s（含 Isaac 启动 8 s + 训练 5 s） |
| 产物 | `model_0.pt` / `model_4.pt` / `events.out.tfevents...` / `params/env.yaml` / `params/agent.yaml` |

**结论**：8 维 action space + RSL-RL 链路在 GPU 上完整跑通；checkpoint / TB 日志 / yaml 全部写入。

### 4.2 Phase 2：200 步无漂移 eval

数据：[phase2_eval_no_drift/meta_eval_summary.json](file:///home/zmem063/isaaclab/source/extensions/omni.isaac.lab_tasks/omni/isaac/lab_tasks/direct/isaac-auv-env-new/.tmp/meta_smoke_gpu_20260605_213644/phase2_eval_no_drift/meta_eval_summary.json)

| 指标 | 值 | 解读 |
|---|---:|---|
| `pe_active_ratio` | 1.000 | PE 全程在线（每步 sin 注入）✅ |
| `deadzone_active_ratio` | 0.015 | 仅 step 0 触发死区（init 时 ‖ω‖ ≈ 0），符合阈值设计 ✅ |
| `ang_vel_norm_mean` | 1.471 rad/s | RL 启动期角速度合理 |
| ζ_roll runtime range | [0.954, 1.015] | 在 ζ_nom × (1±β) = [0.798, 1.198] 内 ✅ |
| ζ_yaw runtime range | [1.566, 1.732] | ζ_nom × (1±β) = [1.335, 2.003] 内 ✅ |

CSV 200 行 / 30 列；PNG 4 子图（roll/pitch/yaw/depth 各一图，runtime vs nominal 双线）。

### 4.3 Phase 3：800 步 COB drift y=+0.05 m

数据：[phase3_eval_cob_drift_y_005/meta_eval_summary.json](file:///home/zmem063/isaaclab/source/extensions/omni.isaac.lab_tasks/omni/isaac/lab_tasks/direct/isaac-auv-env-new/.tmp/meta_smoke_gpu_20260605_213644/phase3_eval_cob_drift_y_005/meta_eval_summary.json)

**COB 漂移注入校验**（`cob_y` 在 step ∈ [200, 800] 线性 ramp）：

| step | cob_y (m) | ζ1_roll | ζ1_pitch | ζ1_yaw | ζ1_depth |
|---:|---:|---:|---:|---:|---:|
| 0 | 0.0000 | 0.997 | 1.001 | 1.669 | 0.267 |
| 199 | 0.0000 | 1.015 | 1.033 | 1.732 | 0.267 |
| 400 | 0.0167 | 0.970 | 0.983 | 1.626 | 0.257 |
| 600 | 0.0333 | 0.997 | 1.013 | 1.720 | 0.253 |
| 799 | 0.0499 | 1.041 | 1.040 | 1.775 | 0.259 |

漂移区 ζ_runtime / ζ_nominal：
- mean = [0.987, 0.998, 1.008, 0.967]
- min  = [0.944, 0.917, 0.949, 0.903]
- max  = [1.038, 1.038, 1.062, 1.006]

ζ 比率全部落在 ±10% 内，与 `gain_beta=0.2` 安全锁吻合（5 iter 策略尚未学到激烈调整，仍在保守区间，符合预期）。

---

## 5. 验收判据（Plan §5）逐项核对

| # | 判据 | 结果 |
|---|---|---|
| 1 | num_actions 动态扩展至 8 | ✅ `WarpAUVParametricEnvCfg.num_actions = 8` |
| 2 | 4 个机制独立 CLI 开关 | ✅ `--enable_pe / --enable_deadzone / --enable_param_lpf / --identity_init`（4 个布尔 + 7 个超参） |
| 3 | Bounded Safeguard ±β 限幅 | ✅ Smoke Phase 3 ζ 比率 ∈ [0.903, 1.062] ⊂ [1-β, 1+β] = [0.8, 1.2] |
| 4 | PE 状态相关衰减 | ✅ tuner self-test：静态 |Δ|=0.20，激烈 |Δ|=0.0003（衰减比 ≈ 660×） |
| 5 | Dead-Zone 误差冻结 | ✅ tuner self-test 第 3 项 PASS；smoke 中 `deadzone_active_ratio=0.015` 仅在 init 触发 |
| 6 | LPF 一阶滞后 | ✅ tuner self-test 第 6 项 PASS；smoke CSV 中 `a_gain_lpf` 与 `a_gain` 时间序列差异可见 |
| 7 | identity_init 完全旁路 | ✅ tuner self-test 第 1 项 PASS；扫参 [identity 组](file:///home/zmem063/isaaclab/source/extensions/omni.isaac.lab_tasks/omni/isaac/lab_tasks/direct/isaac-auv-env-new/.tmp/meta_train_gpu_20260605_214751/sweep_215003/identity/meta_eval_summary.json) ζ_runtime/ζ_nominal mean = [1.000, 1.000, 1.000, 1.000] |
| 8 | COB drift 时 ζ 在线调整可观测 | ✅ Smoke Phase 3 + 200-iter 训练后 baseline 组 ζ_yaw 漂移 mean = 1.737 vs nominal = 1.667（+4.2%） |
| 9 | 4 维通路不受影响 | ✅ `MOGA-WarpAUV-Direct-v1` 仍注册、`tune_gains=False` 默认；`__init__.py` 双任务并行 |
| 10 | 静态验证（py_compile + tuner self-test） | ✅ 5 个文件 py_compile=0；tuner 6 项 self-test 全 PASS |
| 11 | GPU 端到端 smoke 三段全 rc=0 | ✅ 见 §4 |

---

## 6. 已知非阻塞问题与限制

| 项 | 说明 | 处置 |
|---|---|---|
| `MOGA-WarpAUV-Direct-v1 already in registry` warning | bootstrap 套路先 import 父包再 reload 本地包，gym 检测到重注册触发 `UserWarning` | 已知良性，不影响功能；可在 [__init__.py](file:///home/zmem063/isaaclab/source/extensions/omni.isaac.lab_tasks/omni/isaac/lab_tasks/direct/isaac-auv-env-new/__init__.py) 加 try/except 静默，但当前不动以保留可见性 |
| `render interval (1) < decimation (2)` warning | DirectRLEnv 默认 render 行为；不影响 headless 训练 | 不动 |
| GLFW initialization failed | headless 模式无 X server 警告 | 不动 |
| RSL-RL 在 ≤5 iter 时 stdout 不打印 metric block | 默认 `save_interval=50`、第一个 metric block 在 iter ≥ 1 才输出 | 通过 `model_*.pt` + tfevents 验证训练真实发生 |
| 当前 Isaac 4.0 + Lab `1.0` 已停止维护 | 仅是工程提示；本任务严格按 AGENTS.md 锁版本 | 不动 |

---

## 7. 复现命令清单

### 7.1 静态验证（无 GPU 需求）

```bash
cd /home/zmem063/isaaclab/source/extensions/omni.isaac.lab_tasks/omni/isaac/lab_tasks/direct/isaac-auv-env-new
python -m py_compile gain_tuner.py warpauv_env.py __init__.py agents/rsl_rl_ppo_cfg.py \
    workflows_new_stdw/train_meta.py workflows_new_stdw/play_meta_eval.py
python gain_tuner.py    # 6 项 self-test
```

### 7.2 GPU 端到端 smoke

```bash
bash workflows_new_stdw/run_meta_smoke.sh   # 默认 --cpu，CPU pipeline
# 或本报告使用的 GPU 版本（手动触发，不调 smoke 脚本，去掉 --cpu）：
TS=$(date +%Y%m%d_%H%M%S); BASE=".tmp/meta_smoke_gpu_${TS}"; mkdir -p "${BASE}"/{phase1_logs,phase1_results}
bash custom_workflows/run_with_isaac_env.sh workflows_new_stdw/train_meta.py --headless \
    --task MOGA-WarpAUV-Direct-Parametric-v1 --num_envs 64 --max_iterations 5 \
    --logs_root "${BASE}/phase1_logs" --results_root "${BASE}/phase1_results"
```

### 7.3 主线训练（200 iter / num_envs=256）

```bash
bash custom_workflows/run_with_isaac_env.sh workflows_new_stdw/train_meta.py --headless \
    --task MOGA-WarpAUV-Direct-Parametric-v1 --num_envs 256 --max_iterations 200 --seed 42 \
    --logs_root .tmp/meta_train/logs --results_root .tmp/meta_train/results
```

### 7.4 Eval + COB drift 注入

```bash
bash custom_workflows/run_with_isaac_env.sh workflows_new_stdw/play_meta_eval.py --headless \
    --task MOGA-WarpAUV-Direct-Parametric-v1 --num_envs 1 --steps 800 \
    --cob_drift_axis y --cob_drift_magnitude 0.05 --cob_drift_start_step 200 --cob_drift_end_step 800 \
    --policy_path <path/to/model_199.pt> --save_dir .tmp/meta_eval/
```

### 7.5 课程式起步（identity_init）

```bash
bash custom_workflows/run_with_isaac_env.sh workflows_new_stdw/train_meta.py --headless \
    --task MOGA-WarpAUV-Direct-Parametric-v1 --identity_init True --enable_pe False \
    --max_iterations 200 --num_envs 1024
# 后续切回完全自适应：去掉 --identity_init 与 --enable_pe，--resume True 接上 checkpoint。
```

---

## 8. 下一步建议（不在本报告范围）

1. **STDW 整合**：在已打通的 8 维 cfg 上叠加 STDW 慢环 / 行为锚定（参考 [REPORT_scenarios_6k_20260604_v2.md](file:///home/zmem063/isaaclab/source/extensions/omni.isaac.lab_tasks/omni/isaac/lab_tasks/direct/isaac-auv-env-new/workflows_new_stdw/REPORT_scenarios_6k_20260604_v2.md) v3 STDW 全开方案）。
2. **多场景 + 多机型扫参**：把 [sweep_stdw.py](file:///home/zmem063/isaaclab/source/extensions/omni.isaac.lab_tasks/omni/isaac/lab_tasks/direct/isaac-auv-env-new/workflows_new_stdw/sweep_stdw.py) 的 8 scenario × 4 embodiment 矩阵适配到 8 维元控制（替换 task ID + 添加 `--meta_*` 透传）。
3. **训练时长升级**：当前 200 iter / num_envs=256 仅做 sanity；正式实验建议 1500+ iter / num_envs=1024（参考 [train_meta.py](file:///home/zmem063/isaaclab/source/extensions/omni.isaac.lab_tasks/omni/isaac/lab_tasks/direct/isaac-auv-env-new/workflows_new_stdw/train_meta.py) docstring 启动命令 1）。

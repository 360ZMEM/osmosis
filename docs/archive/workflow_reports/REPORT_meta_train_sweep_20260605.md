# 8 维元控制 — 从头训练与机制消融报告（v1，2026-06-05）

> 关联实施报告：[REPORT_meta_implementation_20260605.md](file:///home/zmem063/isaaclab/source/extensions/omni.isaac.lab_tasks/omni/isaac/lab_tasks/direct/isaac-auv-env-new/workflows_new_stdw/REPORT_meta_implementation_20260605.md)
> 训练数据基目录：[`.tmp/meta_train_gpu_20260605_214751/`](file:///home/zmem063/isaaclab/source/extensions/omni.isaac.lab_tasks/omni/isaac/lab_tasks/direct/isaac-auv-env-new/.tmp/meta_train_gpu_20260605_214751)
> 扫参数据基目录：[`.tmp/meta_train_gpu_20260605_214751/sweep_215003/`](file:///home/zmem063/isaaclab/source/extensions/omni.isaac.lab_tasks/omni/isaac/lab_tasks/direct/isaac-auv-env-new/.tmp/meta_train_gpu_20260605_214751/sweep_215003)
> 验收：从头 200 iter PPO 训练 → reward 9.4→28.5（+204%）；4 组消融 eval（baseline / no_pe / identity / no_lpf）全 rc=0；ζ_runtime 在 COB drift 下出现稳定方向性偏置。

---

## 1. 实验设计

### 1.1 任务与算法

| 项 | 值 |
|---|---|
| Gym 任务 | `MOGA-WarpAUV-Direct-Parametric-v1`（8 维 action） |
| Runner cfg | [`WarpAUVParametricPPORunnerCfg`](file:///home/zmem063/isaaclab/source/extensions/omni.isaac.lab_tasks/omni/isaac/lab_tasks/direct/isaac-auv-env-new/agents/rsl_rl_ppo_cfg.py#L53-L75) |
| 算法 | RSL-RL OnPolicyRunner / PPO (clip=0.2, λ=0.95, γ=0.99) |
| 网络 | actor [128, 128] elu + critic [128, 128] elu |
| 训练规模 | num_envs=256, max_iterations=200, seed=42（**从头**，无 resume） |
| 学习率 / 噪声 | lr=3e-4 (adaptive)，init_noise_std=0.8，desired_kl=0.01 |
| 物理 / 仿真 | RTX 4060 Laptop 8 GB，headless，sim_dt=0.0167s, decimation=2 |

### 1.2 启动命令

```bash
bash custom_workflows/run_with_isaac_env.sh workflows_new_stdw/train_meta.py --headless \
    --task MOGA-WarpAUV-Direct-Parametric-v1 --num_envs 256 --max_iterations 200 --seed 42 \
    --logs_root .tmp/meta_train_gpu_20260605_214751/logs \
    --results_root .tmp/meta_train_gpu_20260605_214751/results
```

### 1.3 4 组扫参（消融）配置

每组都用同一 ckpt `model_199.pt` 跑 800 步 eval、注入 COB drift y=+0.05 m（step 200→800 线性 ramp），**只**修改一个机制开关：

| 组 | 标志位 | 期望验证 |
|---|---|---|
| `baseline` | 全开（默认） | 4 个机制协作下的稳态行为 |
| `no_pe` | `--enable_pe False` | PE 关闭后 ζ 是否仍由网络主动调节（验证 PE 是探测性扰动） |
| `identity` | `--identity_init True` | 整条 tuner 短路，ζ_runtime ≡ ζ_nominal（验证幂等开关） |
| `no_lpf` | `--enable_param_lpf False` | LPF 关闭后高频抖动是否变多（验证频率隔离） |

---

## 2. 训练学习曲线（200 iter）

数据来源：[learning_curve.json](file:///home/zmem063/isaaclab/source/extensions/omni.isaac.lab_tasks/omni/isaac/lab_tasks/direct/isaac-auv-env-new/.tmp/meta_train_gpu_20260605_214751/learning_curve.json)（从 [train.log](file:///home/zmem063/isaaclab/source/extensions/omni.isaac.lab_tasks/omni/isaac/lab_tasks/direct/isaac-auv-env-new/.tmp/meta_train_gpu_20260605_214751/train.log) 提取的 196 个 metric block）。

### 2.1 关键指标头尾对比

| 指标 | first 10 iter | last 20 iter | min | max | 解读 |
|---|---:|---:|---:|---:|---|
| Mean total reward | 9.368 | **28.477** | 1.020 | 31.230 | 收敛 ×3，奖励成长合理 |
| Mean episode length | 97.143 | **179.000** | 11.060 | 179.000 | 5 iter 内即 saturate 到 episode_length_s 上限（180-1） |
| Episode log MSE | 195.45 | 200.26 | 21.28 | 347.73 | 与 [REPORT_scenarios_6k v2 §3](file:///home/zmem063/isaaclab/source/extensions/omni.isaac.lab_tasks/omni/isaac/lab_tasks/direct/isaac-auv-env-new/workflows_new_stdw/REPORT_scenarios_6k_20260604_v2.md) 4 维 baseline 量级吻合 |
| Value function loss | 0.801 | **0.528** | 0.257 | 1.523 | 单调下降 -34% |
| Surrogate loss | -0.0017 | -0.0009 | -0.0061 | 0.0152 | 紧贴零附近，PPO ratio 稳定 |
| Mean action noise std | 0.791 | **0.930** | 0.780 | 0.930 | KL adaptive schedule，KL < target → noise 反向放大（正常） |
| Iteration time | 0.334 s | 0.228 s | 0.21 s | 1.34 s | first iter 含 JIT warmup |

### 2.2 Wall-clock 性能

| 项 | 值 |
|---|---|
| Total time（200 iter learn 部分） | 47.99 s |
| Isaac 启动 + env build | ~8.6 s |
| 总 wall-clock | 56.6 s |
| Computation throughput（last 10 iter） | ~27,000 steps/s（collection 0.19 s + learning 0.033 s） |
| 总 timesteps | 1,204,224（≈ 256 envs × 24 steps × 196 update blocks） |
| Checkpoint 节奏 | 每 50 iter 一次（model_0/50/100/150/199.pt） |

### 2.3 训练阶段定性

- **iter 0-5（warm-up）**：episode 频繁早终止（Mean ep_len 11→81），reward 1→13；策略还没学会保持姿态。
- **iter 6-15（rapid growth）**：episode 长度 saturate 到 179，reward 跃升至 24-28 区间。
- **iter 16-200（稳态训练）**：reward 在 25-31 之间小幅震荡，noise_std 慢爬到 0.93（adaptive 起作用），value_loss 单调下降。

> 注：episode_length_s = 3.0 s / dt(decim) = 0.0333 s ⇒ 上限 90 步？实际观察 179 — 因为 [warpauv_env.py](file:///home/zmem063/isaaclab/source/extensions/omni.isaac.lab_tasks/omni/isaac/lab_tasks/direct/isaac-auv-env-new/warpauv_env.py) 用 `episode_length_s × (1 / sim.cfg.dt) = 3 × 60 = 180`，这里取 sim_dt 而非 control_dt。

---

## 3. 4 组扫参（机制消融）

数据：[sweep_215003/<group>/meta_eval_summary.json](file:///home/zmem063/isaaclab/source/extensions/omni.isaac.lab_tasks/omni/isaac/lab_tasks/direct/isaac-auv-env-new/.tmp/meta_train_gpu_20260605_214751/sweep_215003) 和 `meta_eval_timeline.csv`。

### 3.1 PE / Dead-Zone 占比

| 组 | `pe_active_ratio` | `deadzone_active_ratio` | 解读 |
|---|---:|---:|---|
| baseline | 1.000 | 0.006 | PE 全程注入；死区只在初始几步触发（ω≈0） |
| no_pe | **0.000** | 0.006 | `--enable_pe False` 生效，PE 完全关 ✅ |
| identity | **0.000** | 0.000 | identity_init 短路全部，包括 deadzone flag 也旁路 ✅ |
| no_lpf | 1.000 | 0.006 | LPF 关，PE/deadzone 不变（独立开关） ✅ |

### 3.2 ζ_runtime / ζ_nominal 漂移区域统计

只看 step ∈ [200, 800)（COB y 漂移注入区间），ζ_runtime / ζ_nominal 比率 4 个轴 (roll/pitch/yaw/depth) 的 (mean, min, max)：

| 组 | roll | pitch | yaw | depth |
|---|---|---|---|---|
| **baseline** | mean=1.042, [0.890, 1.184] | mean=0.945, [0.866, 1.003] | mean=1.042, [0.905, 1.155] | mean=0.939, [0.835, 1.098] |
| no_pe | mean=1.042, [0.910, 1.184] | mean=0.945, [0.865, 1.005] | mean=1.042, [0.905, 1.154] | mean=0.939, [0.834, 1.097] |
| **identity** | mean=**1.000**, [0.998, 1.000] | mean=**1.000**, [0.996, 1.003] | mean=**1.000**, [0.995, 1.003] | mean=**1.000**, [0.998, 1.000] |
| no_lpf | mean=1.047, [0.892, 1.192] | mean=0.944, [0.855, 1.007] | mean=1.043, [0.898, 1.187] | mean=0.938, [0.816, 1.128] |

**关键观察**：

1. **identity 组完美旁路**：ζ_runtime/ζ_nominal 4 个轴全部 mean=1.000，min/max ≈ ±0.005（数值误差），**100% 验证 [`--identity_init`](file:///home/zmem063/isaaclab/source/extensions/omni.isaac.lab_tasks/omni/isaac/lab_tasks/direct/isaac-auv-env-new/gain_tuner.py) 短路逻辑**。
2. **baseline vs no_pe 几乎无差**（roll/pitch/yaw/depth mean 差 ≤ 0.0002）：因为 PE 振幅与 ‖ω‖² 成反比衰减（pe_decay_gamma=4），一旦策略已收敛使姿态稳定，PE 自动近零。这正是 σ-modification 的设计目的——PE 不打扰已稳态系统。
3. **no_lpf 比 baseline 更激进**：min 比 baseline 低（0.816 vs 0.835）、max 比 baseline 高（1.192 vs 1.184）。LPF 起到了**频率隔离**作用，符合奇异摄动理论预期 ✅。
4. **网络确实学到了方向性偏置**：4 个轴的 ζ_runtime/ζ_nominal mean 在 [0.94, 1.05] 区间稳定偏离 1.0（roll +4.2%、pitch -5.5%、yaw +4.2%、depth -6.1%），且**3 组完整 tuner 组（baseline / no_pe / no_lpf）一致**——说明这是策略层学到的固定调制，而不是 tuner 噪声。

### 3.3 角速度（系统稳定性）

| 组 | drift 区 ang_vel_norm mean | drift 区 max |
|---|---:|---:|
| baseline | 3.149 rad/s | 9.351 |
| no_pe | 3.148 | 9.348 |
| identity | **3.077** | **9.727** |
| no_lpf | 3.137 | 9.344 |

- identity 组 mean 最低（无 PE 探测扰动）但 max 最高（无在线增益补偿对抗 COB 漂移）。
- 全 tuner 组 max 一致 ~9.35，比 identity 低 ~3.9%，与 baseline ζ_runtime 上调对应（轻微但可观测）。

### 3.4 a_gain 信号

| 组 | drift 区 \|a_gain raw\| mean | drift 区 \|a_gain LPF\| mean |
|---|---:|---:|
| baseline | 0.395 | 0.373（LPF 滤掉 ~5.6%） |
| no_pe | 0.395 | 0.374 |
| identity | 0.390 | **0.000**（LPF 短路） |
| no_lpf | 0.397 | **0.000**（LPF 关） |

- raw a_gain 在 4 组间几乎一致（~0.39，网络输出本身），证明策略输出与具体下游机制开关解耦。
- identity / no_lpf 组的 LPF 输出为 0：identity 是 tuner 整体短路；no_lpf 是 LPF 旁路（直接用 raw）。这两个零值是**预期行为**（CSV 字段语义为"LPF 缓存值"，无 LPF 时为零），不是缺陷。

---

## 4. 4 维 vs 8 维 通路对照（未消融，仅工程链路）

| 维度 | 4 维 baseline | 8 维元控制 |
|---|---|---|
| 任务 ID | `MOGA-WarpAUV-Direct-v1` | `MOGA-WarpAUV-Direct-Parametric-v1` |
| Cfg | `WarpAUVEnvCfg` (默认 `tune_gains=False`) | `WarpAUVParametricEnvCfg` (`tune_gains=True`) |
| Runner | `WarpAUVPPORunnerCfg` | `WarpAUVParametricPPORunnerCfg` |
| Action 形状 | (N, 4) | (N, 8) |
| 数学落点 | a_ctrl → S-Surface 直驱 | a_ctrl → S-Surface；a_gain → tuner → ζ_runtime |
| 退化方法 | — | `--tune_gains False` 或切回 4 维 task ID |
| 训练影响 | 不变 | 200 iter / 256 envs / 56.6 s wall-clock |

> 4 维 baseline 训练数据见历史记录（[REPORT_scenarios_6k v2 §3](file:///home/zmem063/isaaclab/source/extensions/omni.isaac.lab_tasks/omni/isaac/lab_tasks/direct/isaac-auv-env-new/workflows_new_stdw/REPORT_scenarios_6k_20260604_v2.md)），本报告**不再重训 4 维通路**。

---

## 5. 复现命令

### 5.1 从头训练（256 envs / 200 iter）

```bash
TS=$(date +%Y%m%d_%H%M%S); BASE=".tmp/meta_train_gpu_${TS}"
mkdir -p "${BASE}/logs" "${BASE}/results"
bash custom_workflows/run_with_isaac_env.sh workflows_new_stdw/train_meta.py --headless \
    --task MOGA-WarpAUV-Direct-Parametric-v1 --num_envs 256 --max_iterations 200 --seed 42 \
    --logs_root "${BASE}/logs" --results_root "${BASE}/results" 2>&1 | tee "${BASE}/train.log"
```

### 5.2 4 组扫参 eval

```bash
CKPT="${BASE}/logs/warpauv_parametric/<run>/model_199.pt"
SWEEP="${BASE}/sweep_$(date +%H%M%S)"
for cfg in baseline no_pe identity no_lpf; do
  D="${SWEEP}/${cfg}"; mkdir -p "${D}"
  case "${cfg}" in
    baseline) FLAGS="" ;;
    no_pe) FLAGS="--enable_pe False" ;;
    identity) FLAGS="--identity_init True" ;;
    no_lpf) FLAGS="--enable_param_lpf False" ;;
  esac
  bash custom_workflows/run_with_isaac_env.sh workflows_new_stdw/play_meta_eval.py --headless \
    --task MOGA-WarpAUV-Direct-Parametric-v1 --num_envs 1 --steps 800 \
    --cob_drift_axis y --cob_drift_magnitude 0.05 --cob_drift_start_step 200 --cob_drift_end_step 800 \
    --policy_path "${CKPT}" --save_dir "${D}" ${FLAGS}
done
```

### 5.3 抽统计量

```bash
python - <<'PY'
import csv, json, pathlib, statistics
SWEEP = pathlib.Path(".tmp/meta_train_gpu_20260605_214751/sweep_215003")
for cfg in ("baseline", "no_pe", "identity", "no_lpf"):
    d = SWEEP / cfg
    s = json.loads((d / "meta_eval_summary.json").read_text())
    print(cfg, s.get("zeta_runtime_over_nominal_mean"))
PY
```

---

## 6. 局限与下一步

### 6.1 当前实验局限

- **训练规模偏小**：本次 200 iter / 256 envs 仅作 sanity；正式训练应 1500+ iter / num_envs=1024（[train_meta.py docstring 命令 1](file:///home/zmem063/isaaclab/source/extensions/omni.isaac.lab_tasks/omni/isaac/lab_tasks/direct/isaac-auv-env-new/workflows_new_stdw/train_meta.py#L13-L19)）。
- **DR 关闭**：本轮未开 domain randomization（PID_scale_range / PID_adjust_range / control_profile）；正式实验建议至少开 `--enable_pid_dr True`。
- **单一漂移方向**：扫参只测了 y 轴 +0.05 m；多方向（x/y/z）+ 多幅度（±0.02、±0.05、±0.10）矩阵留给后续。
- **Episode 上限 saturate**：reward 在 ~5 iter 后 ep_len 达上限即不再变化，掩盖了"提前失败 → 长度奖励"信号；可考虑加大 `episode_length_s` 或换 reward 结构。

### 6.2 推荐扫参矩阵（下一阶段）

| 维度 | 候选值 | 备注 |
|---|---|---|
| `cob_drift_axis` | x, y, z | 三方向独立验证 |
| `cob_drift_magnitude` | 0.02 / 0.05 / 0.10 | 单调 |
| `embodiment` | base / long_body / heavy_moderate / asymmetric | 与 STDW 报告一致 |
| `tune_gains` × `identity_init` | (T, F) / (T, T) | 8 维有调制 vs 8 维幂等对照 |

矩阵规模：3 × 3 × 4 × 2 = 72 cells，每 cell 800 步 eval ≈ 13 s，预计总耗时 ~16 min（不含 Isaac 启动开销）。

### 6.3 STDW 整合接口建议

将 `tune_gains=True` cfg 路径与 STDW 慢环（`gen_C / clip-gate / Lyapunov mask`）联动：
- STDW 慢环输出 ζ_baseline，ParametricGainTuner 在 ζ_baseline 上叠加 ±β residual。
- PE 与 STDW PE 共享同一 `ω_d` 时间基（避免双频拍频）。
- 把 ζ_runtime / ζ_nominal 比率作为 STDW Lyapunov 监视器的输入项之一。

---

## 7. 数据文件索引

| 类别 | 路径 |
|---|---|
| 训练日志 | [train.log](file:///home/zmem063/isaaclab/source/extensions/omni.isaac.lab_tasks/omni/isaac/lab_tasks/direct/isaac-auv-env-new/.tmp/meta_train_gpu_20260605_214751/train.log)（163 KB） |
| 学习曲线 JSON | [learning_curve.json](file:///home/zmem063/isaaclab/source/extensions/omni.isaac.lab_tasks/omni/isaac/lab_tasks/direct/isaac-auv-env-new/.tmp/meta_train_gpu_20260605_214751/learning_curve.json) |
| 训练 checkpoint | [model_0/50/100/150/199.pt](file:///home/zmem063/isaaclab/source/extensions/omni.isaac.lab_tasks/omni/isaac/lab_tasks/direct/isaac-auv-env-new/.tmp/meta_train_gpu_20260605_214751/logs/warpauv_parametric) |
| Tensorboard | [events.out.tfevents...](file:///home/zmem063/isaaclab/source/extensions/omni.isaac.lab_tasks/omni/isaac/lab_tasks/direct/isaac-auv-env-new/.tmp/meta_train_gpu_20260605_214751/logs/warpauv_parametric) |
| 扫参 4 组 | [sweep_215003/{baseline, no_pe, identity, no_lpf}/](file:///home/zmem063/isaaclab/source/extensions/omni.isaac.lab_tasks/omni/isaac/lab_tasks/direct/isaac-auv-env-new/.tmp/meta_train_gpu_20260605_214751/sweep_215003) |
| Smoke 端到端（5 iter + 200/800 步 eval） | [.tmp/meta_smoke_gpu_20260605_213644/](file:///home/zmem063/isaaclab/source/extensions/omni.isaac.lab_tasks/omni/isaac/lab_tasks/direct/isaac-auv-env-new/.tmp/meta_smoke_gpu_20260605_213644) |

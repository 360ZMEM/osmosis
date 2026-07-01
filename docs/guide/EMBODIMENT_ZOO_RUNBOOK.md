# 多 Embodiment 工程 Runbook

本文档面向实施与实验执行，描述 `uuv6` / `uuv4` / `uuv6_angled` / `uuv4_angled` 的设计契约、配置文件、smoke test 顺序、全量训练命令与验收标准。概念总览见 [`EMBODIMENT_ZOO.md`](EMBODIMENT_ZOO.md)；论文叙述见 [`PAPER_ADDENDUM_multi_embodiment_20260701.md`](../principles/PAPER_ADDENDUM_multi_embodiment_20260701.md)。

---

## 1. 目标与执行顺序

目标不是先证明 RL 已经泛化，而是先证明底层 S-Surface + TAM 能在四类形态上启动并产生基本可控行为。

执行顺序固定：

1. 文档与配置检查：确认四类 embodiment 的物理参数、TAM、任务难度、ckpt 策略都可追踪。
2. smoke test：每个 embodiment 先短步运行，确认环境能创建、推力维度正确、无 NaN/reset storm/shape mismatch。
3. 全量训练：smoke 通过后才启动训练；训练入口必须带 `--embodiment`，否则只会训练 base。
4. eval 回填：训练完成后回填实测 MSE、reset、surface breach、thruster saturation、ckpt 适用性。

---

## 2. 机型设计契约

| embodiment | 推进器 | 分配模式 | 可控通道 | 任务约束 | 训练风险 |
|---|---:|---|---|---|---|
| `uuv6` | 6 | `pinv` | roll/pitch/yaw/depth | roll/pitch ±π，yaw 小幅 | 低；全驱布局应可复用 SO(3) 底层 |
| `uuv4` | 4 | `wls` | roll/pitch/depth | yaw 幅必须为 0 | 中；欠驱动，无 yaw 权威 |
| `uuv6_angled` | 6 | `pinv` | roll/pitch/yaw/depth | roll/pitch ±π，yaw 小幅 | 中；非正交耦合依赖 B+ |
| `uuv4_angled` | 4 | `wls` | roll/pitch/depth | yaw 幅必须为 0 | 高；欠驱动 + 非正交 |

密度契约：新增机型均满足 `mass / volume ~= 990 kg/m^3 < water_rho=997 kg/m^3`，即微正浮力。它服务于近边界效应假设：车辆完全浸没时可维持深度，倒置/扰动后有自然回浮趋势，但不应破面。

默认行为契约：`base` / `asymmetric` / `long_body` / `heavy_*` 继续走旧 8 推硬编码混合块；四个新机型才走 config TAM 路径。默认 `--embodiment base`，不改变旧实验。

---

## 3. TAM 与控制语义

`thrust_allocation.py` 定义统一 TAM：

- body wrench 顺序：`[Fx, Fy, Fz, Tx, Ty, Tz]`
- 控制通道：`[roll, pitch, yaw, depth]`
- 映射：`roll -> Tx`，`pitch -> Ty`，`yaw -> Tz`，`depth -> Fz`
- 单推进器列：`B[:, i] = [R(q_i) xhat ; r_i x R(q_i) xhat]`

分配策略：

- `pinv`：全驱或近全驱布局，使用最小二范数伪逆。
- `wls`：欠驱动布局，`controllable_dofs=["heave","roll","pitch"]` 将 yaw 行权重置 0，避免为不可控 yaw 产生伪分配。

重要符号：

- 旧硬编码混合块的 roll/yaw 净力矩与控制通道反号，依赖 `geo_channel_sign=[-1, +1, -1]` 补偿。
- config TAM 路径的目标 wrench 是正号语义，由 B+ 直接解耦；不要再额外乘旧混合块符号。

---

## 4. 配置文件矩阵

| 文件 | 适用 embodiment | 关键差异 |
|---|---|---|
| `workflows/configs/embodiment_uuv6.yaml` | `uuv6` | yaw 幅 `0.3`，频率较低 |
| `workflows/configs/embodiment_uuv4.yaml` | `uuv4` | yaw 幅 `0.0`，欠驱动减载 |
| `workflows/configs/embodiment_uuv6_angled.yaml` | `uuv6_angled` | yaw 幅 `0.3`，非正交减频 |
| `workflows/configs/embodiment_uuv4_angled.yaml` | `uuv4_angled` | yaw 幅 `0.0`，最低频欠驱动压测 |
| `workflows/configs/embodiment_submerge_flip360.yaml` | `base` 或任意机型 | 只验证下沉后 flip360 协议 |

所有配置均开启：

- `attitude_error_mode: so3`
- `reference_mode: flip360_sine`
- `submerge_phase_enable: true`
- `surface_guard_enable: true`
- `starting_depth: 1.5`
- `z_surface_guard: 3.0`
- `surface_margin: 0.15`

安全关系式：

```text
starting_depth < z_surface_guard - vehicle_height/2 - surface_margin
1.5 < 3.0 - 0.15 - 0.15 = 2.7
```

---

## 5. Smoke Test

### 5.1 静态 smoke

```bash
python3 -m py_compile \
  thrust_allocation.py \
  easyuuv_env.py \
  workflows/play_stdw_adapt.py \
  workflows/train_meta.py \
  workflows/sweep_stdw_safety_pressure.py

python3 workflows/tools/test_thrust_allocation.py

python3 workflows/sweep_stdw_safety_pressure.py \
  --profile embodiment_zoo \
  --dry_run \
  --total_steps 10
```

通过条件：

- `py_compile` 无语法错误。
- TAM 单测全部 `[OK]`。
- `embodiment_zoo --dry_run` 生成 9 个 case，命令中四个新机型均出现。

### 5.2 Isaac 短步 smoke

先跑最小步数，确认环境创建、动作维度、推进器维度、下沉守卫都不报错：

```bash
python3 workflows/sweep_stdw_safety_pressure.py \
  --profile embodiment_zoo \
  --run \
  --limit 9 \
  --total_steps 120 \
  --results_root .results/embodiment_zoo_smoke_$(date +%Y%m%d_%H%M%S)
```

通过条件：

- 9 个 case 进程 returncode 为 0。
- `summary.json` 无 nonfinite。
- 没有 `shape mismatch` / `index out of range` / `apply_embodiment_config failed`。
- uuv4/uuv4_angled 不应出现 yaw 任务误设；其 yaml 中 `ref_sine_amp[2] == 0.0`。
- 若发生 surface guard reset，需要检查 `root_pos_w[:,2]` 是否接近 `z_surface_guard - surface_margin`；短步 smoke 不要求性能，只要求不破坏运行。

---

## 6. 全量训练策略

训练入口为 `workflows/train_meta.py`。必须显式传 `--embodiment`，或在 workflow YAML 的 `train.embodiment` 中设置；否则默认训练 base。

### 6.1 推荐第一轮：从现有 SO(3) ckpt fine-tune

先训练四个新机型，每个单独 run，避免一个共享策略同时承受欠驱动/非正交差异。推荐从 `model_2398.pt` resume，`resume_load_optimizer=False`，只加载 policy/normalizer：

```bash
for emb in uuv6 uuv4 uuv6_angled uuv4_angled; do
  cfg="workflows/configs/embodiment_${emb}.yaml"
  bash custom_workflows/run_with_isaac_env.sh workflows/train_meta.py \
    --headless \
    --task EasyUUV-Direct-Parametric-v1 \
    --num_envs 512 \
    --max_iterations 300 \
    --workflow_config "${cfg}" \
    --embodiment "${emb}" \
    --experiment_name easyuuv_parametric \
    --run_name "embodiment_${emb}_ft300" \
    --resume True \
    --load_run 2026-06-08_13-48-14_stage2 \
    --checkpoint model_2398.pt \
    --resume_load_optimizer False \
    --meta_stage 0
done
```

### 6.2 真正全量：提高 env 数和迭代数

smoke + ft300 通过后，再扩大到全量：

```bash
for emb in uuv6 uuv4 uuv6_angled uuv4_angled; do
  cfg="workflows/configs/embodiment_${emb}.yaml"
  bash custom_workflows/run_with_isaac_env.sh workflows/train_meta.py \
    --headless \
    --task EasyUUV-Direct-Parametric-v1 \
    --num_envs 1024 \
    --max_iterations 1500 \
    --workflow_config "${cfg}" \
    --embodiment "${emb}" \
    --experiment_name easyuuv_parametric \
    --run_name "embodiment_${emb}_full1500" \
    --resume True \
    --load_run 2026-06-08_13-48-14_stage2 \
    --checkpoint model_2398.pt \
    --resume_load_optimizer False \
    --meta_stage 0
done
```

### 6.3 训练期间检查

每个 run 关注：

- `train.log` 是否出现 `nan` / `inf` / `shape mismatch`。
- `compact_log.jsonl` 的 `log_mse` 是否从混沌区下降。
- 欠驱动机型不以 yaw MSE 作为主要失败标准；重点看 roll/pitch/depth 与 reset。
- uuv6/uuv6_angled 若 yaw 明显失控，优先检查 TAM layout 和水平推进器 yaw 力偶。
- 若一开始就大量 reset，优先降低 `ref_sine_freq`，其次增加 `submerge_hold_steps`，最后再考虑减小 `ref_sine_amp`。

---

## 7. 训练后 eval 与回填

训练后使用对应 ckpt 跑：

```bash
python3 workflows/sweep_stdw_safety_pressure.py \
  --profile embodiment_zoo \
  --run \
  --total_steps 3000 \
  --policy_path logs/rsl_rl/easyuuv_parametric/<RUN_DIR>/<CKPT>.pt \
  --results_root .results/embodiment_zoo_eval_<tag>
```

回填到文档：

- `docs/guide/EMBODIMENT_ZOO.md`：把 `【TODO：实测】` 补成实测 MSE/reset/是否可复用 ckpt。
- `docs/principles/PAPER_ADDENDUM_multi_embodiment_20260701.md`：补实验表与论文引用。
- `docs/engineering/CHANGELOG.md`：补训练 run id、ckpt、结论。

---

## 8. 失败处理表

| 症状 | 最可能原因 | 处理 |
|---|---|---|
| `motorValues` 与 efficiency shape mismatch | N 推 buffer 未随 `_num_thrusters` 重建 | 检查 `apply_embodiment_config` 是否在训练路径执行 |
| `thruster index out of range` | fault/angle-shift 仍引用 8 推索引 | smoke 阶段关闭 fault/angle-shift；或为新 N 推重配索引 |
| uuv4 yaw 误差很大 | 欠驱动本来无 yaw | 确认 yaml yaw 幅为 0，评估时不把 yaw 当失败主指标 |
| 一开始破面 reset | 下沉深度或 surface margin 不安全 | 降低 `starting_depth`，增加 `submerge_hold_steps` |
| flip360 倒置区卡死 | 恢复力矩 > 控制预算 | 降低频率，启用容差整形或分课程训练 |
| `train_meta.py` 训练出来仍像 base | 忘了传 `--embodiment` | 检查 `params/env.yaml` 和 stdout 中的 `applied embodiment` |

---

## 9. 当前状态

已完成：

- 四个新 embodiment 注册表。
- config TAM 分配路径。
- 下沉后 flip360 与水面守卫配置。
- 5 个 workflow YAML。
- `play_stdw_adapt.py` 和 `train_meta.py` 的 `--embodiment` 接线。
- `sweep_stdw_safety_pressure.py --profile embodiment_zoo`。

下一步：

1. 跑 §5 smoke test。
2. smoke 通过后启动 §6.1 第一轮 ft300。
3. ft300 稳定后再启动 §6.2 full1500。

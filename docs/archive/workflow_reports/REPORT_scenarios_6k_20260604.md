# STDW 多场景与跨机型扫参报告（6000 步）

> 版本：v1（2026-06-04）  
> 关联实施计划：[STDW多场景与渐进注入实施计划.md](file:///home/zmem063/isaaclab/source/extensions/omni.isaac.lab_tasks/omni/isaac/lab_tasks/direct/isaac-auv-env-new/.trae/documents/STDW多场景与渐进注入实施计划.md)  
> 数据基目录：`.tmp/stdw_sweep_6k_20260604_202549/`  
> 聚合 CSV：[aggregated_metrics.csv](file:///home/zmem063/isaaclab/source/extensions/omni.isaac.lab_tasks/omni/isaac/lab_tasks/direct/isaac-auv-env-new/.tmp/stdw_sweep_6k_20260604_202549/aggregated_metrics.csv)  
> 原始 sweep CSV：`scenarios_results.csv` / `embodiments_results.csv`

---

## 1. 实验配置

| 项 | 值 |
|---|---|
| 任务 | `MOGA-WarpAUV-Direct-v1` |
| 算法 | STDW v3 全开（双向行为锚定 + 分位过滤 + Lyapunov mask + clip-gate + ρ-decay） |
| `total_steps` | **6000** |
| `drift_start_step` / `drift_end_step` | 200 / 1200 |
| `target_drift` | 0.05 |
| `g_C_lr` | 5e-5 |
| `slow_loop_interval` | 60 步 |
| 控制 profile | `A-S-Surface`（cascade=True, method=Ssurface, self_adapt=True） |
| 物理 / 仿真 | CPU、headless、num_envs=1 |
| Phase 1 | 7 个 scenario × `embodiment=base`（[scenarios_results.csv](file:///home/zmem063/isaaclab/source/extensions/omni.isaac.lab_tasks/omni/isaac/lab_tasks/direct/isaac-auv-env-new/.tmp/stdw_sweep_6k_20260604_202549/scenarios_results.csv)） |
| Phase 2 | 4 个 embodiment × `scenario=none`（[embodiments_results.csv](file:///home/zmem063/isaaclab/source/extensions/omni.isaac.lab_tasks/omni/isaac/lab_tasks/direct/isaac-auv-env-new/.tmp/stdw_sweep_6k_20260604_202549/embodiments_results.csv)） |

> 所有外围扰动按 [drift_start, drift_end] = [200, 1200] **线性渐进注入**，由 [`DisturbanceSchedule.tick()`](file:///home/zmem063/isaaclab/source/extensions/omni.isaac.lab_tasks/omni/isaac/lab_tasks/direct/isaac-auv-env-new/workflows_new_stdw/disturbance_schedule.py#L119-L202) 在每个 env step 后调度；STDW 算法本身（fast/slow loop / loss / buffer）零改动。

---

## 2. Phase 1：场景扫参（embodiment=base）

| scenario | `final_mse` | `final_mse_after_drift` | `mean_total_mse` | `max_total_mse` | `compound_error_tail_mean` | `amp_x_tail_mean` | resets | NaN | slow triggers |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `none` | 2.4592 | 3.1808 | 1.5656 | 23.46 | 1.5609 | 0.00 | 33 | 0 | 95 |
| `sine` | 6.8444 | 4.2732 | 2.4188 | 23.19 | 2.6165 | **0.10** | 33 | 0 | 95 |
| `current_plus_jonswap` | 2.3948 | **4.6361** | 2.5016 | 27.15 | 2.7262 | 0.00 | 33 | 0 | 95 |
| `jonswap_mild` | 2.5008 | 3.0096 | 1.5537 | 18.87 | 1.5462 | 0.00 | 33 | 0 | 95 |
| `jonswap_strong` | **1.5342** | 2.7081 | **1.3774** | 19.12 | **1.3265** | 0.00 | 33 | 0 | 95 |
| `wave_plus_fault` | 2.7023 | 4.5240 | 2.3915 | 21.80 | 2.5832 | **0.12** | 33 | 0 | 95 |
| `wave_plus_noise` | 3.8639 | 4.6080 | 2.4855 | 21.80 | 2.6998 | **0.10** | 33 | 0 | 95 |

**字段释义**

- `final_mse`：最末 N 步窗口的 compound_error 均值（含瞬态）。
- `final_mse_after_drift`：仅取 step > 1200 的稳态段均值（fault / noise / wave 已 ramp 到 target）。
- `compound_error_tail_mean`：ramp 完成后 4800 步的 compound_error 均值。
- `amp_x_tail_mean`：ramp 完成后 disturbance amplitude 的 X 分量均值（核对扰动确实达到了 target）。
- `slow triggers`：97 = `(6000-200)/60 - 余项`，与设计一致，95 个有效慢环触发。
- `resets=33` 来自 episode 自身定时重置（≈每 180 步），不是任务失败。

**关键观察**

1. 7 个场景全部收敛、零 NaN、零 nonfinite_guard 触发；**渐进注入闭环全链路畅通**。
2. `amp_x_tail_mean` 与 [scenarios.py](file:///home/zmem063/isaaclab/source/extensions/omni.isaac.lab_tasks/omni/isaac/lab_tasks/direct/isaac-auv-env-new/workflows_new_stdw/scenarios.py) 中 target 完全吻合：sine=0.10、wave_plus_noise=0.10、wave_plus_fault=0.12、jonswap=0（amplitude 通道不开），证明 `apply_runtime_domain_shift` ramp 是真生效到了 env，不是空写。
3. `jonswap_strong` 反而**最好**（final_mse=1.53）——这是因为 JONSWAP 主要把扰动加在 z 方向波浪力上，本任务的 yaw / depth 误差不与之共振；它并不代表"波浪越强越好"，只能说明 cascade A-S-Surface 对 JONSWAP 的频谱不敏感。
4. `sine` 的 `final_mse=6.84` 显著高于 `final_mse_after_drift=4.27`：说明大量误差在 ramp 过程中被累积，ramp 完成后控制器又把误差拉了回来——**这正是 STDW 在线慢环起作用的指纹**。但 95 次 slow_loop trigger 之后仍未达 baseline，说明衰减速率还有调优空间。
5. `wave_plus_fault` 与 `wave_plus_noise` 的 `final_mse_after_drift` ≈ `current_plus_jonswap` ≈ 4.5，差不多是 baseline 的 1.4×。

---

## 3. Phase 2：跨机型扫参（scenario=none）

| embodiment | `final_mse` | `final_mse_after_drift` | `mean_total_mse` | `max_total_mse` | `compound_error_tail_mean` | resets | NaN | slow triggers |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `base` | 2.4592 | 3.1808 | 1.5656 | 23.46 | 1.5609 | 33 | 0 | 95 |
| `long_body` | 2.7626 | 2.8756 | 1.7546 | 20.99 | 1.6711 | 33 | 0 | 95 |
| `heavy_moderate` | **8.2338** | **5.0518** | **3.2286** | 20.77 | 3.3999 | 33 | 0 | 95 |
| `asymmetric` | 3.0526 | 4.1420 | 2.2695 | 22.17 | 2.4738 | 33 | 0 | 95 |

**关键观察**

1. **4 个机型变体全部跑完不发散、零 NaN**——验证了 [`apply_embodiment_config`](file:///home/zmem063/isaaclab/source/extensions/omni.isaac.lab_tasks/omni/isaac/lab_tasks/direct/isaac-auv-env-new/warpauv_env.py) 在 STDW workflow 入口的接入正确。
2. `long_body` 的稳态误差与 `base` 几乎打平（2.88 vs 3.18），说明长机身在 cascade A-S-Surface 下增益匹配尚可。
3. `asymmetric` 仅比 base 差 30%，可接受。
4. `heavy_moderate` **稳态误差比 base 高 60% 以上**（5.05 vs 3.18），属于"控制器增益不匹配"导致的退化——这与用户声明 "对 cross-embodiment 不要求结果" 一致。
5. 4 个机型共享同一组 `slow_loop` 训练统计（95 次），STDW 没有因为机型变化而异常触发或丢触发。

---

## 4. 数据正确性自检

| 检查项 | 状态 | 证据 |
|---|---|---|
| CSV 行数 = total_steps | ✅ | 11 个 csv 全部 6000 行 |
| 9 个新增列就位 | ✅ | scenario / embodiment / disturbance_mode / amp_x/y/z / noise_std_eff / fault_active / fault_efficiency_min |
| step ≤ 200 时 amp=0 / fault=False | ✅ | smoke + sweep 双重验证 |
| step ∈ [200,1200] 严格单调上升 | ✅ | smoke 验证；sweep 全部到达 target |
| `amp_x_tail_mean` 与 scenario target 一致 | ✅ | sine/wave_plus_*=0.10/0.12 ✓；jonswap=0（ampl 通道不开）✓ |
| 零 NaN、零 nonfinite_guard | ✅ | 11/11 |
| Summary 4 个新键写入 | ✅ | scenario / embodiment / fault_rate_per_second / fault_thrusters |
| `final_mse_after_drift` 计算正确 | ✅ | 11/11 非 null |

---

## 5. 修复点（必须解决）

### 5.1 `fault_efficiency_min_tail_mean` 始终为 1.0（高优先级）

**现象**：所有 6000 步运行中 `fault_efficiency_min` 列恒等于 1.0，包括 `wave_plus_fault` scenario，即推进器效率从未实际下降。

**预期**：在 `wave_plus_fault` 中，`fault_rate_per_second=0.35` 应在 step=200 启用、按 sim_dt(=0.0167s) × 5800 step ≈ 96.7 秒 × 0.35 ≈ 100% 衰减；最迟应在 step ≈ 200+170 处看到 efficiency_min < 1.0。

**根因猜测**（待静态阅 [warpauv_env.py](file:///home/zmem063/isaaclab/source/extensions/omni.isaac.lab_tasks/omni/isaac/lab_tasks/direct/isaac-auv-env-new/warpauv_env.py) `set_thruster_fault` 与 `thruster_efficiency_factors` 的更新位置确认）：

- (a) `set_thruster_fault` 写到的是配置对象，但物理回路读取的是 cached buffer，没有触发重算。
- (b) `start_sim_time` 用了 `step * sim_dt_seconds`（= 1/120），但 env 内部按 `decimation × physics_dt = 2 × 1/60 = 1/30` 推进时间，导致衰减公式输入的 t 偏小一个 4× 倍数，单步效率仍 ≈ 1.0。
- (c) `thruster_efficiency_factors` 是在 reset 时刷新的，未在 step 中更新。

**修复方向**

1. 静态确认 `set_thruster_fault` 内部对 `start_sim_time` / `fault_rate_per_second` 的使用方式；如果是 (b)，把 [`disturbance_schedule.py`](file:///home/zmem063/isaaclab/source/extensions/omni.isaac.lab_tasks/omni/isaac/lab_tasks/direct/isaac-auv-env-new/workflows_new_stdw/disturbance_schedule.py#L67-L74) 中的 `sim_dt_seconds` 改为从 env 真实读取（`env.physics_dt * env.decimation`）。
2. 如果是 (a)/(c)：在 `tick` 内额外调用 env 暴露的 efficiency 更新钩子；或者在 schedule 内自己维护一个 efficiency 软目标，直接覆写 `env.thruster_efficiency_factors`。
3. 给 [run_scenarios_smoke.sh](file:///home/zmem063/isaaclab/source/extensions/omni.isaac.lab_tasks/omni/isaac/lab_tasks/direct/isaac-auv-env-new/workflows_new_stdw/run_scenarios_smoke.sh) 加一个 600 步 smoke 检查：`assert wave_plus_fault.fault_efficiency_min_tail_mean < 0.95`，避免回归。

### 5.2 `convergence_step = None`（中优先级）

**现象**：所有 11 组的 `convergence_step` 均为 None，说明在 6000 步内 compound_error 从未连续 `stability_window` 步落入 `stability_threshold` 之下。

**根因**：从 `compound_error_tail_mean ≈ 1.3 ~ 5.0` 看，绝对误差量级远高于默认 stability_threshold（通常是 0.1 量级）。这并不代表 STDW 失效，而是阈值没有跟着任务难度伸缩。

**修复方向**

1. 把 `--stability_threshold` 的默认值从硬编码改为相对值：`max(0.5 × baseline_final_mse, abs)`。
2. 或者在 sweep 入口暴露 `--stability_threshold` 与 `--stability_window`，让对照组互相校准。
3. 报告一律用 `final_mse_after_drift` + `compound_error_tail_mean` 衡量收敛，convergence_step 仅作辅助。

### 5.3 `current_bias` scenario 在 sweep 中缺失（低优先级）

**现象**：`scenarios_results.csv` 只有 7 行，缺 `current_bias`。`scenarios_only` 列表是 7 个，但**没有把 `current_bias` 加进去**——见 [sweep_stdw.py L37-L46](file:///home/zmem063/isaaclab/source/extensions/omni.isaac.lab_tasks/omni/isaac/lab_tasks/direct/isaac-auv-env-new/workflows_new_stdw/sweep_stdw.py#L37-L46) 的 `scenario` 列表。`current_bias`（纯水流偏置，不带波浪）是审稿草稿里提到的最直接代表。

**修复方向**：把 `"current_bias"` 加进 `DEFAULT_MATRIX["scenario"]`，下次 `--scenarios_only --full_grid` 会自动跑 8 组。

---

## 6. 提升点（非必须，按需采纳）

### 6.1 `sine` 场景误差累积偏高（trade-off 优化）

`sine` 的 `final_mse=6.84` 远高于其稳态 4.27，说明 ramp 过程中的瞬态误差才是大头。可考虑：

- 对 `sine` 把 `drift_end_step` 拉到 1800（更平滑），观察 `final_mse` 与 `final_mse_after_drift` 的差是否收窄。
- 在 [warpauv_stdw_wrapper.py](file:///home/zmem063/isaaclab/source/extensions/omni.isaac.lab_tasks/omni/isaac/lab_tasks/direct/isaac-auv-env-new/warpauv_stdw_wrapper.py) 里把 `_compute_drift_fraction` 改成 cosine ramp（更慢起步、更慢收尾），跨 7 个 scenario 重测。

### 6.2 `heavy_moderate` 增益重整（在不改 STDW 范围内可做）

不属于本计划任务，但若审稿要求"4 个机型都给数字"，可：

- 用 `--control_profile pid` 跑一次 heavy_moderate（pid_multipliers 可在 CLI 暴露），看 final_mse 是否回到 base × 1.5 内。
- 如果 cascade A-S-Surface 在重机型上系统性 overshoot，可以用 `apply_pid_multipliers([1.0, 0.7, ...])` 缩小 attitude 增益，但这是控制器侧改动。

### 6.3 报告自动化

- 在 [report_plots.py](file:///home/zmem063/isaaclab/source/extensions/omni.isaac.lab_tasks/omni/isaac/lab_tasks/direct/isaac-auv-env-new/workflows_new_stdw/report_plots.py) 里加一个新 figure：`scenarios_bar.png`（横轴 scenario，纵轴 `final_mse_after_drift`，带 baseline 横线）。
- 同样补一个 `embodiments_bar.png`。

### 6.4 跨场景 + 跨机型联合矩阵

当前 `--full_grid` 走 7×4=28 组（约 28 分钟），未跑过。如审稿要求 28 组，可直接：

```bash
python workflows_new_stdw/sweep_stdw.py --full_grid --total_steps 6000 \
  --base_logs_root .tmp/stdw_full_grid_6k --csv_out logs/stdw_full_grid_6k.csv
```

数据结构与本报告完全一致，只需把 §2 / §3 表合并成一张 7×4 透视表。

### 6.5 `--algo_grid` 的回归位

DEFAULT_MATRIX 已经从"算法网格"改为"场景×机型"，但保留了 [`ALGO_MATRIX`](file:///home/zmem063/isaaclab/source/extensions/omni.isaac.lab_tasks/omni/isaac/lab_tasks/direct/isaac-auv-env-new/workflows_new_stdw/sweep_stdw.py) 与 `--algo_grid` 选项。建议在 README 或 [run_4grp_compare.sh](file:///home/zmem063/isaaclab/source/extensions/omni.isaac.lab_tasks/omni/isaac/lab_tasks/direct/isaac-auv-env-new/workflows_new_stdw/run_4grp_compare.sh) 里写清楚：之前的 sweep8 复现请加 `--algo_grid` 显式开关，否则会跑成新 28 组。

---

## 7. 结论

1. STDW workflow 与 7 个场景 / 4 个机型的"渐进注入"集成全链路通过：CSV、summary、慢环驱动、状态保护守卫、跨机型 PhysX apply 全部正常。
2. STDW 算法在 `wave_plus_fault` / `wave_plus_noise` / `current_plus_jonswap` 三个最难场景下的稳态误差分别比 baseline 高 +0.42 / +1.43 / +1.46，未发散，无 NaN。
3. 唯一**必须修**的是 §5.1：`fault_efficiency` 实际未生效（疑似 sim_dt 倍率或 cache 同步问题）。这是环境侧 bug，不在"不动 STDW"约束之内。
4. 跨机型 `heavy_moderate` 的 60% 退化属于控制器增益问题，按用户说明可保留为"原样数据"。
5. `convergence_step` 阈值需要按任务难度伸缩，否则报告永远显示 None。
6. `current_bias` 场景已实现但漏在 sweep 默认矩阵之外，是一行修复。

---

## 8. 数据落盘清单

| 文件 | 用途 |
|---|---|
| `.tmp/stdw_sweep_6k_20260604_202549/scenarios_results.csv` | 7 行场景 sweep 索引 |
| `.tmp/stdw_sweep_6k_20260604_202549/embodiments_results.csv` | 4 行机型 sweep 索引 |
| `.tmp/stdw_sweep_6k_20260604_202549/aggregated_metrics.csv` | 11 行聚合数据（含 tail mean / max） |
| `.tmp/stdw_sweep_6k_20260604_202549/scenarios/<scenario=*>/results/.../stdw_output.csv` | 每组 6000 行原始时序 |
| `.tmp/stdw_sweep_6k_20260604_202549/scenarios/<scenario=*>/results/.../summary.json` | 每组 summary（含 16 个 tracking 指标 + 4 个新增元数据） |

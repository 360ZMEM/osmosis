# STDW 强验证小矩阵报告

日期：2026-06-13

## 1. 目的

本轮补充验证四类部署风险：

1. COM-COB 在线注入是否仍存在，且仍对 RL/STDW observation 不可见。
2. 单推进器效率下降，模拟 8 推进器 UUV 中某个推进器损坏。
3. 水密度从 `1.00` 降到 `0.95`。
4. 每 3-5s 触发一次 runtime torque pulse，覆盖 level `0.5/1.0`。

实验不重训，使用 `EasyUUV-Direct-Parametric-v1`、`model_2398.pt`、`matrix_wave_medium_full.yaml`、`seed=0`、每 cell `1500` step。

产物：

- `.results/exp_stdw_strong_validation_20260613/strong_validation_runs.csv`
- `.results/exp_stdw_strong_validation_20260613/strong_validation_summary.csv`
- `.results/exp_stdw_strong_validation_20260613/strong_validation_report.md`

## 2. 实现确认

- COM-COB 注入仍由 `EasyUUVStdwWrapper._apply_drift()` 每步写入 `env.unwrapped.com_to_cob_offsets`。
- A3 observation 仍为 `[goal_quat(4), depth_z(1), root_quat(4), root_ang_vel_b(3)]`，不包含 COM-COB，因此该注入对 RL/STDW 策略不可见。
- `ramp_shape=step` 表示突发注入；`linear` 表示缓慢注入。
- 推进器故障已支持 `fixed` 与 `ramp_to_target`，本轮单推进器为 `thruster=4,target_efficiency=0.5`。
- 密度通过 `water_density_scale_target=0.95` 注入。
- torque pulse 使用 env/control step 时间，smoke 中确认可在 3-5s 窗口激活。

## 3. Smoke 结果

静态验证全部通过：

- `py_compile`
- `scenarios.py --self-test`
- `disturbance_schedule.py --self-test`
- `sweep_stdw_strong_validation.py --dry_run`
- `aggregate_stdw_strong_validation.py`

补充注入 smoke 确认：

| Case | 观测字段 | 结果 |
|---|---|---:|
| COM linear/step | `drift_fraction` | max = 1.0 |
| COM linear/step | `com_to_cob_offset_x` | last = 0.0500 |
| thruster fixed | `fault_efficiency_min` | min = 0.5 |
| density step | `water_density_scale` | last = 0.95 |
| torque medium | `torque_pulse_active` | True |
| torque medium | `max(|torque_pulse_xyz|)` | 0.4186 |

## 4. 32-cell 主矩阵

所有 32 个 cell 均 `returncode=0`，`nonfinite_guard_count=0`。每个 cell 有 4 次 episode reset，符合 1500 step / 6s episode 的设置。

### 4.1 STDW on/off 差异

`delta_pct = final_mse(STDW on) / final_mse(STDW off) - 1`。负数表示 STDW 更好。

| Group | Ramp | Embodiment | STDW on | STDW off | Δ% |
|---|---|---|---:|---:|---:|
| COM-COB | linear | base | 0.0724 | 0.2267 | -68.05 |
| COM-COB | step | base | 0.2269 | 0.2270 | -0.06 |
| thruster single | linear | base | 0.0725 | 0.2268 | -68.03 |
| thruster single | step | base | 0.2273 | 0.2274 | -0.05 |
| density 0.95 | linear | base | 0.0725 | 0.2245 | -67.72 |
| density 0.95 | step | base | 0.2260 | 0.2241 | +0.82 |
| torque level 0.5 | linear | base | 0.0725 | 0.2267 | -68.03 |
| torque level 1.0 | linear | base | 0.0725 | 0.2267 | -68.02 |
| COM-COB | linear | asymmetric | 0.5277 | 0.3189 | +65.47 |
| COM-COB | step | asymmetric | 0.4509 | 0.5903 | -23.62 |
| thruster single | linear | asymmetric | 0.5413 | 0.2915 | +85.71 |
| thruster single | step | asymmetric | 0.5925 | 0.5780 | +2.51 |
| density 0.95 | linear | asymmetric | 0.3941 | 0.2640 | +49.25 |
| density 0.95 | step | asymmetric | 0.3039 | 0.5471 | -44.45 |
| torque level 0.5 | linear | asymmetric | 0.5283 | 0.3199 | +65.16 |
| torque level 1.0 | linear | asymmetric | 0.5294 | 0.3209 | +64.99 |

### 4.2 结论

1. **base 组**：缓慢注入下 STDW on 稳定优于 off，降幅约 `68%`。这说明在线 COM-COB drift + slow-loop 仍能在标称机型上提供显著收益。
2. **突发注入**：base 组 step 注入下 STDW 与 off 基本持平，说明突发 drift 对短窗适应不友好；这比 linear 更接近故障恢复边界测试。
3. **asymmetric 组**：linear 注入下 STDW on 明显劣于 off，延续此前 P2/P5 结论：默认 drift 方向在 asymmetric 上可能错误。
4. **asymmetric step**：COM-COB step 和 density step 中 STDW on 优于 off，但 thruster fixed 基本持平或略差。该结果说明突发扰动下的响应不是单调结论，需要进一步看组件和 drift 方向。
5. **推进器/密度/力矩扰动接线有效**：聚合表显示 `fault_efficiency_min=0.5`、`water_density_scale=0.95`、`torque_pulse_level=0.5/1.0`。

## 5. 按需消融诊断结果

已按默认建议追加 asymmetric linear 组消融，详见：

- [`REPORT_stdw_strong_ablation_20260613.md`](REPORT_stdw_strong_ablation_20260613.md)
- `.results/exp_stdw_strong_ablation_20260613/strong_ablation_summary.csv`

覆盖 5 个问题组 × 5 个组件关闭项，共 25 cell，全部 `returncode=0`。关键结果：

| Base case | Best ablation | final_mse | vs_full | vs_off |
|---|---|---:|---:|---:|
| cob_linear_asymmetric | no_slow_loop | 0.3189 | -39.56% | +0.00% |
| thruster_single_ramp_to_target_asymmetric | no_slow_loop | 0.2915 | -46.15% | +0.00% |
| density095_linear_asymmetric | no_quantile_filter | 0.2561 | -35.02% | -3.02% |
| torque_l0p5_asymmetric | no_slow_loop | 0.3199 | -39.45% | +0.00% |
| torque_l1p0_asymmetric | no_slow_loop | 0.3209 | -39.39% | +0.00% |

结论：asymmetric linear 的主要劣化来自 slow-loop 在错误 drift 方向上持续更新。`no_slow_loop` 在 4/5 个组中回到 STDW off 水平；`no_pseudo`、`no_lyapunov` 和 `no_trigger_gate` 基本不能修复；`no_quantile_filter` 只在 density 组有单点收益。因此默认修复仍应是 micro-probe/router 选择 drift 方向或证据不足回退 baseline，而不是删除 STDW 组件。

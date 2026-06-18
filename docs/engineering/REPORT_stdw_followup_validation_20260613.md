# STDW 追加验证报告：router/probe 与 density no-quantile 稳定性

日期：2026-06-13

## 1. 范围

本轮按默认建议追加两类小验证：

1. `asymmetric + linear + router/micro-probe`：确认正确 drift 方向或 baseline fallback 是否消除 slow-loop 劣化。
2. `density095 linear + no_quantile_filter` 复跑 seed1/seed2：确认 seed0 上的单点收益是否稳定。

为避免 micro-probe 窗口与 drift ramp 重叠，router/probe 组使用 matched timing：`drift_start_step=320, drift_end_step=1200`。因此额外补跑同 timing 的 `matched_full` 和 `matched_off` 作为严格对照。

产物：

- `.results/exp_stdw_followup_validation_20260613/followup_validation_summary.csv`
- `.results/exp_stdw_followup_validation_20260613/followup_validation_report.md`
- `.results/exp_stdw_followup_controls_20260613/followup_validation_summary.csv`
- `.results/exp_stdw_followup_controls_20260613/followup_validation_report.md`

运行规模：

- router/probe：5 组扰动 × 2 策略 = 10 cell。
- matched controls：5 组扰动 × full/off = 10 cell。
- density seeds：seed1/seed2 × full/off/no_quantile_filter = 6 cell。
- 总计 26 cell，全部 `returncode=0`。

## 2. Router / Micro-Probe 结果

`router` 使用 privileged offset-correct 方向，`micro_probe` 使用 observable-only A/B/A paired-axis scoring。probe 结果 5/5 选择 `baseline`，原因均为 `baseline_preferred_no_consistent_pair`。

| Case | matched full | matched off | router | micro-probe | router vs full | probe vs full | probe selection |
|---|---:|---:|---:|---:|---:|---:|---|
| cob_linear_asymmetric | 0.3145 | 0.3181 | 0.1802 | 0.1911 | -42.70% | -39.24% | baseline |
| thruster_single_ramp_to_target_asymmetric | 0.3050 | 0.2909 | 0.1803 | 0.1911 | -40.90% | -37.34% | baseline |
| density095_linear_asymmetric | 0.2426 | 0.2879 | 0.1819 | 0.1934 | -25.02% | -20.29% | baseline |
| torque_l0p5_asymmetric | 0.3158 | 0.3191 | 0.1803 | 0.1911 | -42.92% | -39.47% | baseline |
| torque_l1p0_asymmetric | 0.3170 | 0.3201 | 0.1803 | 0.1912 | -43.13% | -39.70% | baseline |

结论：

1. privileged router 的 `target_drift=-0.05, drift_axes=[0,1]` 在 5/5 个 asymmetric linear 问题组中均显著优于 matched full/off。
2. observable-only micro-probe 没有强行选择某个 drift 方向，而是 5/5 回退 `baseline`；该 conservative fallback 同样显著优于 matched full/off。
3. 这说明 asymmetric linear 的主要问题不是 STDW 组件本身，而是错误 drift 方向；部署时需要先 router/probe，再决定是否启动 drift adaptation。

## 3. Density no-quantile seed1/seed2

| Seed | full | off | no_quantile_filter | no_quantile vs full | no_quantile vs off |
|---:|---:|---:|---:|---:|---:|
| 1 | 0.3941 | 0.2640 | 0.2561 | -35.02% | -3.02% |
| 2 | 0.3941 | 0.2640 | 0.2561 | -35.02% | -3.02% |

结果与 seed0 完全一致。解释上应谨慎：当前 single-env play/eval 流程在这些 case 中表现为确定性轨迹，`seed1/seed2` 没有引入有效随机性。因此该结果说明该单点收益在当前确定性执行路径下可复现，但还不能等价为多随机种子统计显著性。

## 4. 论文可引用结论

建议论文中引用为：

1. "A privileged offset-correct router reduces asymmetric-linear final MSE from 0.24-0.32 to about 0.18 across COM-COB, actuator, density, and torque disturbances."
2. "The observable-only micro-probe selects baseline in all five asymmetric-linear cases and reduces final MSE to about 0.19, confirming that conservative fallback removes the wrong-direction slow-loop failure without privileged COM-COB access."
3. "The density no-quantile improvement is reproducible under the current deterministic play path, but additional stochastic seeds or randomized initial states are needed before treating it as a general component change."

## 5. 下一步

不建议继续扩大消融。若要进一步提高论文证据强度，下一步应只做一个专门的 stochastic repeat：随机初始姿态/流场/episode seed，而不是继续重复当前确定性 seed 参数。

# STDW asymmetric-linear 消融诊断报告

日期：2026-06-13

## 1. 范围

本轮按强验证报告中的默认建议继续，只诊断 `asymmetric + linear` 中 STDW on 明显劣于 off 的组，不扩大到 base 或 step。

覆盖 5 个问题组：

- `cob_linear_asymmetric`
- `thruster_single_ramp_to_target_asymmetric`
- `density095_linear_asymmetric`
- `torque_l0p5_asymmetric`
- `torque_l1p0_asymmetric`

每组跑 5 个组件关闭项：

- `no_slow_loop`
- `no_lyapunov`
- `no_pseudo`
- `no_trigger_gate`
- `no_quantile_filter`

共 25 cell，全部 `returncode=0`。

产物：

- `.results/exp_stdw_strong_ablation_20260613/strong_ablation_runs.csv`
- `.results/exp_stdw_strong_ablation_20260613/strong_ablation_summary.csv`
- `.results/exp_stdw_strong_ablation_20260613/strong_ablation_report.md`

## 2. 主结果

`vs_full` 表示相对原始 STDW on 的变化；`vs_off` 表示相对 STDW off 的变化。负数更好。

| Base case | Best ablation | final_mse | vs_full | vs_off |
|---|---|---:|---:|---:|
| cob_linear_asymmetric | no_slow_loop | 0.3189 | -39.56% | +0.00% |
| thruster_single_ramp_to_target_asymmetric | no_slow_loop | 0.2915 | -46.15% | +0.00% |
| density095_linear_asymmetric | no_quantile_filter | 0.2561 | -35.02% | -3.02% |
| torque_l0p5_asymmetric | no_slow_loop | 0.3199 | -39.45% | +0.00% |
| torque_l1p0_asymmetric | no_slow_loop | 0.3209 | -39.39% | +0.00% |

完整排序：

| Base case | Ablation | final_mse | vs_full | vs_off | slow triggers |
|---|---|---:|---:|---:|---:|
| cob_linear_asymmetric | no_slow_loop | 0.3189 | -39.56% | +0.00% | 0 |
| cob_linear_asymmetric | no_quantile_filter | 0.3404 | -35.48% | +6.75% | 20 |
| cob_linear_asymmetric | no_pseudo | 0.5249 | -0.52% | +64.60% | 20 |
| cob_linear_asymmetric | no_lyapunov | 0.5273 | -0.07% | +65.35% | 20 |
| cob_linear_asymmetric | no_trigger_gate | 0.5277 | +0.00% | +65.47% | 20 |
| thruster_single_ramp_to_target_asymmetric | no_slow_loop | 0.2915 | -46.15% | +0.00% | 0 |
| thruster_single_ramp_to_target_asymmetric | no_quantile_filter | 0.3387 | -37.43% | +16.20% | 20 |
| thruster_single_ramp_to_target_asymmetric | no_pseudo | 0.5247 | -3.06% | +80.03% | 20 |
| thruster_single_ramp_to_target_asymmetric | no_lyapunov | 0.5289 | -2.28% | +81.48% | 20 |
| thruster_single_ramp_to_target_asymmetric | no_trigger_gate | 0.5413 | +0.00% | +85.71% | 20 |
| density095_linear_asymmetric | no_quantile_filter | 0.2561 | -35.02% | -3.02% | 20 |
| density095_linear_asymmetric | no_slow_loop | 0.2640 | -33.00% | +0.00% | 0 |
| density095_linear_asymmetric | no_lyapunov | 0.3802 | -3.53% | +43.99% | 20 |
| density095_linear_asymmetric | no_trigger_gate | 0.3941 | +0.00% | +49.25% | 20 |
| density095_linear_asymmetric | no_pseudo | 0.4100 | +4.04% | +55.28% | 20 |
| torque_l0p5_asymmetric | no_slow_loop | 0.3199 | -39.45% | +0.00% | 0 |
| torque_l0p5_asymmetric | no_quantile_filter | 0.3415 | -35.36% | +6.76% | 20 |
| torque_l0p5_asymmetric | no_pseudo | 0.5272 | -0.21% | +64.82% | 20 |
| torque_l0p5_asymmetric | no_lyapunov | 0.5273 | -0.20% | +64.83% | 20 |
| torque_l0p5_asymmetric | no_trigger_gate | 0.5283 | +0.00% | +65.16% | 20 |
| torque_l1p0_asymmetric | no_slow_loop | 0.3209 | -39.39% | +0.00% | 0 |
| torque_l1p0_asymmetric | no_quantile_filter | 0.3414 | -35.52% | +6.39% | 20 |
| torque_l1p0_asymmetric | no_pseudo | 0.5267 | -0.51% | +64.14% | 20 |
| torque_l1p0_asymmetric | no_lyapunov | 0.5281 | -0.24% | +64.59% | 20 |
| torque_l1p0_asymmetric | no_trigger_gate | 0.5294 | +0.00% | +64.99% | 20 |

## 3. 诊断结论

1. asymmetric linear 的主要劣化来自 **slow-loop update 在错误 drift 方向上持续学习**。`no_slow_loop` 在 4/5 个组里是最优，且几乎完全回到 STDW off。
2. `no_trigger_gate` 与 full STDW 基本一致，说明当前触发门没有显著加剧或缓解该失败；它只是允许 20 次 slow-loop update 正常发生。
3. `no_pseudo` 和 `no_lyapunov` 也基本保留劣化，说明单独关闭伪动作或 Lyapunov mask 不能修复 asymmetric linear。
4. `no_quantile_filter` 有中等缓解，在 density 组甚至略优于 off（-3.02%），但在 COM、推进器和 torque 组仍劣于 off，因此不能作为通用修复。

## 4. 追加验证结果

默认修复方向应保持此前判断：**不要删除 STDW 组件，而是在 asymmetric 实物/仿真上先做 drift 方向选择或证据不足回退 baseline**。

已按此方向追加两类小验证，详见 [`REPORT_stdw_followup_validation_20260613.md`](REPORT_stdw_followup_validation_20260613.md)：

1. asymmetric linear + router/micro-probe：router 使用 `target_drift=-0.05, drift_axes=[0,1]`，5/5 组显著优于 matched full/off；micro-probe 5/5 选择 `baseline`，同样显著优于 matched full/off。
2. density095 linear + `no_quantile_filter` seed1/seed2：结果与 seed0 完全一致，说明当前确定性 play 路径可复现该单点收益；但由于 seed 未引入有效随机性，不能直接当作多随机种子统计显著性。

base 组和 step 组仍不需要进一步消融。若要继续增强论文证据，应引入随机初始姿态/流场/episode seed，而不是重复当前 deterministic seed 参数。

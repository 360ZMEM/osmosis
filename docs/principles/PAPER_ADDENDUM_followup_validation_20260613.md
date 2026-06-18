# Paper Addendum: Follow-up Router/Probe Validation

日期：2026-06-13

本附录用于直接纳入 `PAPER_STDW_EN.md` / `PAPER_STDW_CN.md` 的实验章节。

## English Draft

After the default asymmetric-linear ablation, we ran a matched-timing follow-up validation to separate drift-direction selection from slow-loop component effects. The matched runs use `drift_start_step=320` and `drift_end_step=1200`, ensuring that the observable-only micro-probe finishes before the main drift ramp. Four variants are compared for each asymmetric-linear disturbance: matched full STDW, matched STDW off, privileged offset-correct router, and observable-only micro-probe.

| Case | matched full | matched off | router | micro-probe | probe selection |
|---|---:|---:|---:|---:|---|
| cob_linear_asymmetric | 0.3145 | 0.3181 | 0.1802 | 0.1911 | baseline |
| thruster_single_ramp_to_target_asymmetric | 0.3050 | 0.2909 | 0.1803 | 0.1911 | baseline |
| density095_linear_asymmetric | 0.2426 | 0.2879 | 0.1819 | 0.1934 | baseline |
| torque_l0p5_asymmetric | 0.3158 | 0.3191 | 0.1803 | 0.1911 | baseline |
| torque_l1p0_asymmetric | 0.3170 | 0.3201 | 0.1803 | 0.1912 | baseline |

The privileged router consistently selects the corrective negative drift (`target_drift=-0.05`, `drift_axes=[0,1]`) and reduces final MSE to approximately 0.18. The observable-only micro-probe selects `baseline` in all five cases with reason `baseline_preferred_no_consistent_pair`, reducing final MSE to approximately 0.19 without privileged COM-COB access. This confirms that conservative baseline fallback removes the wrong-direction slow-loop failure.

For the density095 no-quantile observation, seed1/seed2 exactly reproduce seed0: full = 0.3941, off = 0.2640, and no-quantile = 0.2561. This is deterministic reproducibility evidence under the current play path, but it should not be described as stochastic seed robustness unless future runs randomize initial states, current fields, or episode conditions.

## 中文草稿

默认 asymmetric-linear 消融后，我们追加 matched-timing 验证，用于区分 drift 方向选择和 slow-loop 组件效应。matched runs 使用 `drift_start_step=320`、`drift_end_step=1200`，保证 observable-only micro-probe 在主 drift ramp 前结束。每个 asymmetric-linear 扰动比较四个变体：matched full STDW、matched STDW off、privileged offset-correct router、observable-only micro-probe。

| Case | matched full | matched off | router | micro-probe | probe selection |
|---|---:|---:|---:|---:|---|
| cob_linear_asymmetric | 0.3145 | 0.3181 | 0.1802 | 0.1911 | baseline |
| thruster_single_ramp_to_target_asymmetric | 0.3050 | 0.2909 | 0.1803 | 0.1911 | baseline |
| density095_linear_asymmetric | 0.2426 | 0.2879 | 0.1819 | 0.1934 | baseline |
| torque_l0p5_asymmetric | 0.3158 | 0.3191 | 0.1803 | 0.1911 | baseline |
| torque_l1p0_asymmetric | 0.3170 | 0.3201 | 0.1803 | 0.1912 | baseline |

privileged router 稳定选择 corrective negative drift（`target_drift=-0.05`、`drift_axes=[0,1]`），将 final MSE 降至约 0.18。observable-only micro-probe 在 5/5 个 case 中选择 `baseline`，原因均为 `baseline_preferred_no_consistent_pair`，并在无特权 COM-COB 的情况下将 final MSE 降至约 0.19。这确认保守 baseline fallback 可以消除错误方向 slow-loop 失败。

对于 density095 no-quantile 单点收益，seed1/seed2 与 seed0 完全一致：full = 0.3941，off = 0.2640，no-quantile = 0.2561。该结果可作为当前确定性执行路径下的可复现证据，但不能直接表述为随机种子鲁棒性；若论文需要统计意义，应进一步随机化初始姿态、流场或 episode 条件。

# Paper Addendum: Dynamic Parameter Identification

日期：2026-06-13

本附录用于补充论文实验与讨论章节，重点回答：动态 `density`、推进器效率和推进器角度变化是否需要比 COM-COB drift probe 更丰富的在线辨识。

## English Draft

To test whether the observed asymmetric-linear degradation can be explained solely by COM-COB drift-direction selection, we added a dynamic parameter-family probe. The probe is observable-only: it consumes recent tracking errors, executed actions, and control-effort histories, and it does not access true water density, true thruster efficiency, true thruster orientation, or the injected torque pulse state. We further added a small deterministic action dither during the probe window and a layered detector that first separates impulse-like disturbances from persistent parameter shifts. The probe is diagnostic by default. When enabled with `param_probe_apply_result=True`, it may only gate slow-loop updates after a non-ambiguous detection; ambiguous outputs are treated as rollback states.

We evaluated a 12-cell matrix with four disturbance families and three modes:

| family | full | off | param_probe | selected family |
|---|---:|---:|---:|---|
| density | 0.2426 | 0.2879 | 0.2426 | ambiguous |
| thruster efficiency | 0.3050 | 0.2909 | 0.3050 | ambiguous |
| thruster angle | 0.3129 | 0.3156 | 0.3129 | ambiguous |
| torque negative control | 0.3158 | 0.3191 | 0.3158 | ambiguous |

All probe-enabled runs were classified as `ambiguous` by the layered detector. The best family score was usually density-like, but the margin was only about 0.08, below the conservative acceptance threshold. Therefore the probe did not gate the slow loop and did not compensate any parameter. The resulting performance stayed close to the full STDW baseline, which is the intended rollback behavior.

These results suggest that the previous COM-COB router/probe solves the wrong-direction drift failure, but it does not solve dynamic parameter identification. A3 histories plus small deterministic dithers are not sufficient in this matrix to separate global density changes from local actuator efficiency or actuator geometry changes. Robust handling of these scenarios likely requires an explicit parameter-identification layer, richer sensing, or more structured active excitation. The intermittent torque pulse should remain an external-disturbance negative control rather than a stable parameter-estimation target.

## 中文草稿

为验证 asymmetric-linear 劣化是否只能由 COM-COB drift 方向选择解释，我们加入 dynamic parameter-family probe。该 probe 严格 observable-only：只使用近期 tracking error、executed action 和 control-effort 历史，不读取真实水密度、真实推进器效率、真实推进器方向或注入的 torque pulse 状态。随后又加入 probe 窗口内的主动小幅动作激励，以及分层检测：先区分 impulse-like 外扰，再判断 persistent parameter shift 与 family score。默认只做诊断；当 `param_probe_apply_result=True` 时，也只有非 ambiguous 检测才允许 conservative slow-loop gating，`ambiguous` 必须回退。

我们评估了 12-cell 小矩阵，覆盖四类扰动和三种模式：

| family | full | off | param_probe | selected family |
|---|---:|---:|---:|---|
| density | 0.2426 | 0.2879 | 0.2426 | ambiguous |
| thruster efficiency | 0.3050 | 0.2909 | 0.3050 | ambiguous |
| thruster angle | 0.3129 | 0.3156 | 0.3129 | ambiguous |
| torque negative control | 0.3158 | 0.3191 | 0.3158 | ambiguous |

所有启用 probe 的 run 都被分层检测判为 `ambiguous`。最佳 family score 通常呈 density-like，但 margin 只有约 0.08，低于保守接受阈值。因此最终版本不触发 slow-loop gating，也不进行任何参数补偿，性能基本回到 full STDW 基线，这就是预期的回退行为。

因此，本轮结果支持更保守的论文结论：此前 COM-COB router/probe 修复的是错误 drift 方向失败，但没有解决动态参数辨识问题。在当前矩阵中，A3 历史加小幅 deterministic dither 仍不足以稳定区分全局 density 变化、局部 actuator efficiency 下降和 actuator geometry 改变。若要鲁棒处理这些动态场景，需要显式参数辨识层、更丰富传感或更结构化的主动激励。间歇 torque pulse 应保持为外扰负例，而不是稳定参数估计目标。

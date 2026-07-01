# STDW 学习硬约束专项计划 — 2026-06-28

## 0. 目标

本文件固定当前上下文，并给出后续 step-by-step 修复路线。核心问题是：

> 在关闭 OPR / router / micro-probe 的 `asymmetric` hull 上，默认 STDW 会让策略学坏；现有
> Lyapunov mask 不能防止这种劣化。下一步需要给 STDW 学习闭环增加硬约束，使 eval 不差于
> matched clean off。

这里的“硬约束”不是再调一个 loss 权重，而是把“是否允许 drift / 是否允许慢环更新 / 是否接受更新后策略”
变成可拒绝、可回退的决策。

## 1. 当前事实

### 1.1 最新矩阵结论

来源：[`REPORT_stdw_safety_pressure_20260628.md`](REPORT_stdw_safety_pressure_20260628.md)

- 32-cell `small_hard` 矩阵已完成，全部 `returncode=0`。
- `asymmetric + OPR/router/probe off` 下，默认 STDW 仍显著劣化：
  - medium: `0.2263 -> 0.5456`，约 `+141%`
  - storm: `0.2259 -> 0.5392`，约 `+139%`
- `strict_sample_mask` 与默认 STDW 基本一致，不能修复。
- `guarded_drift + zero_drift` 能把 medium/storm 拉回 clean-off 水平，但这是保守回退，不恢复 STDW 收益。
- 1% 控制器 mismatch 不是主导新失败源：
  - `base` 上默认 STDW 约 `-67.9%`
  - `asymmetric` 上仍重复 `+141%` drift 方向失败
- 360 度后空翻 eval 当前失败，应作为训练课程处理，不应和小角度 Lyapunov guard 共用同一阈值设计。

### 1.2 更早机理结论

来源：

- [`DIAG_p1_p2_p5_20260610.md`](DIAG_p1_p2_p5_20260610.md)
- [`REPORT_stdw_strong_ablation_20260613.md`](REPORT_stdw_strong_ablation_20260613.md)

已知机理：

- `asymmetric` 初始 `com_to_cob=(0.05,0.05,0.01)`。
- 默认 STDW drift 语义是 `base_offset + frac * target_drift`。
- 默认 `--target_drift +0.05 --drift_axes 0` 会把 asymmetric 从 `(0.05,0.05)` 推到 `(0.10,0.05)`。
- 通道劣化主要来自 pitch/depth，不是 yaw。
- `no_slow_loop` 能在多个 asymmetric 组里回到 off，说明“错误 drift 方向上的持续慢环学习”是关键风险。
- router / micro-probe 曾能修复，但本专项要求先研究不依赖 OPR 的硬约束。

## 2. 当前 STDW 学习闭环

### 2.1 drift 注入

位置：`easyuuv_stdw_wrapper.py`

- `_compute_drift_fraction(step)` 计算 `frac`。
- `_apply_drift(step)` 每步写：

```text
com_to_cob_offsets[:, axis] = base_offset[:, axis] + frac * target_drift
```

这意味着当前 `target_drift` 是“增量”，不是“目标绝对 offset”。在 asymmetric hull 上，如果增量方向和已有偏移同向，就会制造更大的偏心。

### 2.2 Lyapunov mask

位置：`easyuuv_stdw_wrapper.py`

当前能量：

```text
V_t = 0.5 * e^T P e
e = [roll_err, pitch_err, yaw_err, depth_err]
```

当前默认 mask：

```text
stdw_mask = 1 iff V_t - V_prev < lyapunov_eps
```

新增的 `strict_sample_mask` 只是更严格地决定 `stdw_mask`，本质仍是样本权重。

### 2.3 replay buffer

位置：`utils/stdw_buffer.py`

buffer 保存：

```text
state, action, pseudo_action, reward, next_state, error, stdw_mask, lyapunov_V, domain_tag, step
```

`sample_pair()` 只按 source/target domain tag 抽样，并可按 error quantile 过滤。它不知道“某次 policy update 是否会让闭环更差”。

### 2.4 slow-loop update

位置：`workflows/play_stdw_adapt.py`

核心 loss：

```text
loss = (1-rho) * L_src + rho * L_tgt + lambda_reg * L_reg
```

其中：

- `L_src`：当前 policy 对 source obs 的动作，贴近 frozen `policy_ref`
- `L_tgt`：当前 policy 对 target obs 的动作，贴近 pseudo_action
- `L_reg`：参数 L2 或 behavior KL
- `stdw_mask` 只用于 `L_src/L_tgt` 加权

当前硬边界只有：

- `torch.isfinite(loss)`
- `effective_batch_frac > 0`
- gradient clip

这些不能保证 update 后闭环指标不恶化。

## 3. 为什么现有 Lyapunov mask 不够

当前 Lyapunov mask 有三个结构性缺陷：

1. **只过滤样本，不拒绝更新。**
   只要 batch 里还有 mask=1 的样本，optimizer 仍可朝错误方向更新。

2. **只看单步 `dV`，不看 update 前后策略差异。**
   它没有比较 `policy_before_update` 和 `policy_after_update` 在同一批状态上的动作变化是否安全。

3. **不约束 drift 方向。**
   asymmetric 的根因是默认 `+x` drift 与已有 offset 同向叠加。mask 再严格也不能改变 drift 正在把系统推向错误域。

结论：Lyapunov mask 是“软样本筛”，不是硬安全约束。

## 4. 硬约束设计空间

### 4.1 层 A：domain/drift 硬约束

约束对象：`target_drift`、`drift_axes`、`drift_fraction`

可实现规则：

- `zero_drift_on_harm`：一旦监测到 harm，令 `target_drift=0`。
- `freeze_drift_on_harm`：一旦监测到 harm，冻结当前 drift fraction。
- `offset_projection`：如果能读到初始 `com_to_cob_xy`，只允许 drift 把 offset norm 变小，不允许变大。

建议：

- 对 OPR-off 安全压测，优先保留 `zero_drift_on_harm`，因为它不依赖 offset 读数。
- 对实际部署/论文提升，`offset_projection` 更有意义，但它属于 router/OPR 类能力，不应混入“不依赖 OPR”的结论。

### 4.2 层 B：batch-level update acceptance

约束对象：一次 slow-loop optimizer step

基本流程：

```text
1. 保存 theta_before
2. 用当前 batch 计算候选 loss
3. 执行一次候选 update
4. 在同一 batch 上计算 safety metrics:
   - behavior drift: ||pi_after(s) - pi_ref(s)||^2
   - action delta: ||pi_after(s) - pi_before(s)||^2
   - target pseudo error: ||pi_after(s_tgt) - pseudo_action||^2
   - Lyapunov-risk proxy: weighted channel error / stored V trend
5. 若任何硬条件失败，恢复 theta_before，记录 rejection
```

候选硬条件：

```text
behavior_mse_after <= behavior_mse_before + eps_behavior
action_delta_mse <= max_action_delta
target_pseudo_mse_after <= target_pseudo_mse_before + eps_target
effective_batch_frac >= min_effective_batch_frac
```

优点：

- 不需要额外仿真步，也不依赖 OPR。
- 可以真正拒绝一次有害参数更新。

限制：

- batch proxy 仍不是闭环 rollout；只能作为第一层硬约束。

### 4.3 层 C：shadow rollout acceptance

约束对象：一次或一组 slow-loop updates 后的策略

流程：

```text
1. 保留 policy_before
2. update 得到 policy_candidate
3. 用短窗口 K step shadow eval 比较 candidate vs baseline/current
4. 若 candidate 的 filtered_error / compound_error 高于 off/current 阈值，则 reject
```

优点：

- 最接近“不会学坏”的定义。

限制：

- Isaac 环境状态复制成本高；当前代码没有轻量 clone/reset-to-state 入口。
- 若直接插入真实环境，会污染主 episode。
- 可作为后续独立 runner / 双环境验证，而不是第一步。

### 4.4 层 D：policy trust region / 参数投影

约束对象：策略参数或动作输出

可实现规则：

- `max_param_delta_norm`：限制 `||theta-theta_pre||`。
- `max_behavior_kl`：对 source/target batch 上 `||pi(theta)-pi_ref||^2` 设硬阈值。
- `max_action_delta`：对 `||pi_after(s)-pi_before(s)||` 设硬阈值。

建议：

- 和层 B 合并做第一版：update 后检查，失败就 rollback。
- 不建议只靠更大的 `lambda_reg`，因为那仍是软约束。

### 4.5 层 E：control-level barrier

约束对象：低层控制器参数 / action / pseudo_action

可实现规则：

- pseudo_action 的 correction 必须降低当前 step 的 channel error proxy，否则不用 pseudo_action。
- 限制 pseudo_action 相对 policy action 的角度/范数。
- 对 AUV 姿态/深度通道建立 channel-wise barrier，例如 pitch/depth 超阈值时禁止 target-domain update。

建议：

- 作为第二步，因为需要额外通道指标分析。
- 对 asymmetric 的 pitch/depth 劣化很相关，但实现前应先把 batch-level acceptance 做起来。

## 5. 推荐修复路线

### Step 1：修正验收统计的浮点容差

问题：

- `sweep_stdw_safety_pressure.py` 当前 `pass_vs_off` 使用严格 `final_mse <= off`。
- medium guard cell 只比 off 大 `4.45e-12`，被标成 False。

修复：

- 新增 `--pass_abs_tol`，默认 `1e-9`。
- 新增 `--pass_rel_tol`，默认 `1e-6`。
- 判据改成：

```text
final_mse <= off + max(pass_abs_tol, pass_rel_tol * abs(off))
```

验收：

- 重新聚合现有 `.results/stdw_safety_pressure_20260628_221237/pressure_runs.csv`。
- medium/storm guard 均应 pass。

### Step 2：把 guarded fallback 从“drift 回退”拆成明确状态机

当前问题：

- `guarded_drift + zero_drift` 能阻止 asymmetric 学坏，但它把“检测 harm”“停止 slow-loop”“回退 drift”耦合在一起。
- 对 360 flip，guard 触发很多次并更糟，说明它缺少 reference-aware 状态。

建议状态机：

```text
NORMAL:
  允许 drift 和 slow-loop
  若 harm_score > threshold -> SUSPECT

SUSPECT:
  暂停 slow-loop
  冻结 drift fraction
  连续观察 K 步
  若恢复 -> NORMAL
  若继续恶化 -> FALLBACK

FALLBACK:
  target_drift = 0 或 drift_frac 退回 0
  slow-loop 禁止更新
  本 episode 不再重新启用 STDW
```

必须记录：

- `stdw_safety_state`
- `stdw_safety_transition`
- `stdw_update_rejected_count`
- `stdw_fallback_step`

验收：

- normal wave asymmetric：不差于 off。
- flip360：不能比当前 guard 更差；若检测到 reference_mode=flip360_sine，可默认禁用该 guard 或使用单独阈值。

### Step 3：实现 batch-level update acceptance / rollback

这是最关键的“学习硬约束”。

新增 CLI：

```text
--stdw_update_acceptance off|batch_trust
--stdw_max_behavior_mse
--stdw_max_action_delta_mse
--stdw_max_target_mse_increase
--stdw_min_effective_batch_frac
```

实现位置：

- `workflows/play_stdw_adapt.py` slow-loop update 段。

实现方式：

```text
theta_before = clone(policy.state_dict())
metrics_before = eval_batch_metrics(policy)
optimizer.step()
metrics_after = eval_batch_metrics(policy)
if violates(metrics_before, metrics_after):
    policy.load_state_dict(theta_before)
    update_rejected_count += 1
else:
    update_accepted_count += 1
```

第一版 acceptance 推荐硬条件：

```text
effective_batch_frac >= 0.2
behavior_mse_after <= behavior_mse_before + 1e-4
action_delta_mse <= 1e-3
target_mse_after <= target_mse_before + 1e-4
```

验收矩阵：

- 先跑 `lyapunov_asym_guard` 的 8 cell。
- 再跑 `controller_misset` 的 asymmetric 9 cell。
- 目标：`stdw_default + batch_trust` 不差于 off，且比 `zero_drift` 更少触发彻底回退。

### Step 4：给 pseudo_action 加 channel-aware hard gate

背景：

- asymmetric 失败主要是 pitch/depth。
- 现有 pseudo-action 只 clip correction，不判断 correction 是否改善通道风险。

新增 CLI：

```text
--pseudo_channel_gate off|pitch_depth
--pseudo_channel_gate_threshold
```

规则：

```text
if pitch_or_depth_error is above threshold and correction increases risky action component:
    a_pseudo = action
```

第一版可以保守：

- 只在 `asymmetric` 或 `stdw_safety_state != NORMAL` 时启用。
- 只影响 pseudo label，不影响实时 policy action。

验收：

- asymmetric default STDW 的 final MSE 应低于当前 `0.5456`。
- 若不能低于 off，仍需 batch acceptance / fallback 保底。

### Step 5：训练阶段处理 360 flip

360 不是当前 Lyapunov 小角度 guard 能修的。

建议新训练课程：

```text
phase A: reference_mode=sine_sweep, amp=[0.5,0.5,0.2]
phase B: flip360_sine, amp=[pi/2,pi/2,0]
phase C: flip360_sine, amp=[pi,pi,0]
```

验收：

- clean off flip360 final MSE 必须先显著低于当前 base `3.37` / asymmetric `5.90`。
- 只有 clean off 可控后，再讨论 STDW/guard 是否提升。

## 6. 建议的下一轮实验顺序

### E1：只修统计容差，不改学习

目的：避免浮点误标干扰判断。

命令：

```bash
python workflows/tools/aggregate_stdw_safety_pressure.py \
  --matrix_dir .results/stdw_safety_pressure_20260628_221237 \
  --pass_abs_tol 1e-9 --pass_rel_tol 1e-6
```

若不新增聚合脚本，也可直接在 `sweep_stdw_safety_pressure.py` 加 replay/aggregate 模式。

### E2：batch acceptance 小矩阵

目的：验证“拒绝有害学习更新”是否能替代简单 zero-drift。

最小矩阵：

```text
embodiment = asymmetric
wave = medium, storm
variant = off_clean, stdw_default, stdw_batch_trust, lyap_guard_zero
```

通过标准：

```text
stdw_batch_trust.final_mse <= off.final_mse + tolerance
nonfinite/reset 不增加
update_rejected_count > 0 表明约束实际工作
```

### E3：batch acceptance + 1% mismatch

目的：确认在极小控制器初值下，硬约束不会误伤 base 的 STDW 收益，也能保护 asymmetric。

矩阵：

```text
embodiment = base, asymmetric
mismatch = all_pd_0p01
variant = off_clean, stdw_default, stdw_batch_trust, lyap_guard_zero
```

通过标准：

- base: `stdw_batch_trust` 应尽量保留默认 STDW 的收益。
- asymmetric: `stdw_batch_trust` 不差于 off。

### E4：360 单独训练计划

目的：不要让 360 eval 失败污染 STDW 安全结论。

先跑训练入口 dry run / short run，再独立评估：

```bash
bash custom_workflows/run_with_isaac_env.sh workflows/train_meta.py \
  --task EasyUUV-Direct-Parametric-v1 \
  --num_envs 512 \
  --max_iterations <short> \
  --reference_mode flip360_sine \
  --headless
```

## 7. 论文叙事边界

可以说：

- 默认 STDW 在 asymmetric hull 上会因错误 drift 方向劣化。
- 单步 Lyapunov sample mask 不是安全保证。
- 硬约束必须作用在 drift/update acceptance/fallback 层。
- Conservative fallback 能保证不比 clean off 差，但牺牲 STDW 收益。

不能说：

- “Lyapunov mask 已经保证 STDW 安全。”
- “360 后空翻可由 eval 阶段 STDW 自适应解决。”
- “1% 控制器 mismatch 是 asymmetric 失败主因。”

## 8. 当前建议

下一步不要先扩大矩阵，也不要先重训。优先做：

1. `pass_vs_off` 容差修正。
2. `stdw_update_acceptance=batch_trust`，支持 update rollback。
3. asymmetric 8-cell 小矩阵验证。
4. 若 batch trust 有效，再做 1% mismatch 小矩阵。
5. 360 另开训练课程，不与当前 STDW 安全矩阵混在一起。

## 9. OPR / 无先验 Domain Transfer 保留项

当前先不继续实现 OPR/router/projection，保留为后续拍板项：

- `guarded_drift + zero_drift` 只能提供 conservative fallback。
- 若后续要求用无先验 domain transfer 替代 OPR，应优先比较：
  - micro-probe 选择 drift sign/axis；
  - drift-level projection，只允许 COM-COB offset norm 下降；
  - 无先验 candidate ensemble / online model selection。
- 该问题不阻塞 flip360 训练；flip360 当前作为训练课程单独推进。

## 10. Flip360 训练结果（已更新到连续参考）

报告：

- `docs/engineering/REPORT_flip360_training_20260628.md`

> 重要更正：早期 `flip360_sine` 只在 reset 时设一次目标、控制步内不推进，导致旧的
> `model_2550`（base 2.24 / asym 2.18）实为“静态大姿态保持”假象，已作废。`_update_reference`
> 现已逐步推进参考，下列为连续后空翻下的真实结果。

当前最佳 checkpoint（幅度课程 A→B）：

```text
logs/rsl_rl/easyuuv_parametric/2026-06-29_23-50-50_flip360_curric_b_full_pi_stage0/model_2846.pt
```

连续 flip360 eval（total_steps=1500, use_stdw=False, seed=0）：

- base: `2.8675 -> 2.0701`
- asymmetric: `5.5360 -> 3.6606`

ordinary medium regression：

- base: `0.2257 -> 0.2229`
- asymmetric: `0.2263 -> 0.2266`（噪声内，无遗忘）

结论：`model_2846` 是首个在 flip360 上 Pareto 占优 A3 且 ordinary 不退化的 checkpoint。
但连续 flip360 残余姿态 RMSE 仍约 0.85–0.91 rad，误差集中在 roll≈π/2~2π/3。见第 12 节后续计划。

## 11. OPR / 无先验 Domain Transfer 保留项

当前先不继续实现 OPR/router/projection，保留为后续拍板项：

- `guarded_drift + zero_drift` 只能提供 conservative fallback。
- 若后续要求用无先验 domain transfer 替代 OPR，应优先比较：
  - micro-probe 选择 drift sign/axis；
  - drift-level projection，只允许 COM-COB offset norm 下降；
  - 无先验 candidate ensemble / online model selection。
- 该问题不阻塞 flip360 训练；flip360 当前作为训练课程单独推进。

## 12. Flip360 平滑课程与物理极限验证计划（2026-06-30）

来源：`REPORT_flip360_training_20260628.md` 的 Round 2 与物理极限分析章节。
对 `model_2846` 的逐步 CSV 分析判定为 **ANGLE-LIMITED**（误差–参考角速度相关≈0.10，
近饱和步 <0.3%，误差集中在 roll≈π/2~2π/3），参考峰值角速度仅≈1 rad/s。
因此先用准静态 keep 测试核查物理回正力矩极限，再决定是否放慢参考或做平滑课程。

### Step F1：难角度准静态 keep 测试（最高优先，先做）

目的：把 roll∈[π/2,2π/3] 的难区间从“连续穿越”改成“准静态保持”，
若准静态都保不住且推力饱和 → 物理极限；若准静态能保住 → 是动态/学习不足。

做法（不改命令契约，用极低频近似准静态）：

```bash
# keep 测试：ref_sine_freq 设到接近 0，使目标在大角度处缓慢驻留
bash custom_workflows/run_with_isaac_env.sh workflows/play_stdw_adapt.py \
  --load_run 2026-06-29_23-50-50_flip360_curric_b_full_pi_stage0 \
  --checkpoint model_2846.pt \
  --workflow_config workflows/configs/pressure_flip360_keep_slow.yaml \
  --embodiment base --use_stdw False
# asymmetric 同上换 --embodiment asymmetric
```

判据：用 `workflows/analyze_flip360_limits.py` 看难角度箱 rmse 与近饱和步占比。

### Step F2：参考速度敏感性扫描

目的：给“放慢参考是否有效”直接证据。

```bash
# 对 ref_sine_freq ∈ {0.025, 0.05, 0.1} 各跑一次 flip360 eval，再用分析脚本对比 RMSE
```

判据：若 RMSE 随频率降低而单调下降 → rate 相关，可放慢参考；
若几乎不变 → 确认 angle-limited，转 Step F3/F4。

### F1 + F2 取证结果（2026-06-30，已完成）

来源：`.results/flip360_keep_analysis.md`、`.results/flip360_freq_sweep_analysis.md`。
对 `model_2846` 只改 `ref_sine_freq`，其余 flip360 配置不变：

| ref_sine_freq (Hz) | base RMSE | asym RMSE | base 近饱和步占比 |
|---|---:|---:|---:|
| 0.005（准静态 keep） | 1.742 | 1.118 | **15.2%** |
| 0.025 | 1.001 | 0.948 | 0.3% |
| 0.05 | 0.915 | 0.854 | 0.1% |
| 0.1 | 0.792 | 0.794 | 0.2% |

结论：**RMSE 随参考变快而单调下降**，与 rate-limited 完全相反。判定为
**驻留时间惩罚型角度极限（dwell-time-penalized angle limit）**：难区间（接近倒置）
浮力回正力矩持续反向，驻留越久越饱和；快速穿过反而更好。
准静态 keep 下 base 15.2% 步推力近饱和却仍保不住姿态 = 触及推力力矩物理上限。
**“放慢参考”假设被否定**。据此：跳过原 F4 的“放慢”分支，进入 F3，并把 F4 改为“不放慢 + 容差整形”。

### Step F3：平滑幅度爬坡课程（若仍有力矩裕度）

把“A(±π/2) 一步跳 B(±π)”改为多段平滑：π/2 → 3π/4 → π，每段约 120 iters，
flip 概率 `ref_mix_flip_prob` 由 0.3 退火到 0.5，配合双目标（flip360 + ordinary）挑 checkpoint。

#### F3 执行与结果（2026-06-30，已完成，未胜出）

三段从 A3 `model_2398` 顺序 fine-tune（`--meta_stage 0 --resume_load_optimizer False`），
配置见 `workflows/configs/train_flip360_f3_s{1,2,3}_*.yaml`：

| 段 | 幅度 | flip_prob | lr | 起点 | run 目录 | 末 ckpt |
|---|---|---:|---:|---|---|---|
| S1 | ±π/2 | 0.3 | 1.0e-4 | model_2398 | `2026-06-30_22-11-54_flip360_f3_s1_halfpi_stage0` | model_2517 |
| S2 | ±3π/4 | 0.4 | 9.0e-5 | model_2517 | `2026-06-30_22-14-39_flip360_f3_s2_3qpi_stage0` | model_2636 |
| S3 | ±π | 0.5 | 8.0e-5 | model_2636 | `2026-06-30_22-19-46_flip360_f3_s3_fullpi_stage0` | model_2755 |

双目标 eval（total_steps=1500, use_stdw=False, seed=0, freq0.05）对比 `model_2846`：

| checkpoint | flip360 base | flip360 asym | ordinary base | ordinary asym | S3 训练 log_mse |
|---|---:|---:|---:|---:|---:|
| **2846（2 段课程，当前最佳）** | **2.0701** | **3.6606** | 0.2229 | 0.2266 | — |
| F3 model_2650（S3 最低训练 MSE） | 3.1214 | 3.5536 | （未测） | （未测） | 23.27 |
| F3 model_2755（S3 末段） | 3.2019 | 4.0127 | 0.2246 | 0.2755 | 39.57 |

**结论：F3 平滑爬坡未超过 2 段课程 `model_2846`，保留 2846 为最佳。**
根因：S3 进入满 ±π 后训练 `log_mse` 从段初 ~6 立即爆到 20–85 的混沌区间且全程不收敛
（见 `.../2026-06-30_22-19-46_..._stage0/mse_curve.jsonl`）。
F3 最佳候选 model_2650（S3 训练 MSE 最低 23.27）flip360 base 仍 3.12，比 2846 差约 50%；
2700/2750 训练 MSE 更高（39.9/41.0），无法在 flip360 base 上回补 ~50% 差距，故未续测。
asym 上 model_2650 (3.5536) 与 2846 (3.6606) 接近但 base 明显更差，非 Pareto 占优。
判定：平滑爬坡分段本身并未稳住满 ±π 段——瓶颈在满倒置区训练不收敛（与 F1/F2 的
驻留时间惩罚型角度极限一致），而非课程跳变幅度。

### Step F4：难区间容差整形（已据 F1/F2 修正：不放慢）

F1/F2 已证明放慢参考更差，故 F4 不再含“放慢”分支。确认触及物理上限后改为：
在难区间放宽跟踪容差（reward/参考整形），必要时让参考**更快**穿过难区间（更短驻留），
避免策略在“长时间倒置保持”这种不可达目标上学坏。

#### F4 执行与结果（2026-06-30，已完成，未胜出 → 确认架构边界）

实现：`easyuuv_env.py` 新增 `flip_tol_relax / flip_tol_band_lo / flip_tol_band_hi`
（默认 relax=0 零行为变更）；`_get_rewards` 内按 goal 倾角 `acos(R[2,2])` 落入
[120°,180°] 用 smoothstep 放宽 `rew_ang` 高斯宽度（÷(1+relax·w)）。
配置 `train_flip360_f4_tolshape.yaml`（relax=2.0），从 `model_2846` fine-tune 150 iter。

双目标 eval（freq0.05, total_steps=1500, use_stdw=False, seed=0）：

| checkpoint | flip360 base | flip360 asym | ordinary base | ordinary asym |
|---|---:|---:|---:|---:|
| **model_2846（最佳）** | **2.0701** | **3.6606** | 0.2229 | 0.2266 |
| F4 model_2850 | 2.0520 | 3.6747 | 0.2229 | 0.2268 |

**全面持平，无胜出也无遗忘**；F4 训练 MSE 同样在倒置区 14–42 混沌不收敛（与 F3 同构）。

#### 物理力矩边界判定（`workflows/analyze_flip360_torque_budget.py`）

| 项 | roll | pitch | vs 回正力矩(最坏) |
|---|---:|---:|---|
| 执行器硬上限(PWM=1.0) | 114.5 | 70.4 Nm | base 2.22 / asym 15.89 Nm |
| 名义工作点(S面,action_lim=0.15) | 17.0 | 10.4 Nm | asym pitch 10.4 < 15.89 |

- **不是执行器力矩受限**：硬上限超 asym 最坏回正 4–7x，满 ±π 原则物理可达。
- **名义工作点 pitch 权威 10.4 Nm < asym 最坏回正 15.9 Nm**：克服回正须压进 sigmoid
  饱和尾部（≈bang-bang），难学习。
- **主导边界 = 倒置区不稳定平衡 + 欧拉轴控制奇异 + 名义工作点可学习带宽不足**，
  均不可由 reward 整形/课程移除。

**结论（边界在哪里/是否无解）**：满 ±π 后空翻**非严格物理无解**（执行器有余量），
但**当前 S 面 + 欧拉角 + action_lim=0.15 架构下倒置区训练不收敛是架构性边界**，
F1–F4 全部止步于 base RMSE ≈0.85–0.91 rad（误差集中倒置区）。要突破需改控制架构
（SO(3)/几何姿态控制、增大 action_lim、重设计分配），而非继续 reward/课程调参。

### 执行顺序

F1 → F2 →（据结论）F3 → F4。**F1、F2、F3、F4 已全部完成并取证**：

- F1/F2：判定为驻留时间惩罚型角度极限，放慢参考无效。
- F3（平滑爬坡课程）：未超过 2 段课程 `model_2846`，根因满 ±π 段训练不收敛。
- F4（不放慢 + 难区间容差整形）：与 `model_2846` 全面持平，容差整形不改变倒置区不收敛；
  物理边界分析确认非执行器力矩受限，瓶颈是控制架构（S面+欧拉角+action_lim）。

**主线收敛结论**：`model_2846` 为 flip360 最佳 checkpoint 且已逼近当前架构上限；
进一步压低倒置区残差需更换控制架构，超出 reward/课程调参范畴。
保留项（OPR / drift-level projection / 无先验 domain transfer）仍待用户指引。

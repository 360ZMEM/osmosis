# IMU 噪声 / 延迟 / STDW 消融报告（2026-06-13）

## 1. 实验范围

本轮按“实用主义覆盖足够、不做完整 48-cell”的要求，完成两个小矩阵：

- C1：IMU 角速度噪声、D 项低通、观测/动作延迟，12 个去重 cell。
- C2：STDW 组件消融，14 行（其中 full baseline 复用 C1 clean baseline）。

固定条件：

- `task=EasyUUV-Direct-Parametric-v1`
- `checkpoint=logs/rsl_rl/easyuuv_parametric/2026-06-08_13-48-14_stage2/model_2398.pt`
- `workflow_config=workflows/configs/matrix_wave_medium_full.yaml`
- `seed=0`
- `total_steps=1500`
- `embodiment ∈ {base, asymmetric}`

产物：

- `.results/exp_imu_noise_delay_20260613/imu_noise_delay_summary_dedup.csv`
- `.results/exp_stdw_ablation_20260613/stdw_ablation_summary.csv`

## 2. C1：IMU 噪声、低通与延迟

### 2.1 结果表（final_mse）

| Case | Embodiment | ang_vel_noise | d_filter_tau | obs_delay | act_delay | final_mse | Δ vs clean |
|---|---|---:|---:|---:|---:|---:|---:|
| clean | base | 0.00 | 0.00 | 0 | 0 | 0.0725 | +0.0% |
| D low-pass only | base | 0.00 | 0.05 | 0 | 0 | 0.0725 | +0.0% |
| IMU gyro noise | base | 0.05 | 0.00 | 0 | 0 | 0.1383 | +90.8% |
| noise + D low-pass | base | 0.05 | 0.05 | 0 | 0 | 0.1385 | +91.0% |
| noise + obs delay 2 | base | 0.05 | 0.00 | 2 | 0 | 0.1634 | +125.4% |
| noise + act delay 2 | base | 0.05 | 0.00 | 0 | 2 | 0.1583 | +118.3% |
| clean | asymmetric | 0.00 | 0.00 | 0 | 0 | 0.5456 | +0.0% |
| D low-pass only | asymmetric | 0.00 | 0.05 | 0 | 0 | 0.5443 | -0.2% |
| IMU gyro noise | asymmetric | 0.05 | 0.00 | 0 | 0 | 0.4765 | -12.7% |
| noise + D low-pass | asymmetric | 0.05 | 0.05 | 0 | 0 | 0.5231 | -4.1% |
| noise + obs delay 2 | asymmetric | 0.05 | 0.00 | 2 | 0 | 0.7691 | +41.0% |
| noise + act delay 2 | asymmetric | 0.05 | 0.00 | 0 | 2 | 0.7607 | +39.4% |

### 2.2 结论

1. `ζ2` 的 D 项在当前 A3 中已经使用 body angular velocity，因此不需要“更换为角速度”；部署问题应表述为“角速度噪声下是否需要低层滤波”。
2. `d_filter_tau=0.05` 在本轮没有带来明确收益：
   - base：噪声下 `0.1383 -> 0.1385`，基本不变；
   - asymmetric：噪声下 `0.4765 -> 0.5231`，反而变差。
3. IMU 级角速度噪声对 base 明显有害（final_mse +90.8%），说明实物上必须重视 IMU 标定、量纲、采样周期和低层去噪。
4. 在默认有害 drift 的 asymmetric 上，噪声使 final_mse 从 `0.5456` 降到 `0.4765`，这更像噪声打散了错误 drift/慢环耦合，不应解读为“噪声有益”。实际部署仍应优先使用 router/probe 避免错误 drift。
5. 2 step 观测或动作延迟显著恶化：
   - base：+118% 到 +125%；
   - asymmetric：+39% 到 +41%。
   实物部署需要尽量降低从 IMU 到策略、策略到执行器的链路延迟；若不可避免，应重新做延迟域随机化或增加低层补偿。

## 3. C2：STDW 组件消融

### 3.1 结果表（final_mse）

| Variant | Base final_mse | Δ vs full | Asymmetric final_mse | Δ vs full |
|---|---:|---:|---:|---:|
| full STDW | 0.0725 | +0.0% | 0.5456 | +0.0% |
| STDW off | 0.2262 | +212.0% | 0.3147 | -42.3% |
| no slow loop | 0.2262 | +212.0% | 0.3147 | -42.3% |
| no Lyapunov fence | 0.0724 | -0.1% | 0.5395 | -1.1% |
| no pseudo-action | 0.0721 | -0.6% | 0.5284 | -3.2% |
| no quantile filter | 0.2010 | +177.2% | 0.3364 | -38.4% |
| no trigger gate | 0.2702 | +272.7% | 0.5456 | +0.0% |

### 3.2 结论

1. base 上 full STDW 明显优于 STDW off，说明默认 STDW 机制对常规 embodiment 有效。
2. asymmetric 上 full STDW 反而差于 STDW off，这与此前 P2 诊断一致：问题根因是默认 drift 方向错误，而不是慢环损失本身。
3. `no_slow_loop` 与 `STDW off` 数值一致，说明这一路径只保留了物理 drift/fast loop，不再产生在线策略更新；它是一个清晰的部署回退基线。
4. Lyapunov fence 与 pseudo-action 单独关闭时影响很小，说明在本 small matrix 下它们不是主导项；但这不代表可删除，因为它们的价值主要出现在更复杂/高风险样本过滤中。
5. `no_quantile_filter` 和 `no_trigger_gate` 在 base 上显著变差，说明滤波和触发门对常规场景的稳态抑制很重要。
6. asymmetric 上移除 quantile filter 或 slow loop 反而改善，本质仍是默认 drift 方向错误被削弱或不再在线放大。正确路线不是删除组件，而是使用 offset router / micro-probe 选择保守 baseline 或 corrective drift。

## 4. 对实物部署的直接建议

1. `ζ2` 不需要再“替换为角速度”；代码已经这样做。
2. 当前 `d_filter_tau=0.05` 不能作为默认启用项。建议实物默认 `d_filter_tau=0.0`，仅在真实 IMU 噪声确认很高且重新验证后启用。
3. IMU 角速度质量是关键风险。上板时先记录静止陀螺噪声和采样周期，再决定是否做低层滤波。
4. 延迟比低通更危险。应优先压低串口/推理/执行器链路延迟；2 个控制步延迟已经显著抬高 final_mse。
5. STDW 在实物上必须保守启动：baseline 稳定后再 micro-probe；证据不足选 `baseline`；不要默认施加固定 `+x` drift 到 asymmetric 机型。


# Research Analysis & Improvement Roadmap

本文从科研角度梳理 EasyUUV-STDW 当前的实证结论、设计中的张力，以及下一步可行的改进方向。
**所有数值均来自 [`_legacy/REPORT_full_matrix_20260606.md`](../archive/REPORT_full_matrix_20260606.md) 的 48-cell 全矩阵评估**
（3 wave × 4 emb × 2 tune × 2 stdw × 1 seed，单 seed = 0，每 cell 1500 step / 25 s 仿真）。

---

## 1. 实证结论速览（含单位）

### 1.1 STDW 主效应

跨 wave × tune 配对的 `final_mse` 平均变化（单位 m²；负号表示 STDW on 改善）：

| Wave | 全 8 组合（含 heavy_moderate） | 6 组合（去 heavy_moderate） |
|---|---:|---:|
| calm (hs=0.3 m, fp=0.18 Hz) | −19.0% | **−38.6%** |
| medium (hs=0.8, fp=0.13) | −19.6% | **−38.7%** |
| storm (hs=1.5, fp=0.10) | −23.4% | **−38.6%** |

**STDW 在 base / long_body / asymmetric 三个 embodiment 上一致改善 final_mse 28%–47%；
跨 wave 三档收益几乎不变（−38.6 ± 0.1%）**。这是本仓库最重要的可复现结论。

### 1.2 heavy_moderate 反向异常

| Wave | tune | fmse_off (m²) | fmse_on (m²) | Δ% |
|---|---|---:|---:|---:|
| calm | full | 3.480 | 6.093 | **+75.1%** |
| medium | full | 3.499 | 5.613 | **+60.4%** |
| storm | full | 3.466 | 4.521 | **+30.4%** |

**heavy_moderate 在 STDW off 已是 4 个 embodiment 中最低 fmse**，STDW on 反而劣化。
直观解释：STDW 把 COB 推向 0.05 m drift target，对**已偏离最优配重**的 emb 是修正，
对**接近最优配重**的 emb 反而是干扰。

> **结论**：STDW 不是"无差别加速器"，部署前应先 STDW off 短跑判定基线，再决定是否启用。

### 1.3 整定（tune）维度

跨 wave × emb 平均 `fmse_off`（裸控制基线，不受 STDW 干扰）：

| Tune mode | mean fmse_off (m²) |
|---|---:|
| identity（旁路 4 机制） | 6.000 |
| full（PE+死区+LPF+β=0.2） | **6.111** |

差异 **+1.9% 在统计噪声范围内**。说明 1500 iter 训练出来的 8 维 meta-control policy
**整定头还没真正学到有意义的 ζ 调节策略**——`a_gain` 输出基本是噪声。

### 1.4 Embodiment 维度

平均 `fmse_off`：

| Embodiment | mean fmse_off (m²) | 相对 base |
|---|---:|---:|
| base | 6.610 | 1.00× |
| long_body | 7.309 | 1.10× |
| heavy_moderate | **3.534** | **0.53×** |
| asymmetric | 6.769 | 1.02× |

`long_body`（转动惯量 [0.1, 2.5, 2.5] vs base [0.37, 0.97, 1.19]）最难控；
`asymmetric`（com_to_cob xy 偏移 0.05/0.05 m）几乎与 base 持平，说明 1500 iter 已学到一定不对称鲁棒性。

---

## 2. 设计中的张力（trade-offs）

### T1. 慢环锚定 vs 学习速率

`--lambda_reg 1e-3` 与 `--g_C_lr 5e-5` 的比例决定 STDW 学到多大幅度的局部修正。
当前默认偏保守：在 base 上 −38% fmse 但训练曲线后半段平台明显，
说明 anchor 太强、policy 已经被锁定在 init 附近。

### T2. PE 注入 vs 噪声放大

`pe_amp=0.05, pe_freq=0.5 Hz` 设计目的是在 ζ 上注入持续激励以辅助系统辨识，
但当前 1500 iter 的 a_gain head 还没学到响应 PE，故 PE 仅作为 + 5% 振幅噪声出现。
在 heavy_moderate + STDW on + full tune 的组合下，PE 的扰动会与 STDW drift 推力同向叠加，导致 +75% 劣化。

### T3. Bounded Safeguard 的 β 上限

`gain_beta=0.2` 给出 ±20% ζ_nom 的 hard bound，是物理稳定性硬保证。
但在大风浪（hs=1.5 m）下 ±20% 似乎不够：`storm × full × heavy_moderate` 仍劣化 +30%。
一种思路是让 β 随 wave hs 自适应放大，但这会让 Lyapunov 稳定证明失效。

---

## 3. 改进方向（按优先级）

### P1. 训练时长 ≥ 3000 iter（必做）

当前 1500 iter 只够 4-D ctrl head 收敛；a_gain head 还在 noise 区。
3000 iter 起步能让 8 维 policy 把整定头训出意义，full vs identity 应该至少有 3-5% 净改善。

### P2. 引入 STDW gating（生产部署用）

观察到 STDW 是"修偏移"机制，对接近最优的 emb 反向。
建议：先用 200 step 短跑 STDW off 估出 `fmse_short`；若低于 4.0 m² 阈值则关闭 STDW，
否则启用。这能避免 heavy_moderate 上 +75% 的劣化。

### P3. Wave-conditioned β

`gain_beta` 当前是常数 0.2。让 β = 0.2 + 0.05·tanh(hs)，
在 storm 下扩到 0.25，在 calm 下保持 0.2。
需重做 Lyapunov 稳定证明（用更紧的 LMI）。

### P4. 多 seed 报告

当前 48-cell 单 seed，置信区间无法估计。
增加到 3 seed × 48 cell = 144 trial，wall time 约 70 min，可接受。
这之后才能正式在论文里写"显著改善"。

### P5. Eval 子模块对接 ROS / micro-ROS

`eval/policy_loader.py` 现已支持 ONNX。下一步把 `obs_from_state` 包成
ROS topic 订阅 + `act` 发布到 thruster，写一个端到端 ROS demo
（参见 [`EVAL_SOP.md`](../guide/EVAL_SOP.md) §4）。

### P6. JONSWAP 时变（沿任务周期变化海况）

当前一个 episode 内 hs/fp 不变。可让 `apply_runtime_domain_shift` 在每个 reset 时
按高斯随机抽取 hs ∈ [0.3, 1.5]，研究 STDW 是否能学到 wave-aware adaptive。

---

## 4. 数据制品索引（论文复现用）

| 文件 | 含义 |
|---|---|
| `_legacy/REPORT_full_matrix_20260606.md` | 48-cell 完整报告（核心实证） |
| `_legacy/REPORT_full_train_72cell_stdw_20260605.md` | 72-cell 早期 sweep（含训练曲线） |
| `.results/full_matrix_<date>/full_matrix.csv` | 48 trial 原始指标 |
| `.results/full_matrix_<date>/stdw_pairwise.csv` | 24 行 STDW off↔on 配对 |
| `compact_log.jsonl` + `mse_curve.jsonl` | 训练 metric 极简日志 |

---

## 5. 复现命令（论文级 sanity check）

```bash
# 1. 跑当前 A3 stage2 全矩阵
bash custom_workflows/run_with_isaac_env.sh workflows/sweep_full_matrix.py \
    --policy_path logs/rsl_rl/easyuuv_parametric/2026-06-08_13-48-14_stage2/model_2398.pt \
    --out_root .results/sweep_a3_stage2_$(date +%Y%m%d_%H%M%S) \
    --total_steps 1500 --seeds 0

# 2. 聚合
bash custom_workflows/run_with_isaac_env.sh workflows/tools/aggregate_full_matrix.py \
    --matrix_dir .results/sweep_a3_stage2_<timestamp>

# 3. 关键三个数（从 stdw_pairwise.csv 读出来）
#    base mean over wave×tune:           Δfmse ≈ −67.8%
#    long_body mean over wave×tune:      Δfmse ≈ −65.2%
#    asymmetric mean over wave×tune:     Δfmse ≈ +158%（异常点必须复现）
```

如果三个数偏差超过 ±5%，先检查 [`ERROR_CASES.md`](../guide/ERROR_CASES.md) §1（JONSWAP 注入）和
§3（STDW 双关基线）。

完整命令契约以 [`COMMAND_CONTRACT.md`](../guide/COMMAND_CONTRACT.md) 为准；旧
`model_1499.pt` 复现路径仅保留在 `docs/archive/` 作为历史对照。

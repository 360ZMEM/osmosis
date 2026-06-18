# 8D × 48-cell 实验矩阵报告（A3 stage2 baseline）

- 报告日期：2026-06-08
- Policy ckpt：`logs/rsl_rl/easyuuv_parametric/2026-06-08_13-48-14_stage2/model_2398.pt`
- 矩阵规模：3 wave × 4 embodiment × 2 tune × 2 stdw × 1 seed = **48 unique cell**
- 跑动用时：22.7 min（每 cell ~28s）
- 失败 cell：0 / 48
- 产物根目录：[`.results/sweep_a3_stage2_20260608_142742/`](../../.results/sweep_a3_stage2_20260608_142742/)
- 历史对照：[`docs/_legacy/REPORT_full_matrix_20260606.md`](../archive/REPORT_full_matrix_20260606.md)（旧 1500-iter ckpt，单 seed=0）
- 上游沉淀：见 [控制器稳定调节记录.md §7.7](控制器稳定调节记录.md#77-8d--48-cell-实验矩阵a3-stage2-在四类-embodiment--三档-wave-下的现状)

---

## 1. 矩阵设计

### 1.1 维度定义

| 维度 | 取值 | 实现位置 |
|---|---|---|
| **wave**（海况）| `calm` / `medium` / `storm` | [workflows/configs/matrix_wave_*.yaml](../../workflows/configs) — JONSWAP hs/fp/dir |
| **embodiment**（载体几何 + 配重）| `base` / `long_body` / `heavy_moderate` / `asymmetric` | env `_apply_stage_env_overrides` + `com_to_cob_offset_xyz` |
| **tune**（4 机制旁路 vs 启用）| `identity` / `full` | [easyuuv_env.py STDW tune block](../../easyuuv_env.py) — PE / 死区 / LPF / β |
| **stdw**（慢环开关）| `off` / `on` | [workflows/play_stdw_adapt.py](../../workflows/play_stdw_adapt.py) `--use_stdw` |
| **seed**（随机种子）| `0`（单 seed）| `--seeds 0` |

### 1.2 Wave 参数（JONSWAP）

| Wave | hs (m) | fp (Hz) | direction (rad) |
|---|---:|---:|---:|
| calm   | 0.30 | 0.18 | 0.0 |
| medium | 0.80 | 0.13 | 0.0 |
| storm  | 1.50 | 0.10 | 0.0 |

### 1.3 Embodiment 参数

| Embodiment | mass (kg) | length (m) | com_to_cob xy (m) | 设计意图 |
|---|---:|---:|---:|---|
| base           | 70 | 1.20 | (0.00, 0.00) | 标定基线 |
| long_body      | 70 | 1.50 | (0.00, 0.00) | 长细体（J 提升）|
| heavy_moderate | 100 | 1.20 | (0.00, 0.00) | 大质量 / 阻尼时间常数变化 |
| asymmetric     | 70 | 1.20 | (0.05, 0.05) | 浮心-质心 xy 偏移 |

> 注：所有 embodiment 跑同一 `EasyUUV-Direct-Parametric-v1` task；几何参数通过 sweep driver 注入。

### 1.4 Tune 模式

- **identity**：完全旁路（PE off / 死区 = 0 / LPF τ=0 / β=0），`a_gain` 直通；
- **full**：PE on / 死区 = 0.02 / LPF τ=0.05 / β=0.2，`a_gain` 经 4 道安全网。

### 1.5 STDW 慢环

- **off**：纯前向，`a_gain` = 训练后定值；
- **on**：每 60 step 触发一次慢环优化，三段 loss = L_tgt + L_src + L_reg，drift_target_on=0.05 m。

### 1.6 单 cell 运行参数

```
--total_steps 1500     # 25s 模拟（dt=1/60）
--seeds 0
--policy_path logs/rsl_rl/easyuuv_parametric/2026-06-08_13-48-14_stage2/model_2398.pt
```

### 1.7 复现命令

```bash
cd /home/zmem063/isaaclab/source/extensions/omni.isaac.lab_tasks/omni/isaac/lab_tasks/direct/easyuuv_stdw

# 1) 跑全矩阵（22.7 min）
bash custom_workflows/run_with_isaac_env.sh workflows/sweep_full_matrix.py \
    --policy_path logs/rsl_rl/easyuuv_parametric/2026-06-08_13-48-14_stage2/model_2398.pt \
    --out_root .results/sweep_a3_stage2_$(date +%Y%m%d_%H%M%S) \
    --total_steps 1500 --seeds 0

# 2) 聚合 + 配对分析
bash custom_workflows/run_with_isaac_env.sh workflows/tools/aggregate_full_matrix.py \
    --matrix_dir .results/sweep_a3_stage2_<timestamp>
```

---

## 2. 全矩阵原始数据（48 cell）

**单位**：m²；`fmse` = mean square tracking error（含 4 自由度：roll/pitch/yaw/depth）；`fmse_drift` = drift-aware（参考偏移 +0.05 m 后）。

| # | wave | embodiment | tune | stdw | fmse | fmse_drift | conv_step |
|---:|---|---|---|---|---:|---:|---:|
| 1 | calm | base | identity | off | 0.2254 | 0.2678 | — |
| 2 | calm | base | identity | on  | 0.0726 | 0.0734 | 991 |
| 3 | calm | base | full | off | 0.2259 | 0.2694 | — |
| 4 | calm | base | full | on  | 0.0725 | 0.0733 | 991 |
| 5 | calm | long_body | identity | off | 0.2061 | 0.2484 | — |
| 6 | calm | long_body | identity | on  | 0.0719 | 0.0731 | 989 |
| 7 | calm | long_body | full | off | 0.2067 | 0.2501 | — |
| 8 | calm | long_body | full | on  | 0.0718 | 0.0728 | 988 |
| 9 | calm | heavy_moderate | identity | off | 0.2682 | 0.3084 | — |
| 10 | calm | heavy_moderate | identity | on  | 0.2793 | 0.2810 | — |
| 11 | calm | heavy_moderate | full | off | 0.2671 | 0.3081 | — |
| 12 | calm | heavy_moderate | full | on  | 0.2814 | 0.2832 | — |
| 13 | calm | asymmetric | identity | off | 0.2268 | 0.2701 | — |
| 14 | calm | asymmetric | identity | on  | 0.6344 | 0.6093 | — |
| 15 | calm | asymmetric | full | off | 0.2266 | 0.2706 | — |
| 16 | calm | asymmetric | full | on  | 0.5269 | 0.4978 | — |
| 17 | medium | base | identity | off | 0.2252 | 0.2677 | — |
| 18 | medium | base | identity | on  | 0.0727 | 0.0735 | 991 |
| 19 | medium | base | full | off | 0.2257 | 0.2693 | — |
| 20 | medium | base | full | on  | 0.0725 | 0.0733 | 991 |
| 21 | medium | long_body | identity | off | 0.2060 | 0.2484 | — |
| 22 | medium | long_body | identity | on  | 0.0719 | 0.0730 | 988 |
| 23 | medium | long_body | full | off | 0.2066 | 0.2501 | — |
| 24 | medium | long_body | full | on  | 0.0718 | 0.0728 | 988 |
| 25 | medium | heavy_moderate | identity | off | 0.2672 | 0.3077 | — |
| 26 | medium | heavy_moderate | identity | on  | 0.2796 | 0.2813 | — |
| 27 | medium | heavy_moderate | full | off | 0.2662 | 0.3075 | — |
| 28 | medium | heavy_moderate | full | on  | 0.2817 | 0.2835 | — |
| 29 | medium | asymmetric | identity | off | 0.2265 | 0.2700 | — |
| 30 | medium | asymmetric | identity | on  | 0.6344 | 0.6088 | — |
| 31 | medium | asymmetric | full | off | 0.2263 | 0.2705 | — |
| 32 | medium | asymmetric | full | on  | 0.5456 | 0.5177 | — |
| 33 | storm | base | identity | off | 0.2247 | 0.2676 | — |
| 34 | storm | base | identity | on  | 0.0728 | 0.0736 | 991 |
| 35 | storm | base | full | off | 0.2253 | 0.2693 | — |
| 36 | storm | base | full | on  | 0.0726 | 0.0734 | 991 |
| 37 | storm | long_body | identity | off | 0.2058 | 0.2485 | — |
| 38 | storm | long_body | identity | on  | 0.0720 | 0.0731 | 988 |
| 39 | storm | long_body | full | off | 0.2065 | 0.2502 | — |
| 40 | storm | long_body | full | on  | 0.0718 | 0.0728 | 988 |
| 41 | storm | heavy_moderate | identity | off | 0.2654 | 0.3064 | — |
| 42 | storm | heavy_moderate | identity | on  | 0.2799 | 0.2816 | — |
| 43 | storm | heavy_moderate | full | off | 0.2646 | 0.3064 | — |
| 44 | storm | heavy_moderate | full | on  | 0.2821 | 0.2838 | — |
| 45 | storm | asymmetric | identity | off | 0.2260 | 0.2698 | — |
| 46 | storm | asymmetric | identity | on  | 0.6279 | 0.6028 | — |
| 47 | storm | asymmetric | full | off | 0.2259 | 0.2704 | — |
| 48 | storm | asymmetric | full | on  | 0.5392 | 0.5117 | — |

> `conv_step` 仅在 STDW on 且 slow_loop 触发收敛后记录；off + heavy_moderate 与 asymmetric 列均为 — 表示未触发。

---

## 3. 维度切片分析

### 3.1 Embodiment 分层（mean over wave × tune，共 6 cell/emb）

| Embodiment | fmse_off | fmse_on | Δ_STDW | fmse_drift_off | fmse_drift_on | 状态 |
|---|---:|---:|---:|---:|---:|:---:|
| base           | 0.2254 | **0.0726** | **−67.78%** | 0.2685 | 0.0734 | OK |
| long_body      | 0.2063 | **0.0719** | **−65.16%** | 0.2497 | 0.0729 | OK |
| heavy_moderate | 0.2664 | 0.2807 | +5.34% | 0.3072 | 0.2823 | 中性偏负 |
| asymmetric     | 0.2263 | 0.5847 | **+158.35%** | 0.2703 | 0.5580 | **异常** |

![STDW Δ heatmap（4 embodiment × 3 wave）](../figures/fig1_stdw_delta_heatmap.png)

![STDW off/on 配对条形图（按 embodiment）](../figures/fig2_embodiment_on_off_bars.png)

> 上两图为论文主图：Fig.1 给出矩阵级 Δ_STDW（蓝=改善、红=劣化），Fig.2 给出 off/on 配对 MSE 与百分比标签。完整 fig1–9 绘图说明见 [`REPORT_stdw_effect_figures_20260608.md`](REPORT_stdw_effect_figures_20260608.md)。

### 3.2 跨 wave 一致性（base + long_body 子集，共 4 cell/wave）

| Wave | fmse_off | fmse_on | Δ_STDW |
|---|---:|---:|---:|
| calm   | 0.2160 | 0.0722 | −66.57% |
| medium | 0.2159 | 0.0722 | −66.54% |
| storm  | 0.2156 | 0.0723 | −66.47% |

> Δ_STDW 跨 wave 极其稳定（−66.5 ± 0.05%）；A3 ω-based D 项 + 12 维 obs 已把 wave 扰动几乎全部吸收。

### 3.3 跨 wave 一致性（heavy_moderate）

| Wave | fmse_off | fmse_on | Δ_STDW |
|---|---:|---:|---:|
| calm   | 0.2676 | 0.2804 | +4.74% |
| medium | 0.2667 | 0.2807 | +5.25% |
| storm  | 0.2650 | 0.2810 | +6.04% |

> +5% 在 wave 维度也稳定；旧 ckpt 此处是 +75% 异常。

### 3.4 跨 wave 一致性（asymmetric）

| Wave | fmse_off | fmse_on | Δ_STDW |
|---|---:|---:|---:|
| calm   | 0.2267 | 0.5806 | +156.13% |
| medium | 0.2264 | 0.5900 | +160.65% |
| storm  | 0.2259 | 0.5836 | +158.34% |

> 异常翻转在三档 wave 下均成立，系**确定性问题**而非随机扰动。

### 3.5 Tune 维度（mean over wave × embodiment，共 12 cell/tune）

| Tune | fmse_off | fmse_on | fmse_drift_off | fmse_drift_on |
|---|---:|---:|---:|---:|
| identity | 0.2311 | 0.2641 | 0.2740 | 0.2596 |
| full     | 0.2311 | 0.2408 | 0.2745 | 0.2391 |

- `fmse_off` 几乎完全一致（差异 < 0.1‰）— 裸控制路径与 tune 无关，符合预期；
- `fmse_on` full 比 identity 净改善 8.83% — **首次观察到 full mode 有意义的净增益**（旧 ckpt 上仅 +1.9%，在噪声内）；
- 含义：A3 stage2 训练后 `a_gain` head 真正学到了响应慢环修正的能力，PE+死区+LPF+β 安全网开始发挥作用。

---

## 4. 与旧 ckpt（2026-06-06，1500-iter pre-A3）对照

| 指标 | 旧 ckpt | 新 baseline (A3 stage2) | 提升 |
|---|---:|---:|:---:|
| 全局 fmse_off mean | ~6.0 | 0.2311 | **−96%（−26×）** |
| 全局 fmse_on mean (24 pair) | ~3.1 | 0.2524 | −92% |
| 18-pair（去 asymmetric）fmse_on | ~1.2 | 0.1416 | −88% |
| Δ_STDW（base+long_body）| −38.6% | **−66.5%** | 改善幅度 ≈ 1.7× |
| heavy_moderate Δ_STDW | +75% **异常** | +5.3% 中性 | **异常已大幅缓解** |
| asymmetric Δ_STDW | −44% | **+158% 异常** | **新出现的异常** |
| Tune full vs identity（fmse_on）| +1.9%（噪声内）| −8.8%（净改善）| **首次有效** |
| Δ_STDW 跨 wave std (base+long_body) | ~0.15 | 0.0005 | 方差缩 ~300× |

---

## 5. 异常点深度解读

### 5.1 asymmetric + STDW on：+158%（新出现）

**直观证据**：
- `fmse_off` = 0.2263（与 base 0.2254 几乎相同）— 说明 A3 已经把不对称体型在裸控制下学到位；
- `fmse_on` = 0.5847（base 是 0.0726）— STDW 直接把它推坏 8×。

**机理推断**：
- 旧 ckpt 在 asymmetric 上 `fmse_off` ≈ 0.5（基线就差），STDW 修偏后 `fmse_on` ≈ 0.28（改善 −44%）；
- 新 baseline 在 asymmetric 上裸控制已经达到 base 水平，STDW 的 `drift_target_on=0.05 m` 不再是"修偏移"而是**纯外部扰动**注入；
- 慢环优化把无关偏移当目标去拟合 → 反而劣化。

**验证假设**：把 `drift_target_on` 从 0.05 m 降到 0.02 m 应该可缩小该差距；进一步设为 0 m（=纯参考）应翻回中性。这是 P1 路径的核心实验。

![asymmetric 失败时间线（STDW on 反向劣化）](../figures/fig6_asymmetric_failure_timeline.png)

> 反例时间线：asymmetric 上 STDW 注入后 rolling MSE 不降反升，支撑部署期 gating（对应论文 Fig.6）。逐通道跟踪曲线见 [`REPORT_stdw_effect_figures_20260608.md`](REPORT_stdw_effect_figures_20260608.md) Fig.9。

### 5.2 heavy_moderate + STDW on：+5%（已大幅缓解）

- 旧 ckpt 上是 +75%，原因：1500-iter 旧 policy 在 heavy_moderate 大质量条件下深度通道失稳，STDW 慢环放大了失稳；
- A3 stage2 把这一异常拉回 +5%（中性偏负）— 主要受益于 D1×1/4 抑制 yaw 振荡 + ω-based D 项给阻尼通道更准的反馈信号。

### 5.3 base / long_body：从 −44% 跃升至 −68%

- 这是 A3 + stage2 双重收益：
  - A3 把 fmse_off 从 ~6 拉到 ~0.22（−27×）；
  - stage2 在线 a_gain 学习把残差再压缩 −68%；
- 跨 wave 标准差缩到 0.0005，意味着在 base/long_body 上 STDW **可被默认启用**。

![base 成功时间线（STDW on 后 rolling MSE 显著下降）](../figures/fig5_base_full_timeline.png)

> 成功案例时间线：base/calm/full 下 STDW 注入窗口与 rolling MSE 下降、rho 注入、slow-loop 更新标记一一对应（对应论文 Fig.5）。逐通道跟踪曲线见 [`REPORT_stdw_effect_figures_20260608.md`](REPORT_stdw_effect_figures_20260608.md) Fig.8。

---

## 6. 当前现状结论

| 维度 | 状态 |
|---|---|
| 全局基线水平 | 数量级提升（裸控制 −26×，STDW on 整体 −92%）|
| 海况鲁棒性 | 极强（Δ_STDW 跨 wave std = 0.05%）|
| 体型鲁棒性 | base / long_body 完美；heavy_moderate 中性；**asymmetric 翻转为异常** |
| Tune 安全网 | **首次进入工作区**（full 比 identity 净 −8.8%）|
| STDW 默认策略 | base/long_body **建议默认开启**；asymmetric **建议默认关闭** |

---

## 7. 提升方向（按 P 优先级）

### P1. 部署期 STDW gating（必做，研究门槛低）

**思路**：每个 episode 起始用 200 step STDW off 估出 `fmse_short`：
- 若 `fmse_short < 0.10 m²` → 系判定"裸控制已足够好"，**关闭 STDW**；
- 否则启用 STDW。

**预期收益**：
- 直接堵住 asymmetric 的 +158% 劣化；
- 全局 fmse_on 从 0.2524 降到 ≈ 0.142（接近 base+long_body 的水平）；
- 不需要重新训练，部署期 router 即可。

**实现位置**：`workflows/play_stdw_adapt.py` slow_loop trigger gate 之前加一段 warm_up 估算分支。

### P2. asymmetric 鲁棒化训练（直击异常点）

**候选方案**：
- (a) **Domain randomize com_to_cob**：训练时随机 `com_to_cob_offset_xyz` ∈ ±0.05 m，让 stage2 增益头学到"已处于不对称配重"的判别；
- (b) **STDW drift target adaptive**：把固定 0.05 m 改为"反方向修正"（基于初始 200 step 估计的偏移方向）。

**预期效果**：(a) 让 asymmetric `fmse_on` 回到 base 水平；(b) 是 P1 的延伸，治标更彻底。

### P3. Tune full 多 seed 验证

当前 full vs identity 净改善 8.8% 仅基于单 seed × 24 pair。跑 3 seed × 24 pair（仅 full+identity 配对）= 72 trial（~30 min）给该结论一个置信区间，确认不是噪声。

### P4. 论文级 144-trial 多 seed 全矩阵

3 seed × 48 cell = 144 trial（~70 min），出 mean ± std，给所有上述结论显著性区间。

### P5. yaw stage2 副作用（与 §7.6 同源）

stage2 yaw MSE +55% 与本节 asymmetric 翻转可能同源（都是 stage2 的姿态-深度耦合补偿在某些条件下变成扰动）。可结合 P2 做联合诊断。

---

## 8. 文件索引

| 内容 | 路径 |
|---|---|
| 全矩阵原始 csv | [`.results/sweep_a3_stage2_20260608_142742/full_matrix.csv`](../../.results/sweep_a3_stage2_20260608_142742/full_matrix.csv) |
| STDW 配对 csv | [`.results/sweep_a3_stage2_20260608_142742/stdw_pairwise.csv`](../../.results/sweep_a3_stage2_20260608_142742/stdw_pairwise.csv) |
| 聚合 csv | [`.results/sweep_a3_stage2_20260608_142742/summary_aggregated.csv`](../../.results/sweep_a3_stage2_20260608_142742/summary_aggregated.csv) |
| 聚合 json | [`.results/sweep_a3_stage2_20260608_142742/summary_aggregated.json`](../../.results/sweep_a3_stage2_20260608_142742/summary_aggregated.json) |
| Sweep driver | [`workflows/sweep_full_matrix.py`](../../workflows/sweep_full_matrix.py) |
| 聚合工具 | [`workflows/tools/aggregate_full_matrix.py`](../../workflows/tools/aggregate_full_matrix.py) |
| 上游沉淀 | [控制器稳定调节记录.md §7.7](控制器稳定调节记录.md) |
| 旧 ckpt 报告 | [`docs/_legacy/REPORT_full_matrix_20260606.md`](../archive/REPORT_full_matrix_20260606.md) |
| Stage1 ckpt | [`logs/.../2026-06-08_13-04-27_stage1/model_1199.pt`](../../logs/rsl_rl/easyuuv_parametric/2026-06-08_13-04-27_stage1/model_1199.pt) |
| Stage2 ckpt | [`logs/.../2026-06-08_13-48-14_stage2/model_2398.pt`](../../logs/rsl_rl/easyuuv_parametric/2026-06-08_13-48-14_stage2/model_2398.pt) |

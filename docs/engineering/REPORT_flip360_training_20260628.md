# Flip360 训练报告 — 2026-06-28

## 背景

OPR / drift-level projection / 无先验 domain transfer 等线索保留在
`PLAN_stdw_hard_constraints_20260628.md`，本条线切换到独立的 flip360 训练。

## 训练

最初 resume 失败：RSL-RL 默认会加载旧的 optimizer state，但当前分阶段训练（stage-wise）使用了不同的
optimizer parameter group。`workflows/train_meta.py` 现支持：

```bash
--resume_load_optimizer False
```

它只加载 policy / normalizer，不加载 optimizer state，这是跨训练范式做 fine-tune 时的正确行为。

成功的短程 fine-tune：

```bash
bash custom_workflows/run_with_isaac_env.sh workflows/train_meta.py \
  --headless \
  --task EasyUUV-Direct-Parametric-v1 \
  --experiment_name easyuuv_parametric \
  --run_name flip360_ft200_from_a3 \
  --num_envs 512 \
  --max_iterations 200 \
  --reference_mode flip360_sine \
  --ref_sine_amp 3.1415926,3.1415926,0.0 \
  --ref_sine_freq 0.05,0.05,0.0 \
  --resume True \
  --resume_load_optimizer False \
  --load_run 2026-06-08_13-48-14_stage2 \
  --checkpoint model_2398.pt
```

运行目录：

```text
logs/rsl_rl/easyuuv_parametric/2026-06-28_23-07-38_flip360_ft200_from_a3_stage1/
```

保存的 checkpoint：

```text
model_2400.pt
model_2450.pt
model_2500.pt
model_2550.pt
model_2597.pt
```

## Flip360 Checkpoint 扫描

评估配置：

```text
workflows/configs/pressure_flip360_medium_full.yaml
total_steps = 1500
use_stdw = False
seed = 0
```

结果：

| checkpoint | base final_mse | asymmetric final_mse |
|---|---:|---:|
| A3 stage2 model_2398 | 3.366683 | 5.895537 |
| model_2400 | 5.335180 | 10.698211 |
| model_2450 | 4.578107 | 10.890509 |
| model_2500 | 2.789910 | 3.112985 |
| model_2550 | 2.237671 | 2.184740 |
| model_2597 | 10.671348 | 4.500114 |

> 注意：上述数值在 Round 2 被发现是“静态大姿态保持”假象（见下文“关键修复”），不能作为连续后空翻结论。

## Ordinary Medium 回归

评估配置：

```text
workflows/configs/matrix_wave_medium_full.yaml
total_steps = 1500
use_stdw = False
seed = 0
```

结果：

| policy | base final_mse | asymmetric final_mse |
|---|---:|---:|
| A3 stage2 model_2398 | 0.226182 | 0.226305 |
| flip360 model_2550 | 0.225773 | 0.286082 |

解读：

- `model_2550` 不伤 ordinary medium/base。
- `model_2550` 伤 ordinary medium/asymmetric 约 `+26.4%`。

## 决策点（Round 1）

短程 flip360 课程证明训练可以改善 360 跟踪，但纯满幅度 fine-tune 不稳定，并造成 asymmetric ordinary 任务遗忘。

后续选项：

1. 把 `model_2550.pt` 只当作 flip360 专用 checkpoint。
2. 增加课程 / 混合任务训练：
   - 阶段 A：小幅度 sine sweep；
   - 阶段 B：flip360 幅度爬升；
   - 阶段 C：混合 `flip360_sine + sine_sweep/step` 回放以保住 ordinary asymmetric。
3. 增加 checkpoint 选择 / early stopping，使用双目标验证集：
   - flip360 base/asymmetric；
   - ordinary medium base/asymmetric。

建议下一步：实现一个小的混合课程训练入口或配置，然后重跑同一套 checkpoint 扫描。

---

# Flip360 调优 Round 2 — 2026-06-29

## 关键修复：flip360 参考此前不是连续的

原 `flip360_sine` 只在 `_reset_goal` 时设置一次目标姿态，控制步内从不推进，
所以早期“flip360”评估实际是**静态大姿态保持**，并非连续 360 后空翻。这使 Round 1
`model_2550` 的数值（base 2.24 / asym 2.18）作废。`easyuuv_env._update_reference`
现已对 `flip360_sine` 与新增的 `mixed_sine_flip360` 模式每步推进 roll/pitch 参考。

新增代码（不破坏命令契约）：
- `easyuuv_env.py`：逐步连续参考 + `mixed_sine_flip360`（每个 env 按 `ref_mix_flip_prob` 抽 flip 或 ordinary sine）。
- `workflows/train_meta.py`：`--meta_stage 0` fine-tune 模式（不做梯度隔离 / 不覆盖 env），以及 `--ref_mix_*` 透传。
- `workflows/sweep_stdw_safety_pressure.py`：窄 profile `flip360_only` / `ordinary_only`，用于单 checkpoint 快速筛选。

## 连续 flip360 重新基线（total_steps=1500，use_stdw=False，seed=0）

| policy | flip360 base | flip360 asym |
|---|---:|---:|
| A3 stage2 model_2398 | 2.8675 | 5.5360 |
| round1 mixed model_2400 | 5.7761 | 4.9773 |
| round1 mixed model_2500 | 9.1592 | 4.5270 |
| round1 mixed model_2550 | 3.0109 | 6.3930 |
| round1 mixed model_2597 | 4.8090 | 11.9218 |

Round 1 单次直接拉满 ±π 的 mixed replay **没有产生 Pareto 占优 checkpoint** —— 证实直接上满幅度过于激进。

## 幅度课程（最终胜出方案）

阶段 A（预热，±π/2 flip + ordinary 回放）：`workflows/configs/train_flip360_curric_a.yaml`，从 A3 stage2 训 200 iters。
阶段 B（爬升到满 ±π）：`workflows/configs/train_flip360_curric_b.yaml`，从阶段 A 末尾 checkpoint 训 250 iters，lr `8e-5`。

```text
阶段 A run: logs/rsl_rl/easyuuv_parametric/2026-06-29_23-48-23_flip360_curric_a_half_pi_stage0/
阶段 B run: logs/rsl_rl/easyuuv_parametric/2026-06-29_23-50-50_flip360_curric_b_full_pi_stage0/
最佳 checkpoint: model_2846.pt
```

### Flip360（连续）

| policy | flip360 base | flip360 asym |
|---|---:|---:|
| A3 stage2 model_2398 | 2.8675 | 5.5360 |
| curric model_2700 | 3.0752 | 4.6351 |
| **curric model_2846** | **2.0701** | **3.6606** |

### Ordinary medium 回归

| policy | ordinary base | ordinary asym |
|---|---:|---:|
| A3 stage2 model_2398 | 0.2257 | 0.2263 |
| **curric model_2846** | **0.2229** | 0.2266 |

## 结论（Round 2）

`curric model_2846` 在 flip360 上 **Pareto 占优 A3**（base `-27.8%`，asym `-33.9%`），
且 **ordinary medium 无遗忘**（ordinary asym 0.2263 -> 0.2266，噪声内）。这是首个可考虑并入主线的
flip360 checkpoint。幅度课程 + 混合回放是决定性因素；纯满幅度 fine-tune 不行。

保留项（不变）：OPR / drift-level projection / 无先验 domain transfer。

---

# Flip360 课程现状与平滑训练流程 — 2026-06-30

## 1. 当前课程训练现状

- 两段幅度课程已跑通并稳定胜出：阶段 A（±π/2）→ 阶段 B（±π），均混入 40% ordinary sine 回放。
- 产出 `model_2846`：flip360 base/asym 全面优于 A3，ordinary medium 不退化。
- 仍有残余误差：连续 flip360 的整体姿态 RMSE 约 0.85–0.91 rad，且**误差并非均匀**，集中在特定姿态区间（见第 3 节物理极限分析）。
- 已知不足：
  - 两段课程仍是“离散跳变”（A 直接跳到 B），幅度从 π/2 一步到 π，仍偏陡。
  - 单次训练 checkpoint 之间方差较大，需要靠扫描挑选，不够稳。

## 2. 更平滑的训练流程建议

把“离散两段”升级为“连续幅度爬坡 + 双目标早停”，降低跳变冲击并减小方差：

1. **连续幅度课程（amplitude ramp）**
   - 用线性/cosine 方式把 `ref_sine_amp` 的 roll/pitch 分量从 `0.5π` 平滑爬到 `π`，
     而不是 A→B 的一次性跳变。建议在 300–400 iters 内完成爬坡，再保持满幅度收敛。
   - 可先用现有 CLI 多段近似（π/2 → 3π/4 → π 三段，每段 ~120 iters），无需新代码即可更平滑。
2. **混合回放权重退火**
   - flip 概率 `ref_mix_flip_prob` 由 0.3 缓升到 0.5，前期多保 ordinary，后期多练 flip，
     兼顾防遗忘与专项能力。
3. **双目标早停 / checkpoint 选择**
   - 每隔 N iters 同时评估 flip360（base+asym）与 ordinary medium（base+asym），
     以“flip360 改善且 ordinary 不退化”作为接受准则挑选 checkpoint，减少靠人工扫描的方差。
4. **降低参考角速度（仅当判定为 rate-limited 时）**
   - 通过减小 `ref_sine_freq` 放慢参考；见第 3 节判据，当前数据判定为 angle-limited，暂不需要。

若上述平滑流程仍无法压低残余误差，则转入第 3 节：在难维持角度做 keep 测试，并考察物理极限。

## 3. 物理极限 / 参考速度分析（含分析脚本）

### 分析脚本

新增 `workflows/analyze_flip360_limits.py`，读取 `play_stdw_adapt.py` 产出的逐步 CSV
（`stdw_output_*.csv`），输出：

- 姿态误差随 **|参考角|** 分箱（找“难维持角度”）；
- 姿态误差随 **|参考角速度|** 分箱（判断“参考是否太快”）；
- 误差–参考角速度相关系数、高/低速箱 rmse 比；
- `control_effort` p95/max 与近饱和步占比；
- 自动判据：`RATE-LIMITED`（建议放慢参考）/ `ANGLE-LIMITED`（建议 keep 测试 + 查物理极限）。

用法：

```bash
# 方式一：直接给某次评估的 results_root，自动发现各 case 的逐步 CSV
python workflows/analyze_flip360_limits.py \
  --root .results/flip360_curric_eval_2846 \
  --report .results/flip360_curric_eval_2846/physical_limit_analysis.md

# 方式二：显式给 CSV 与标签
python workflows/analyze_flip360_limits.py \
  --csv <path>/stdw_output_*.csv --label base \
  --csv <path>/stdw_output_*.csv --label asym
```

> dt/decimation 默认 `1/120` 与 `10`（控制步 ≈ 0.083s）；与本仓库 eval 一致。

### 对 `model_2846` 的分析结论

来源：`.results/flip360_curric_eval_2846/physical_limit_analysis.md`

| 项 | base | asymmetric |
|---|---:|---:|
| 整体姿态 RMSE (rad) | 0.915 | 0.854 |
| 误差–参考角速度相关 | 0.110 | 0.099 |
| control_effort p95 / max | 11.97 / 18.86 | 11.51 / 19.13 |
| 近饱和步占比 (≥0.9·max) | 0.1% | 0.3% |
| 最差角度区间 (roll) | [1.57, 2.09) rmse 1.32 | [1.57, 2.09) rmse 1.34 |

判据：**ANGLE-LIMITED**（角度受限），不是 rate-limited：

- 误差与参考角速度几乎不相关（≈0.10），且高速箱样本极少（1–4 个，是 euler wrap 的瞬时尖峰，不是平滑运动）。
- 控制力度几乎不饱和（近饱和步 < 0.3%），说明不是推力被打满。
- 误差明确集中在 **roll ≈ π/2 ~ 2π/3（接近翻转/侧倒）** 区间。
- 参考本身已经很慢：幅度 π、频率 0.05Hz，参考峰值角速度仅 ≈ `π·2π·0.05 ≈ 1.0 rad/s`。

物理解释：接近翻转姿态时，COM–COB 偏置产生的浮力回正力矩与翻转方向相反，是该区间难维持的主要物理来源；
这与“angle-limited”判据一致。

### Step F1（准静态 keep）+ Step F2（参考速度扫描）取证结论（2026-06-30）

来源：`.results/flip360_keep_analysis.md`（keep，freq 0.005）、`.results/flip360_freq_sweep_analysis.md`（freq 0.025/0.05/0.1）。
对 `model_2846` 在同一 flip360 配置下只改 `ref_sine_freq`，得到一条清晰的单调关系：

| ref_sine_freq (Hz) | base 姿态 RMSE | asym 姿态 RMSE | base 近饱和步占比 |
|---|---:|---:|---:|
| 0.005（准静态 keep） | 1.742 | 1.118 | **15.2%** |
| 0.025 | 1.001 | 0.948 | 0.3% |
| 0.05（标准 eval） | 0.915 | 0.854 | 0.1% |
| 0.1 | 0.792 | 0.794 | 0.2% |

关键结论（与直觉相反，但数据明确）：

- **放慢参考不但无效，反而单调变差**；准静态 keep 最差，且推力近饱和步从 <0.3% 暴增到 15.2%。
- 这不是 rate-limited（误差–参考角速度相关始终 ≈0.06–0.16，弱相关）。
- 真正的机理是 **“驻留时间惩罚型角度极限”（dwell-time-penalized angle limit）**：
  - 接近翻转/侧倒姿态时，COM–COB 偏置的浮力回正力矩持续与保持方向相反；
  - 参考越慢 → 在难区间驻留越久 → 需要持续对抗回正力矩 → 推力饱和、误差累积；
  - 参考越快 → 借助角动量快速“穿过”难区间，驻留短 → 误差更小且不饱和。
- 因此 **用户最初“超出推力极限就放慢参考”的假设被数据否定**：对本系统，放慢只会加重物理负担。

物理上限的直接证据：准静态 keep 下 base 有 15.2% 的步推力近饱和（≥0.9·max），却仍保不住姿态，
说明在难角度做**长时间倒置保持**已接近/触及可用推力力矩的物理上限；这部分能力不能靠继续加大幅度训练补回。

## 4. 下一步建议（结合第 2、3 节，F1/F2 取证后更新）

F1（准静态 keep）+ F2（速度扫描）已闭环，明确：本任务是 **驻留时间惩罚型角度极限**，
不是 rate-limited，**放慢参考无效（且更差）**。因此原“放慢参考”分支作废，新建议如下：

1. **不要把目标定为“在难角度长时间倒置保持”**
   - 数据显示准静态保持已 15.2% 步推力近饱和却仍保不住，这是物理上限附近，继续加大幅度训练补不回来。
   - 连续后空翻（freq 0.05–0.1）才是该机型可达的合理任务面；keep 不应作为训练/验收目标。
2. **不放慢、必要时甚至略快地穿过难区间（F4 的修正版）**
   - 若要进一步压低残余误差，倾向于在难区间用**更快**的参考（更短驻留）配合参考整形，
     而不是放慢；同时在难区间放宽跟踪容差，避免策略在不可达的“长时间倒置”目标上学坏。
3. **若仍要压低连续 flip360 误差，走第 2 节平滑流程（F3）**
   - 连续幅度爬坡（π/2 → 3π/4 → π）+ 混合回放退火 + 双目标早停；
   - 注意：这能改善“穿越能力”，但无法突破“长时间倒置保持”的物理上限，预期收益在残余 RMSE 0.8 附近递减。
4. **物理侧（超出本轮范围，备选）**
   - 若确需更强的倒置保持能力，属于硬件/配置问题：减小 COM–COB 偏置、增大推力裕度或力臂，
     而非控制/训练能解决；建议作为机型设计反馈而非策略调优项。

执行选择：默认进入 **F3（平滑爬坡课程）** 压低连续 flip360 残余误差；**F4 改为“不放慢 + 容差整形”**。

保留项不变：OPR / drift-level projection / 无先验 domain transfer，等待进一步指引。

## 5. F3 平滑幅度爬坡课程结果（2026-06-30，未胜出）

把上一轮“A(±π/2) 一步跳 B(±π)”的 2 段课程，升级为三段平滑爬坡 π/2 → 3π/4 → π，
flip 概率 `ref_mix_flip_prob` 由 0.3 退火到 0.5，每段从上段末 checkpoint 顺序 fine-tune
（`--meta_stage 0 --resume_load_optimizer False`，起点 A3 `model_2398`）。
配置：`workflows/configs/train_flip360_f3_s{1,2,3}_*.yaml`。

| 段 | 幅度 | flip_prob | lr | run 目录 | 末 ckpt |
|---|---|---:|---:|---|---|
| S1 | ±π/2 | 0.3 | 1.0e-4 | `2026-06-30_22-11-54_flip360_f3_s1_halfpi_stage0` | model_2517 |
| S2 | ±3π/4 | 0.4 | 9.0e-5 | `2026-06-30_22-14-39_flip360_f3_s2_3qpi_stage0` | model_2636 |
| S3 | ±π | 0.5 | 8.0e-5 | `2026-06-30_22-19-46_flip360_f3_s3_fullpi_stage0` | model_2755 |

双目标 eval（total_steps=1500, use_stdw=False, seed=0, freq0.05）：

| checkpoint | flip360 base | flip360 asym | ordinary base | ordinary asym |
|---|---:|---:|---:|---:|
| **curric model_2846（当前最佳）** | **2.0701** | **3.6606** | 0.2229 | 0.2266 |
| F3 model_2650（S3 训练 MSE 最低 23.27） | 3.1214 | 3.5536 | — | — |
| F3 model_2755（S3 末段，训练 MSE 39.57） | 3.2019 | 4.0127 | 0.2246 | 0.2755 |

**结论：F3 平滑爬坡未超过 2 段课程 `model_2846`，当前最佳仍为 `model_2846`。**

- S3 进入满 ±π 后训练 `log_mse` 从段初 ~6 立即爆到 20–85 的混沌区间且全程不收敛，
  整段无可用收敛点（见 `.../2026-06-30_22-19-46_..._stage0/mse_curve.jsonl`）。
- F3 最佳候选 model_2650 flip360 base 仍 3.12，比 2846 差约 50%；2700/2750 训练 MSE 更高
  （39.9/41.0），不可能回补 ~50%，故未续测。model_2755 还出现 ordinary asym 轻微遗忘（0.2755 vs 0.2266）。
- 这与 F1/F2 的“驻留时间惩罚型角度极限”一致：瓶颈在**满倒置区训练不收敛**，
  平滑课程分段并不能稳住该区段——问题不在课程跳变幅度。

**因此 F3 的预期“走平滑流程压低残余误差”被否定**：下一步应转向 F4（不放慢 + 难区间
容差整形），直接对满 ±π 段做 reward/参考整形，而非继续堆课程分段。F4 是否启动等待用户拍板。

---

## 6. F4 难区间容差整形 + 物理边界判定（未胜出，确认架构边界）

### F4 实现

不放慢参考（沿用 freq0.05 满 ±π），改为在**接近倒置的难区间**放宽姿态跟踪容差：
`easyuuv_env.py` 新增 cfg `flip_tol_relax / flip_tol_band_lo / flip_tol_band_hi`（默认
relax=0 零行为变更）。在 `_get_rewards` 内，按 goal 相对竖直的倾角 `tilt = acos(R[2,2])`
落入 `[band_lo, band_hi]` 时用 smoothstep 把 `rew_ang` 高斯指数除以 `(1 + relax·w)`，
等效降低倒置区误差惩罚斜率，避免始终在线的姿态项对物理不可达/不稳定目标持续扣分。
配置 `train_flip360_f4_tolshape.yaml`：`flip_tol_relax=2.0`，band=[120°,180°]，
从 `model_2846` fine-tune 150 iter（`--meta_stage 0 --resume_load_optimizer False`）。
run 目录 `2026-06-30_23-14-55_flip360_f4_tolshape_stage0`，最佳保存 ckpt model_2850。

### 双目标 eval（total_steps=1500, use_stdw=False, seed=0, freq0.05）

| checkpoint | flip360 base | flip360 asym | ordinary base | ordinary asym |
|---|---:|---:|---:|---:|
| **curric model_2846（当前最佳）** | **2.0701** | **3.6606** | 0.2229 | 0.2266 |
| F4 model_2850（容差整形, 训练 MSE 最低） | 2.0520 | 3.6747 | 0.2229 | 0.2268 |

**结论：F4 与 model_2846 全面持平（四项均在噪声内），无胜出也无遗忘。**

- 容差整形并未压低倒置区残差；F4 训练 `log_mse` 从段初 6.1 立即脱离并全程在
  14–42 混沌区间震荡、再未回到 6.1（最佳保存 ckpt 2850 已是 15.2），与 F3 完全同构。
- 这进一步坐实：瓶颈是**倒置区训练不收敛**，对 reward 容差整形不敏感。

### 物理力矩边界判定（`workflows/analyze_flip360_torque_budget.py`）

逐轴对比控制力矩预算 vs 倒置区浮力回正力矩 `|r_com_to_cob × F_buoy|`（F_buoy=222.5 N）：

| 项 | roll | pitch | 说明 |
|---|---:|---:|---|
| 执行器硬上限（PWM=1.0 clip） | 114.5 Nm | 70.4 Nm | 8 推进器满推 |
| 名义工作点（S 面 a≈1, action_lim=0.15） | 17.0 Nm | 10.4 Nm | 策略实际可达 |
| 回正力矩 base / asym（最坏 sin=1） | — | 2.22 / 15.89 Nm | r=0.010 / 0.071 m |

- **不是执行器力矩受限**：硬上限超 asym 最坏回正 15.9 Nm 约 4–7 倍，满 ±π 原则上可达。
- **名义工作点瓶颈**：S 面 action_lim=0.15 下名义 pitch 权威 10.4 Nm < asym 最坏回正
  15.9 Nm；克服回正须把动作压进 sigmoid 饱和尾部（≈bang-bang），难学习。
- **主导边界 = 倒置区不稳定平衡 + 欧拉轴控制奇异 + 名义工作点可学习带宽不足**，
  三者均不可由 reward 整形/课程（F1–F4）移除。

**总结论（回答“边界在哪里 / 是否无解”）**：满 ±π 后空翻**并非严格物理无解**（执行器
力矩有 4–7x 余量），但在**当前 S 面 + 欧拉角 + action_lim=0.15 控制架构下，倒置区训练
不收敛属于架构性边界**。F1–F4 全部止步于此残差（base RMSE ≈0.85–0.91 rad，误差集中在
倒置区）。要进一步压低需改控制架构（SO(3)/几何姿态控制、增大 action_lim、重设计分配），
而非继续 reward/课程调参。**model_2846 已逼近该架构上限，保持为最佳 checkpoint。**

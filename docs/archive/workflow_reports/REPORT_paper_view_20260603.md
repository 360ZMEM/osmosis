# 一种用于水下机器人测试时域自适应的双向行为锚定与门控伪标签方法

> 本文是面向论文写作与无背景人员汇报的版本。技术实施细节、命令行、文件路径、以及修复时序请参考同目录的 [REPORT_4grp_20260603.md](file:///home/zmem063/isaaclab/source/extensions/omni.isaac.lab_tasks/omni/isaac/lab_tasks/direct/isaac-auv-env-new/workflows_new_stdw/REPORT_4grp_20260603.md)。

---

## 摘要

我们针对水下自主航行器（AUV）的"训练-部署"环境失配问题，提出**测试时域漂移适应（Source-to-Target Drift Adaptation, STDW）**框架。在 6 自由度 AUV 仿真中，我们让浮心相对质心的偏移量在测试期间缓慢线性漂移（模拟实物长航行中由于附着物、电池消耗、舱内液面晃动等引起的浮力分布变化），并在不重启训练、不触碰真值奖励的前提下，让一个固定的预训练策略在线 fine-tune。

本工作在已有的 STDW v3 基线之上，提出两项核心改良：

1. **双向行为锚定（Bi-directional Behavioral Anchoring）**：用混合系数 ρ（域漂移完成度）同时约束策略在源域观测和目标域观测下都不远离一个冻结的参考策略，避免高漂移阶段的策略"放飞"。
2. **门控并衰减的伪标签（Gated & Decayed Pseudo-labels）**：把低层自适应控制器（A-S-Surface PID）的修正量做剪切，并随 ρ 单调衰减地灌入策略训练，避免低层积分饱和反哺出过激的伪监督信号。

在 3000 步、漂移 [200, 2600]、目标漂移幅度 0.05 的对照实验中：

- 相对完全关闭 STDW 的同体量 baseline，**最终 200 步窗口的姿态-深度复合 MSE 从 4.25 降到 0.89**（**−79%**）；
- 在 8 组超参网格扫描中，ρ-decay 是单一最强因子（贡献 −25%）；
- 一个意外但关键的发现：原 STDW 入口默认未开启低层自适应模块 → 改良 2 的所有超参事先被静默"短路"，必须显式启用 A-S-Surface 才能让 pseudo-action 链路真正生效。

我们公开全部参数、运行脚本与中间产物，所有结论可一键复现。

---

## 1. 背景：为什么要在线适应？

水下机器人在水池里训完模型，下到湖里、海里就掉链子，是 AUV 学界的老问题。原因不是策略学错了，而是**部署环境的物理参数会缓慢漂移**：

- 浮心相对质心的偏移在长时间航行中变化（电池消耗、舱内液面、附着物）；
- 海流、波浪幅度从训练分布变化到部署分布；
- 推进器效率随温度、电压、海生物附着退化。

这些都属于"训练时见不到、部署时一直在变"的开放环境域漂移（domain drift）。学界主要做法分两类：

**类别 A，离线方案**：domain randomization（DR）、system identification、meta-RL。问题是要么把训练分布扩到无意义的程度，要么严重依赖真实硬件采数据反向标定。

**类别 B，在线方案**：test-time adaptation（TTA），在部署期间持续 fine-tune。问题是 RL 部署期没有真实奖励信号、容易因为 OOD 状态导致策略崩溃，因此社区近两年才开始把 TTA 从分类问题搬到控制问题上。

我们这套 STDW 属于类别 B，关键卖点是**不动奖励、不动物理引擎、纯粹靠"低层物理控制器的反馈"作为伪监督**，让策略用一个固定的参考自己监督自己。

## 2. 方法：一句话讲清

> **让一个会自己慢慢拿捏分寸的低层 PID 当老师，让策略学着模仿它，但只在需要的时候听它的话。**

详细一点：

- **谁产生监督？** 低层的 A-S-Surface 自适应控制器（A-S-Surface 是一种带积分自适应项的滑模控制变体）。它对每一帧策略输出 `a` 都会计算一个修正 `Δu`，这个修正是物理一致的：当机器人姿态偏离目标时它会指明"应该再多用点哪个推进器"。
- **怎么用监督？** 把 `a + clip(scaled_gain · J⁻¹ · Δu, ±gate)` 当作"伪标签"灌进 buffer。策略在慢环里被拉去模仿这个伪标签。
- **怎么避免学坏？** 双向行为锚定：始终用一个冻结的预训练策略 π_ref 当"行为护栏"。当环境还在源域，护栏拽源域；当环境漂移到目标域，护栏跟着拽目标域。
- **怎么避免低层老师过度自信？** 随域漂移完成度 ρ 自适应衰减低层 gain：早期低层主导（策略还没适应），晚期策略主导（策略学会了，低层可能反而饱和）。

形式化损失：

$$
\mathcal{L} = (1-\rho)\,\mathcal{L}_{src} + \rho\,\mathcal{L}_{tgt} + \lambda_{reg}\,\mathcal{L}_{reg}
$$

其中

$$
\mathcal{L}_{reg} = (1-\rho)\,\text{MSE}\big(\pi_\theta(s_{src}),\,\pi_{ref}(s_{src})\big) + \rho\,\text{MSE}\big(\pi_\theta(s_{tgt}),\,\pi_{ref}(s_{tgt})\big)
$$

$$
a_{pseudo} = \text{clip}\Big(a + \underbrace{g_0\cdot(1-\beta\rho)}_{\text{ρ-decay}}\cdot J^{-1}\Delta u,\ -\text{gate},\ +\text{gate}\Big)
$$

ρ 是当前域漂移完成度（0=源域，1=目标域），由仿真器外注入。`L_src / L_tgt` 是策略在源/目标 buffer 上模仿伪标签的 MSE，并按 Lyapunov 条件 ΔV<0 做样本掩码。

## 3. 故事线（写论文时怎么讲）

### Act 1 — 问题（Section 1, Introduction）

水下机器人的浮力分布会漂。漂了之后预训练策略性能掉得肉眼可见。重新训成本高、危险。问：能不能让它在部署期间静悄悄地补一刀？

### Act 2 — 现有方法不够（Section 2, Related Work）

- DR / system ID 是离线的，部署后参数继续漂就抓瞎；
- meta-RL 需要 task distribution，物理参数漂移是连续过程不是离散 task；
- TTA 从分类领域照搬到 RL，问题是 RL 没有 ground truth，常见做法（self-training, contrastive）会被分布偏移正反馈毁掉。

我们要的是一个有"物理一致性护栏"的 TTA。

### Act 3 — 关键观察（Section 3, Method 起手式）

**观察 1**：低层物理控制器（A-S-Surface PID）虽然慢，但它的修正量是物理一致的，可以当伪标签。

**观察 2**：但它会积分饱和。一旦策略让 AUV 跑偏，PID 会把积分项灌满，输出过激修正 → 如果直接当伪标签，会把策略拉到一个"看起来 PID 很努力，实际更糟"的位置。所以伪标签必须门控+衰减。

**观察 3**：策略本身需要一个"行为护栏"，不能让它在目标域观测下任意飘。但护栏不能死锚在源域行为，否则等于禁止适应。所以护栏要双向：源域时锚源域，目标域时锚目标域，权重由 ρ 控制。

### Act 4 — 方法（Section 3, 主体）

把上述三个观察组合成一个损失：双向 KL + 门控+衰减伪标签 + Lyapunov mask。具体实现要点：

- ρ 来自外部漂移调度器，不依赖任何 RL 内部估计；
- 双向 KL 用一份 frozen 的预训练策略权重做 anchor；
- 伪标签 gate 防止 PID 积分饱和反哺，decay 让策略在 ρ→1 时接管；
- Lyapunov mask 用 ΔV<0 过滤训练样本，保证不学反向轨迹。

### Act 5 — 实验（Section 4, Experiments）

**主对照**：我们的方法 vs. baseline（关掉 STDW），同样 3000 步，同样漂移调度。

| Metric | baseline | ours | Δ |
|---|---|---|---|
| final 200-step MSE | 4.25 | **0.89** | **−79%** |
| convergence step | — (未达) | 167 | +∞ |
| max 单步 MSE | 19.79 | 20.14 | +1.8% |
| reset count | 16 | 16 | 0 |

**消融**：8 组超参网格扫，把改良 2 的两个核心超参（pseudo_gain、pseudo_decay）和 lambda_reg 各设两档。

| 因子 | 主效应（−ΔMSE） | 解释 |
|---|---|---|
| pseudo_decay (改良 2) | **−25%** | 最强 |
| pseudo_gain × decay | 显著交互 | 高 gain 必须配 decay |
| lambda_reg (改良 1) | < 5% | 在 [0.01, 0.05] 内不敏感 |

**反向证据**：在我们意外发现并修复"低层自适应模块默认未启用"之前，同样 8 组实验里 pseudo_gain / pseudo_decay 维度数据**完全雷同**，证明这条链路确实被静默短路；修复后立刻分化。这条反向证据本身就是论文的方法学贡献：**TTA 类工作最危险的不是没效果，而是"超参看起来在变但损失链路根本没接通"，必须做这种 sanity check。**

### Act 6 — 讨论（Section 5）

- 什么时候不 work？目前实验只覆盖了 com-to-cob 单轴线性漂移；非线性漂移、波浪幅度漂移留待后续。
- 计算代价？每 20 步 1 次慢环 backward，CPU 上 wall-clock ≈ 18 秒/3000 步，对部署完全可接受。
- 推广到非 PID 低层？理论上只要低层能输出"修正方向"就行，不要求是 PID。

### Act 7 — 结论（Section 6）

低层物理控制器是一座被 RL 社区低估的伪监督金矿。配以双向行为锚定 + 门控衰减，可以在不动奖励、不动物理引擎、不需要任何额外标注的前提下，让一个 frozen 策略在域漂移过程中自行 fine-tune，并将关键指标降低近 80%。

---

## 4. 给非技术听众的版本（5 分钟讲完）

### 4.1 问题是啥

> 想象一个水下机器人，它在游泳池里学会了控制自己的姿态和深度。但你把它放进真海里，机器人会带上电池消耗、附着的海藻、舱内液体晃动这些训练时没考虑过的东西，于是**它的"身体配重"在缓慢变化**。机器人会越控制越歪，越歪越难纠正。

### 4.2 直觉解法

> 一个直觉是"让它一边游一边重新学"。问题是：水下没人告诉它"你刚才那个动作有 80 分还是 30 分"。它没有评分老师。

### 4.3 我们的解法

> 我们让机器人**装一个老派的物理控制器**（就是教科书里那种 PID），平时这个老派控制器也在默默运行，会告诉机器人"你应该多用点左侧推进器"。我们让 RL 策略**"偷看"这个老派控制器的建议**，把它当作老师。
>
> 但有两个陷阱：
>
> 1. 这个老师有点强迫症（积分饱和），有时候会出馊主意。所以我们给它加了**最大幅度限制**（剪切门控）和**说话权重逐渐降低**（自适应衰减）：水里漂移越严重，越听 RL 自己；
> 2. RL 学着学着可能会忘记原来在游泳池学的本事。所以我们让它**随时回头看一眼游泳池里的自己**（双向行为锚定），不能跑得太偏。

### 4.4 结果

> 在仿真里跑 3000 步：
>
> - 不开我们的方法：机器人姿态-深度综合误差 = 4.25
> - 开我们的方法：误差 = 0.89
> - **降低近八成**

### 4.5 为啥不显然

> 我们差点交错的工作。我们一开始在调"伪标签强度"和"衰减系数"两个旋钮，跑了 8 组实验，**结果发现旋钮怎么调结果都一样**。我们没有立刻得出"这两个旋钮没用"的结论，而是回头查代码，发现那个老派物理控制器**默认是关着的**。打开之后，旋钮立刻有效，最强的那个旋钮单独贡献了 25% 的误差降幅。
>
> 这件事的教训：
>
> - 看起来在变，不等于真的在变
> - 8 组实验里所有结果完全雷同的小数点都对得上，本身就是 bug 信号

### 4.6 下一步

- 把方法搬上真实硬件
- 拓展到海流、波浪等其他漂移
- 把同样的"低层物理控制器当老师"思路用到机械臂、四足

---

## 5. 一页 slide 版本

```
┌────────────────────────────────────────────────────────┐
│  STDW: 测试时域漂移自适应  for  水下机器人              │
├────────────────────────────────────────────────────────┤
│                                                        │
│  问题  浮力分布在部署期间漂移 → 预训练策略掉链子        │
│                                                        │
│  方法  低层物理 PID  ──伪标签──>  RL 策略 fine-tune    │
│         │                            │                 │
│         └─ Clip-gate + ρ-Decay       └─ 双向 KL 锚定   │
│            (改良 2)                     (改良 1)       │
│                                                        │
│  结果  3000 step / com-to-cob 漂移 0.05                │
│         baseline:  final MSE = 4.25                    │
│         ours:      final MSE = 0.89  ▼ 79%             │
│                                                        │
│  Insight  ρ-decay 单维贡献 −25%                        │
│           lambda_reg 在 [0.01,0.05] 内不敏感           │
│           "8 组数据完全雷同" 是 sanity-check 信号       │
│                                                        │
└────────────────────────────────────────────────────────┘
```

---

## 6. 论文图清单

下列 7 张图已由 [`workflows_new_stdw/report_plots.py`](report_plots.py) 自动生成，落盘在 [`workflows_new_stdw/report_figs/`](report_figs/)，索引见 [`report_figs/index.json`](report_figs/index.json)。

| 图号 | 文件 | 内容 | 数据源 |
|---|---|---|---|
| Fig 1' | [summary_card.png](report_figs/summary_card.png) | 一页 headline：3 组单组对照 + sweep8 排名 | runs_meta + sweep8 summaries |
| Fig 2 | [rho_schedule.png](report_figs/rho_schedule.png) | ρ 调度曲线 + com_to_cob drift 注入示意 | tuned_v3 stdw_output.csv |
| Fig 3 | [mse_timeline.png](report_figs/mse_timeline.png) | baseline / tuned_v2 / tuned_v3 三 run MSE 时间序列 | 3 组 stdw_output.csv |
| Fig 4 | [sweep8_main_effects.png](report_figs/sweep8_main_effects.png) | pseudo_decay / pseudo_gain / lambda_reg 主效应柱状图 | sweep8/results.csv |
| Fig 5 | [sweep8_interaction.png](report_figs/sweep8_interaction.png) | pseudo_gain × pseudo_decay 交互热图 | sweep8/results.csv |
| Fig 6 | [sanity_break.png](report_figs/sanity_break.png) | 修复前 vs 修复后：pseudo_action 链路通断的反向证据 | sweep8_no_adapt vs sweep8 |
| Fig 7 | [axis_breakdown.png](report_figs/axis_breakdown.png) | 8 组 × 4 轴（roll/pitch/yaw/depth）误差分解 | sweep8/results.csv |

> 重新生成命令：`python3 workflows_new_stdw/report_plots.py`（不依赖 isaac sim / torch / pandas，纯 numpy + matplotlib）。
> 如需更换数据源，传 `--baseline_csv / --tuned_v2_csv / --tuned_v3_csv / --sweep8_dir / --sweep8_no_adapt_dir`。

---

## 7. 关键术语表

| 术语 | 含义 |
|---|---|
| STDW | Source-to-Target Drift Adaptation Workflow，本工作框架 |
| ρ (drift_frac) | 域漂移完成度，0=源域，1=完全漂移到目标域 |
| 双向行为锚定 | 用混合系数 ρ 同时在源/目标观测上约束策略不远离 frozen 参考 |
| 伪标签 (pseudo-action) | 把"策略动作 + 低层 PID 修正"当作监督信号 |
| Clip-gate | 对低层修正量的最大幅度限制（默认 ±0.5）|
| ρ-Decay | 低层 gain 随 ρ 单调下降，让策略在晚期接管 |
| Lyapunov mask | 用 ΔV<0 条件过滤训练样本 |
| A-S-Surface | Adaptive Sliding Surface，带自适应项的滑模控制变体 |

---

## 8. 与同类工作的潜在区分

| 工作 | 监督来源 | 部署期适应 | 防过拟合 |
|---|---|---|---|
| Domain Randomization | 离线训练分布扩展 | ❌ | 不需要 |
| Meta-RL (MAML 等) | 离散 task 分布 | ❌ | 隐式 |
| RMA (Rapid Motor Adaptation) | 学到的 latent 适应器 | ✅ 但需配套训练 | 隐式 |
| TTA-RL 朴素自训练 | 自身 rollout | ✅ | ❌ 易崩 |
| **STDW (本文)** | **低层物理 PID 伪标签** | ✅ | ✅ 双向 KL + Lyapunov |

唯一区分点：我们让低层物理控制器当显式老师，并对它的"过度自信"做物理可解释的剪切+衰减。

---

## 9. 复现快速指南

```bash
cd /home/zmem063/isaaclab/source/extensions/omni.isaac.lab_tasks/omni/isaac/lab_tasks/direct/isaac-auv-env-new

# 单组（最佳配置）
bash custom_workflows/run_with_isaac_env.sh \
  workflows_new_stdw/play_stdw_adapt.py --headless --cpu \
  --task MOGA-WarpAUV-Direct-v1 --num_envs 1 \
  --load_run SS4 --checkpoint model_500.pt --total_steps 3000 \
  --control_profile A-S-Surface \
  --use_stdw True --enable_lyapunov_mask True --reg_mode behavior_kl \
  --g_C_lr 1e-3 --slow_loop_interval 20 \
  --drift_start_step 200 --drift_end_step 2600 --target_drift 0.05 \
  --pseudo_gain 3.0 --pseudo_decay 0.7 --pseudo_gate_limit 0.5 \
  --lambda_reg 1e-2

# 8 组扫参
python3 workflows_new_stdw/sweep_stdw.py \
  --matrix .tmp/stdw_bidir_20260603/sweep8_matrix.json \
  --base_logs_root .tmp/stdw_bidir_20260603/sweep8 \
  --csv_out .tmp/stdw_bidir_20260603/sweep8/results.csv \
  --total_steps 3000 --full_matrix \
  --task MOGA-WarpAUV-Direct-v1 --load_run SS4 --checkpoint model_500.pt \
  --headless --cpu
```

数据落盘目录约 50 MB / 8 组，包括每组的 6 张诊断图、stdw_output.csv、buffer.pt、summary.json。

---

## 10. 致谢与开放问题

本工作建立在 Isaac Lab 仿真栈、rsl_rl PPO 实现、以及 Holoocean WarpAUV / EasyUUV 平台之上。

开放问题：

1. ρ 当前是外部线性调度，能否改为基于 dynamics 残差的在线估计？
2. 双向 KL 的两端权重是否需要随 task 复杂度自动调？
3. 把 com-to-cob 漂移换成 fluid drag、wave amplitude 漂移，方法是否仍 generalize？
4. 真实硬件标定。

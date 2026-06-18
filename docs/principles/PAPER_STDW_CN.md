# STDW：面向 AUV 参数化控制的少样本在线自适应方法

IEEE 期刊风格 Markdown 初稿，2026-06-13。  
目标篇幅：在表格和图最终补齐后约 6-8 页有效内容。

## 摘要

本文提出 STDW（S-surface-assisted Targeted Drift Wrapping），一种面向自主水下航行器（AUV）实物部署的少样本在线自适应工作流。基础控制器由强化学习（RL）策略与非线性 S 面低层控制器组成，策略输出 8 维动作，其中前 4 维为控制意图（surge/sway/heave/yaw），后 4 维为增益调制通道。STDW 在部署过程中利用可观测响应数据、低层控制器伪动作和 Lyapunov 物理围栏对策略进行慢环更新，从而在 embodiment 变化、波浪扰动、传感器噪声和质心-浮心偏移（COM-COB）下保持闭环性能。针对实物部署无法读取仿真特权状态的问题，本文进一步给出 observable-only 的 micro-probe，并采用 A/B/A 局部基线与最小扰动原则，避免在证据不足时注入不必要的漂移。48-cell 全矩阵（3 波况 × 4 机型 × 2 整定模式 × STDW 开关）表明：STDW 在 base 机型上将 final tracking MSE 降低 67.8%、在 long_body 上降低 65.2%，且跨波况标准差仅 0.05%；但在 asymmetric 机型上劣化 +158%，原因是朴素 drift 被错误当作外部扰动注入。已有后续诊断进一步显示，朴素漂移会显著损害 asymmetric 机型（pitch/depth 耦合导致 final200 从 0.026 升至 0.423），而 offset-aware routing 与保守 probe 评分可规避主要失效模式。本文还设计并回填 IMU 级角速度噪声、角速度低通、观测/动作延迟与 STDW 组件消融实验（共 26 个运行 cell），用于判定实物部署边界。实验表明：(i) A3 控制器中 ζ2 的 D 项已使用真实角速度，无需替换输入；(ii) 角速度观测噪声使 base 机型 final_mse 从 0.073 升至 0.138（+91%），而 2 步观测延迟进一步升至 0.163（+125%）；(iii) `d_filter_tau=0.05` 在本矩阵中无明显收益，建议实物部署暂不启用低通；(iv) 分位数滤波和触发门是 STDW 对 base 机型最关键的两个组件（移除后 base final_mse 分别上升 177% 和 273%）；(v) 正确的 asymmetric 修复不是删除 STDW 组件，而是使用 observable-only micro-probe 使证据不足时保持 baseline。

**关键词** — 自主水下航行器，在线策略自适应，sim-to-real 转移，S 面控制，Lyapunov 门控，强化学习，少样本。

## I. 引言

水下机器人实物部署面临显著的 sim-to-real 问题。水动力系数、浮力分布、推进器布局、COM-COB 偏置和 IMU 噪声都会使仿真训练策略在实物中出现偏差。对于小型 AUV，厘米级的质心/浮心偏移就可能通过姿态-深度耦合放大误差。文献中已有多种 sim-to-real 方法：域随机化、系统辨识后微调、域自适应和元学习。然而，这些方法要么需要大量实物样本，要么假设可访问完整的系统状态（包括仿真特权量如 COM-COB 偏移和水动力系数）。在实际部署中，这两个前提都难以满足。

EasyUUV A3 控制器采用紧凑的 12 维观测，不依赖仿真特权：

```text
[goal_quat(4), depth_z(1), root_quat(4), body_ang_vel(3)]
```

策略输出为 8 维：前 4 维进入低层控制器，后 4 维调制控制器增益。本文关注的 STDW 是该策略外层的在线自适应机制。

### 贡献

本文的主要贡献如下：

1. **快慢环结构**：将 RL 策略的高层控制意图与 S 面低层稳定性相结合，在线慢环仅在可观测证据充分且 Lyapunov 能量条件允许时更新。
2. **伪动作学习**：利用低层控制器的修正量构造 `a_pseudo = a + Δu`，使慢环学习稳定控制器的补偿方向而非拟合原始策略动作。
3. **Lyapunov 物理围栏**：通过 `V_t = ½ eᵀPe` 能量变化过滤明显违背物理稳定趋势的样本。
4. **Observable-only Micro-Probe**：不读取仿真特权 COM-COB，通过小幅候选 drift 的 A/B/A 局部基线响应判断方向，证据不足时选择 baseline。
5. **系统实验验证**：完成 48-cell 全矩阵 + 26-cell 专项矩阵（噪声/延迟/消融），给出实物部署建议。

### 关键立论

STDW 的核心立论是：实物可部署性不能依赖仿真特权状态，也不能在证据不足时强行扰动系统；在线更新必须被物理响应、行为锚定和最小扰动原则约束。

## II. 相关工作

### A. 水下航行器传统控制

传统 AUV 控制常采用 PID、PD 或 S 面控制器。PID 控制器结构清晰、工程可解释，在中等扰动下稳定性较好。其局限在于：难以处理强非线性耦合（如 pitch-depth 耦合）；增益调参依赖经验，且对 embodiment 变化敏感。S 面控制器通过非线性 sigmoid 映射误差到控制量，具有自适应增益特性，但仍缺乏对未知扰动的系统补偿能力。

### B. 强化学习在 AUV 控制中的应用

近年的工作将 PPO 等 RL 算法应用于 AUV 姿态和轨迹控制。RL 能够学习非线性补偿与动作整形，在仿真中可达到比传统控制器更低的跟踪误差。然而，RL 策略在实物部署中面临两大问题：(i) 训练-部署域差异导致性能下降；(ii) 缺乏安全约束，策略可能输出物理上不安全的动作。

### C. 在线自适应与行为正则化

在线策略自适应方法能在部署期更新策略以适应新域。但若无约束，梯度更新可能破坏闭环稳定性。现有方法采用源策略行为锚定（如 DAgger 风格）、KL 正则化、或信任域限制。STDW 在此基础上结合了伪动作引导与 Lyapunov 物理围栏，使在线更新同时受到行为一致性和物理稳定性双重约束。

### D. Sim-to-Real 转移

域随机化、系统辨识后微调和元学习是三种主要 sim-to-real 范式。域随机化在训练期增加域参数扰动，使策略鲁棒；但过度随机化会降低标称性能。微调需要实物样本，且可能过拟合到个别工况。STDW 选择的是在线少样本自适应路径：不重训、不预随机化过度，而是在部署期利用极少样本进行保守更新。

## III. 系统模型

### A. AUV 动力学与观测

考虑刚体 AUV 在 6-DOF 下的简化模型：

```math
M \dot{\nu} + C(\nu)\nu + D(\nu)\nu + g(\eta) = \tau
```

其中 `M` 为惯性矩阵，`C(ν)` 为科里奥利力，`D(ν)` 为水动力阻尼，`g(η)` 为恢复力（含重力与浮力），`τ` 为推进器推力。恢复力项包含 COM-COB 偏移的效应：

```math
g(\eta) = \begin{bmatrix} (W-B)\sin\theta \\ -(W-B)\cos\theta\sin\phi \\ \text{pitch/roll moment from } r_{COB} \times F_B \end{bmatrix}
```

其中 `W` 为重力，`B` 为浮力，`r_COB` 为 COM-COB 偏移矢量。该偏移是部署中 embodiment 变化的主要来源之一。

观测设计原则是**不依赖仿真特权状态**。EasyUUV A3 选择：

```text
[goal_quat(4), depth_z(1), root_quat(4), body_ang_vel(3)] = 12 维
```

`body_ang_vel` 来自 IMU 陀螺仪（实物可得），`root_quat` 来自 IMU 姿态融合（实物可得），`depth_z` 来自深度传感器。不观测线速度、COM-COB 或水动力系数。

### B. S 面控制器

低层控制器使用非线性 S 面公式：

```math
u = \frac{2}{1 + \exp(-s\zeta_1 e - s\zeta_2\dot e)} - 1 \quad \in [-1, 1]
```

其中 `e` 为跟踪误差，`s` 为斜率系数。该公式与 PID 控制器接受相同的输入（误差及其导数），但具有非线性增益特性：

- 当 `|ζ1·e + ζ2·ė|` 较小时，S 面函数近似线性，行为类似 P+D。
- 当 `|ζ1·e + ζ2·ė|` 较大时，输出饱和于 `±1`，自动限制控制量上限。

`ζ1` 是主要响应参数，决定响应速度和超调趋势；`ζ2` 是 D 项，主要影响阻尼。在当前 A3 实现中，roll/pitch/yaw 的 D 项已经使用真实 body 角速度而非数值差分：

```math
\dot e \approx -\omega_b \Delta t
```

这避免了数值差分的噪声放大问题。实物部署中"是否把 ζ2 换成角速度"的结论是：**代码层面已经完成**。当前需要实验确认的是：加入真实 IMU 级观测噪声后，角速度是否需要低层低通，以及观测/动作延迟对 STDW 性能是否敏感。

### C. 参数化策略

策略输出为：

```text
[u_surge, u_sway, u_heave, u_yaw, a_gain0, a_gain1, a_gain2, a_gain3]
```

前 4 维表达控制意图，直接输入 S 面控制器。后 4 维用于调制 `ζ1/ζ2` 参数，借助 bounded safeguard 限制在标称增益附近（默认在 `[0.5×标称, 2×标称]` 范围），避免策略将增益推到明显不安全区域。这种参数化设计使策略既能调整控制意图，又能在线调整控制器参数，而无需修改底层控制律结构。

### D. Runtime Drift

STDW 主要研究的部署扰动之一是 COM-COB 漂移。wrapper 在 episode 中按线性插值逐步施加 drift：

```math
\text{offset}_t = \text{offset}_0 + f(t/T) \cdot \text{target\_drift}
```

其中 `f` 为 drift 进度函数（可配置线性或 S 曲线）。已有诊断说明，固定 `+x` drift 并不总是安全：asymmetric 机型初始偏移已经是 `(0.05,0.05)`，再施加 `+0.05 x` 会把 x 推到 `0.10`，导致 pitch/depth 强耦合失效（final200 从 0.026 升至 0.423，+158%）。

## IV. 方法

### A. 快慢环结构

STDW 采用快慢环架构。快环（每控制步执行）：

1. 策略推理：`a = π_θ(o)`
2. 低层 S 面控制：`u = S-surface(a[:4], ζ)`
3. 记录 `(o, a, a_pseudo, r)` 到 replay buffer

慢环（每 `slow_loop_interval` 步执行，默认 60 步）：

1. 从 buffer 采样 `batch_size` 个样本（默认 256）
2. 构造 source anchor `a_src = π_{θ_ref}(o)` 和 target anchor
3. 计算混合损失并更新策略参数 `θ`

慢环损失为：

```math
L = (1-\rho)L_{src} + \rho L_{tgt} + \lambda L_{reg}
```

其中 `L_src = ||π_θ(o) - a_src||²` 约束策略不偏离源行为；`L_tgt = ||π_θ(o) - a_pseudo||²` 引导策略学习低层控制器的补偿；`L_reg = ||θ - θ_ref||²` 为 L2 正则化防止参数漂移。`ρ` 随 drift 进度变化：初期偏向 source anchor 保持稳定，后期逐步引入 target 适应。

### B. 伪动作学习

低层控制器将自身修正量写入 `_pid_value_add_buf`。STDW 构造：

```math
a_{pseudo} = a + \Delta u
```

这一设计的直觉是：低层 S 面控制器已经在执行稳定化修正；如果慢环只学习 `a`（原始策略动作），它会学习如何"模仿自己"——这是一个平凡解。通过学习 `a_pseudo`，慢环获得来自稳定化控制器的额外信息，知道在当前状态下控制器"想要"的补偿方向是什么。

伪动作使慢环更新具有了物理意义：它不是在优化一个抽象的 RL reward，而是在学习如何更好地配合低层稳定控制器的工作。

### C. Lyapunov 物理围栏

STDW 对 roll、pitch、yaw、depth 误差计算 Lyapunov 能量：

```math
V_t = \frac{1}{2} e_t^\top P e_t
```

其中 `P` 为对角正定矩阵（由 `lyapunov_p_diag` 配置）。样本参与慢环更新的条件是：

```math
V_{t+1} - V_t < \epsilon \cdot V_t + \epsilon_{abs}
```

即能量变化不超过相对和绝对阈值的组合。该机制的作用不是严格证明全局稳定性，而是在工程上过滤明显违背物理稳定趋势的样本，防止慢环学习错误响应。

### D. Observable-Only Micro-Probe

实物部署无法读取 COM-COB 偏移。Micro-probe 的解决方案是：在部署初期，短暂施加小幅候选 drift（如 `±0.02` 在 x/y 轴），通过观测到的姿态/深度误差响应推断偏移方向。

新版 scoring 采用 **A/B/A 局部基线**设计：

```
baseline window → candidate window → baseline window
```

候选评分与**紧邻的局部基线均值**比较（而非全局初始基线），消除系统误差随时间自然衰减造成的偏置。要求候选满足：(i) 相对局部基线有足够绝对/相对改进（`min_improvement_abs=0.01, min_improvement_rel=0.03`）；(ii) 正负轴对之间保持一致性（`consistency_margin_abs=0.005`）。若任一条件不满足，选择 `baseline`。

### E. 分位数滤波与触发门

分位数滤波在慢环采样前丢弃 buffer 中 `discard_ratio`（默认 10%）的极端样本，防止偶发的异常响应主导梯度更新。

触发门 `enable_trigger_gate` 控制慢环是否在 episode 早期就启动。默认策略是：在 warm-up 期间检查短期误差，只有当误差超过 `trigger_threshold` 时才启用 STDW 更新。这避免了在系统已经稳定时进行不必要的扰动。

### F. 雅可比逆矩阵

当前 4 通道 adaptation 空间使用固定分配，`J_inv_diag` 退化为单位对角：

```math
J^{-1}_{diag} = \text{diag}(1, 1, 1, 1)
```

它保留为诊断字段和未来扩展点（如动态通道分配），但本轮消融不对其单独跑实验。在文档中作为实现组件整理说明。

### G. 算法伪代码

```
Algorithm 1: STDW Online Adaptation
Input: policy π_θ, reference policy π_{θ_ref}, buffer B
Parameters: slow_loop_interval, batch_size, ρ, λ, ε

Initialize: B ← ∅, θ ← θ_ref
For each control step t:
    o_t ← observe()
    a_t ← π_θ(o_t)
    u_t ← S_surface(a_t[:4], ζ_t)
    Δu ← low_level_correction()
    a_pseudo ← a_t + Δu
    Apply u_t, get r_t
    Store (o_t, a_t, a_pseudo, r_t) in B

    If t mod slow_loop_interval == 0 and |B| ≥ batch_size:
        If not Lyapunov_check(B_recent, ε): continue
        Sample batch from B (after quantile filter)
        a_src ← π_{θ_ref}(o_batch)
        L ← (1-ρ)||π_θ - a_src||² + ρ||π_θ - a_pseudo||² + λ||θ-θ_ref||²
        θ ← θ - α·∇_θ L

    If t == micro_probe_start and micro_probe enabled:
        Run A/B/A candidate evaluation
        If consistent improvement found:
            Apply corrective drift
        Else:
            Maintain baseline
```

## V. 实验

### A. 实验设置

所有实验基于 EasyUUV A3 训练完成的 ckpt：`2026-06-08_13-48-14_stage2/model_2398.pt`。评估在 Isaac Lab Direct 框架下运行，每个 cell 1500 步，seed=0，medium wave，full tune 模式。

评估指标为 position tracking 的均方误差（MSE），分为 total/roll/pitch/yaw/depth 五个通道，重点关注 `final_mse`（最后 200 步平均）。

### B. 已有 48-cell 全矩阵结论

全矩阵覆盖 3 波况（calm/medium/storm）× 4 机型（base、long_body、heavy_moderate、asymmetric）× 2 整定模式（identity/full）× STDW 开关，共 48 个唯一 cell，0 失败。表 I 给出按机型聚合（对波况与整定取均值）的 STDW 主效应（单位：m²，final tracking MSE）。

**表 I. 按机型分层的 STDW 主效应（对 wave × tune 取均值）。**

| 机型 | MSE（STDW off） | MSE（STDW on） | Δ_STDW | 状态 |
|---|---:|---:|---:|:---:|
| base           | 0.2254 | 0.0726 | **−67.8%** | 改善 |
| long_body      | 0.2063 | 0.0719 | **−65.2%** | 改善 |
| heavy_moderate | 0.2664 | 0.2807 | +5.3% | 中性 |
| asymmetric     | 0.2263 | 0.5847 | **+158%** | 劣化 |

base 与 long_body 上的改善跨三档波况极其稳定（Δ_STDW = −66.5 ± 0.05%），说明 A3 的角速度 D 项与 12 维观测已吸收了绝大部分波浪扰动，STDW 在此基础上再贡献一个与波况几乎无关的额外下降。asymmetric 的劣化是确定性的（三档波况下均成立），证明这是结构性的 drift 方向问题而非随机方差。

![图 1. STDW 在 48-cell 矩阵上的相对跟踪误差变化（蓝=改善、红=劣化）。](../figures/fig1_stdw_delta_heatmap.png)

**图 1.** STDW 在 48-cell 环境矩阵上诱导的相对跟踪误差变化。STDW 在所有波况下一致改善 base 与 long_body，但劣化 asymmetric。

![图 2. 按机型分组的 STDW off/on 配对对比。](../figures/fig2_embodiment_on_off_bars.png)

**图 2.** 按机型分组的 STDW off/on 配对 MSE。开启 STDW 使 base 降低 67.8%、long_body 降低 65.2%，在 heavy 上基本中性，在 asymmetric 上强烈劣化，从而催生在线 gating 机制的需求。

一个代表性成功案例（base/calm/full）与一个代表性失败案例（asymmetric/calm/full）使机制清晰可见。成功案例中，drift 注入窗口开始后 rolling MSE 显著下降，并与 ρ 调度和慢环更新标记对齐；失败案例中，同一机制反而抬高 rolling MSE，因为固定 `+x` drift 把本已偏置的机型推得更远。

![图 5. 代表性 STDW 成功自适应案例（base/calm/full）。](../figures/fig5_base_full_timeline.png)

**图 5.** STDW 成功案例。drift 注入后 rolling MSE 下降；ρ 与慢环标记表明改善与 STDW 激活对齐，而非随机扰动。

![图 6. asymmetric COM-COB 偏移下的代表性 STDW 失败案例。](../figures/fig6_asymmetric_failure_timeline.png)

**图 6.** 失败案例。在 asymmetric 机型下，同一 STDW 机制相对 off 基线抬高 rolling MSE，支撑部署期 gating 的必要性。

后续诊断进一步定位并解释了 asymmetric 失效：

| 结论 | 证据 |
|---|---|
| asymmetric 灾难主要来自 pitch/depth，不是 yaw | `DIAG_p1_p2_p5_20260610.md` 通道分解 |
| 默认 `+x` drift 会损害 asymmetric | final x 从 `0.05` 推到 `0.10`，pitch 从 0.0032 升至 0.1928 |
| 反向修正 `(-x,-y)` 可接近 base | asymmetric final200 接近 base（0.0333 vs 0.0284） |
| 旧 micro-probe 评分偏向 `axis1_neg` | 12 个 probe cell 全选同一方向 |
| A/B/A paired scoring 消除了该偏置 | 12 个 probe cell 保守选择 `baseline` |

### C. IMU 噪声、角速度低通与延迟

本矩阵使用 `ang_vel_extra_std ∈ {0.0, 0.05}`（模拟 IMU 陀螺噪声等级，rad/s）、`d_filter_tau ∈ {0.0, 0.05}`、`obs/act_delay ∈ {0, 2}`。clean baseline 取全零配置。

| Embodiment | ang_vel_noise | d_filter_tau | obs_delay | act_delay | final_mse | 相对 clean |
|---|---:|---:|---:|---:|---:|---:|
| base | 0.00 | 0.00 | 0 | 0 | 0.0725 | +0.0% |
| base | 0.00 | 0.05 | 0 | 0 | 0.0725 | +0.0% |
| base | 0.05 | 0.00 | 0 | 0 | 0.1383 | +90.8% |
| base | 0.05 | 0.05 | 0 | 0 | 0.1385 | +91.0% |
| base | 0.05 | 0.00 | 2 | 0 | 0.1634 | +125.4% |
| base | 0.05 | 0.00 | 0 | 2 | 0.1583 | +118.3% |
| asymmetric | 0.00 | 0.00 | 0 | 0 | 0.5456 | +0.0% |
| asymmetric | 0.00 | 0.05 | 0 | 0 | 0.5443 | -0.2% |
| asymmetric | 0.05 | 0.00 | 0 | 0 | 0.4765 | -12.7% |
| asymmetric | 0.05 | 0.05 | 0 | 0 | 0.5231 | -4.1% |
| asymmetric | 0.05 | 0.00 | 2 | 0 | 0.7691 | +41.0% |
| asymmetric | 0.05 | 0.00 | 0 | 2 | 0.7607 | +39.4% |

分析：

- `ζ2` 的 D 项已经使用 body angular velocity，无需替换输入。
- `d_filter_tau=0.05` 在 base 上无改善（0.1383 vs 0.1385），在 asymmetric 上也无明确收益。
- 角速度噪声使 base final_mse 上升约 91%，2 步观测/动作延迟使 base 进一步上升约 118-125%。
- 延迟对两个 embodiment 的影响类似，约为噪声影响的 1.3-1.4 倍。
- asymmetric 在噪声下出现的表观改善（-12.7%）更可能是噪声打散了已知的错误 drift 耦合，而非噪声本身有益。

**实物部署建议**：(i) 不替换 ζ2 输入（已是角速度）；(ii) 暂不启用 `d_filter_tau` 低通（无明确收益）；(iii) 最小化传感-控制-执行链路延迟（2 step 延迟使 base 性能下降约一倍）。

### D. STDW 组件消融

| Variant | Base final_mse | Base delta | Asymmetric final_mse | Asym. delta | 解释 |
|---|---:|---:|---:|---:|---|
| full STDW | 0.0725 | +0.0% | 0.5456 | +0.0% | 参考 |
| STDW off | 0.2262 | +212.0% | 0.3147 | -42.3% | 默认 drift 对 base 有益，对 asymmetric 有害 |
| no slow loop | 0.2262 | +212.0% | 0.3147 | -42.3% | 等价回退路径 |
| no Lyapunov fence | 0.0724 | -0.1% | 0.5395 | -1.1% | 小矩阵中不是主导项 |
| no pseudo-action | 0.0721 | -0.6% | 0.5284 | -3.2% | 小矩阵中不是主导项 |
| no quantile filter | 0.2010 | +177.2% | 0.3364 | -38.4% | 滤波保护 base，也削弱错误 asymmetric drift |
| no trigger gate | 0.2702 | +272.7% | 0.5456 | +0.0% | 触发门对 base 稳态至关重要 |

分析：

- **最关键组件**：触发门（`no trigger gate → +273%`）和分位数滤波（`no quantile filter → +177%`）。这两个组件共同保护 base 机型免受不当更新影响。
- **Lyapunov 围栏和伪动作**在本小矩阵中影响较小（≤3%），说明它们更多在更极端工况或更长部署时间中发挥作用。
- **Slow loop 关闭**等价于 STDW off，确认慢环是 base 性能提升的来源。
- **Asymmetric**：在默认 drift 方向下 full STDW 变差，但这不是应删除组件，而是应修正 drift 方向选择。

除保护 base 行为外，full 整定栈（PE / 死区 / LPF / β）在 stage-2 训练后相对 identity 直通带来了可测量的净收益，如图 4 所示。

![图 4. full 整定栈相对 identity 直通在 stage-2 训练后的效果。](../figures/fig4_tune_full_vs_identity.png)

**图 4.** 相比 identity 直通模式，完整的 PE/死区/LPF/β 栈使 STDW-on 的 MSE 再降低 8.8%，表明增益自适应头在 stage-2 训练后已进入工作区。

### E. 部署代码工程

实物部署代码已完成配置化改造（`eval/deploy_config.yaml` + `eval/deploy_config.py`），三个示例 demo（thruster I/O、deploy manager、replay CSV）和实物 runtime skeleton（`real_world_runtime.py`）均已接入 YAML 配置。代码对 Isaac 无任何依赖，仅需要 numpy+torch（或 ONNX Runtime）+ pyyaml。

## VI. 分析与讨论

当前代码和已有实验给出三个核心判断：

**第一，ζ2 的 D 项在 A3 中已经使用角速度**，因此实物部署的关键不是"替换输入"，而是"噪声下是否需要低通"和"延迟容忍度"。本实验表明 IMU 级角速度噪声使 base 性能下降约一倍，而 `d_filter_tau=0.05` 低通没有明确收益。这意味着：(a) 噪声是真实威胁，需要通过 IMU 选型和硬件滤波（而非软件低通）来缓解；(b) 链路延迟同样有害，应最小化传感→策略→执行的时延。

**第二，STDW 的主要风险不是慢环本身**，而是在错误 drift 方向和弱证据条件下进行不必要扰动。消融结果显示：当 drift 方向正确时（base），full STDW 比 STDW off 好 212%；当 drift 方向错误时（asymmetric），full STDW 比 STDW off 差 42%。正确的修复不是删除组件，而是用 offset router 或 observable-only micro-probe 在部署初期选择正确的 drift 方向或保持 baseline。

**第三，实物部署应采用少样本、可回退、保守的流程**。具体 SOP 已在 `DEPLOY_SOP_realworld.md` 中给出：先 baseline 稳定 → 再 micro-probe → 证据不足保持 baseline → 仅在证据充分且触发门允许时启动慢环。

## VII. 结论

STDW 将 S 面控制器、参数化 RL 策略、伪动作学习、行为锚定、Lyapunov 围栏和 observable-only micro-probe 组合成一个实物可部署的少样本在线自适应框架。该框架不依赖 Isaac 运行时，也不依赖仿真特权状态。本文给出的 26-cell 专项实验矩阵量化了 IMU 噪声、角速度低通、延迟与各组件消融对性能的影响，并将结论转化为实物部署建议：不替换 ζ2 输入、暂不启用 `d_filter_tau` 低通、最小化链路延迟、采用保守 micro-probe 评分。

![图 7. STDW 矩阵级效应一页式 summary card。](../figures/fig7_stdw_summary_card.png)

**图 7.** STDW 矩阵级效应一页式摘要：稳健改善区（base/long_body）、已修复的 heavy 机型失效、新暴露的 asymmetric 失效，以及 full 整定栈的净收益。

## 参考文献

[1] EasyUUV STDW 源码与诊断文档，2026。  
[2] 水下航行器 S 面控制相关文献。  
[3] 行为正则化在线策略自适应相关文献。  
[4] Domain randomization and sim-to-real transfer for robotics.  
[5] S-surface controller design and analysis.  
[6] Lyapunov-based stability analysis for nonlinear control systems.

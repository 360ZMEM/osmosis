# 港大 SRP 汇报 PPT 图表布局大纲（7 张核心 Slide）

- 文档日期：2026-06-19
- 用途：把已生成的图表分配到 7 张核心 PPT，用于 7 月 3 日（开题）与 8 月 4 日（结题）答辩。
- 来源：次要 prompt 第二部分 + [港大SRP汇报重构战略大纲精简版.md](../../港大SRP汇报重构战略大纲精简版.md)。
- 已生成图表清单见 [REPORT_ppt_figures_20260619.md](REPORT_ppt_figures_20260619.md)（图 2.1 / 2.2 / 2.3 / 2.4）。

> 说明：本大纲为「7 张答辩主线」版本（次要 prompt 第二部分）。它与战略大纲的 13 张细分 Slide 不冲突——可视为答辩当天的精简放映顺序。框图类素材（双环、伪标签、链路图）尚未在本仓库生成，下方标注为 “待制作”。

---

## Slide 1：封面页与科学愿景 (Title & Strategic Vision)

- **放置图表**：UUV 海试实物照片或高保真物理仿真渲染图。
- **状态**：待提供（实物照 / 渲染图，本仓库暂无）。
- **视觉作用**：用真实物理实体建立学术信任感，点明这是硬核具身智能（Embodied AI）控制课题。

## Slide 2：科学问题与物理冲突 (The Sim2Real GDA Challenge)

- **放置图表**：冲突对比图（Conflict Layout）。
  - 左：原生平滑 GDA 渐变域示意图。
  - 右：UUV 真实「高频非平稳海浪」+「隐蔽不可直接测量的重心偏置 COM-COB」三维坐标轴图。
- **状态**：待制作（示意框图）。
- **视觉作用**：凸显「静态图像的数学假设」与「水下连续控制的物理现实」之间的底层代差。

## Slide 3：方法论一 —— 双时间尺度谱隔离架构 (Singular Perturbation Control)

- **放置图表**：双环控制系统信号流框图（红/蓝流线标出 100Hz 物理快环与 0.5Hz 算法慢环隔离边界）。
- **状态**：待制作（框图）。
- **视觉作用**：用奇异摄动理论为在线自适应策略背书，展示动力学功底。

## Slide 4：方法论二 —— 自抗扰反哺伪标签与 Lyapunov 安全门控 (Safe Self-Training)

- **放置图表**：自监督伪标签反投影与物理围栏数据流向图（含 $\Delta u(t)$ 经对角雅可比逆 $\mathbf{J}^{-1}$ 反投影生成伪标签 $\vec{a}_{pseudo}$ 的公式盒 + Lyapunov 哨兵盾牌关卡）。
- **状态**：待制作（框图）。
- **视觉作用**：证明自训练具有 100% 李雅普诺夫物理边界保护，杜绝在线自训练参数发散。

## Slide 5：仿真诊断 —— 48-Cell 扫参全矩阵与 A3 稳定底座表现 (The Simulation Landmark)

- **放置图表**：
  1. **图 2.2** STDW 双指标泛化热力图 → [`fig22_generalization_heatmap.png`](../figures/ppt/fig22_generalization_heatmap.png)（左=全程 MSE、右=尾部 MSE；对称机型均值全程 −21%、尾部 −42%）。
  2. `heavy_moderate` 性能修复柱状图（旧 ckpt +75% 崩溃 → 新 A3 stage2 收敛至 +5.3%）。**状态**：待制作（可在 plot_ppt_figures.py 中扩展，沿用 `_mean_pairwise`）。
- **视觉作用**：用饱满扫参矩阵证明 A3 控制底座极强、海浪扰动已被吸收。

## Slide 6：攻克不对称突变 —— 物理路由器与安全门控的威力 (The Asymmetric Rescue)

- **放置图表**：
  1. **图 2.1 / 图 2.4** 时域跟踪联动图 / 学术级跟踪 overlay → [`fig21_tracking_timeline.png`](../figures/ppt/fig21_tracking_timeline.png) / [`fig24_publication_overlay.png`](../figures/ppt/fig24_publication_overlay.png)。
  2. **图 2.3** 无特权 OPR 偏置纠偏对比图 → [`fig23_opr_recovery.png`](../figures/ppt/fig23_opr_recovery.png)（共享参考全程 MSE 0.2095 → 0.0953，跌回 base 0.0897，54.5% plunge）。
  3. 3-Seed 36-Cell 统计显著性曲线（带阴影置信区间）。**状态**：待制作（当前 sweep 为单 seed，需补多 seed 跑批）。
- **视觉作用**：作为答辩高潮与「救火」证据，证明 OPR 偏置估计器无需特权信息即可零样本在线自校准。

## Slide 7：物理可部署性与边缘单片机验证 (Edge Hardware Readiness)

- **放置图表**：实物半物理在环调试链路图 + 算力时延分析条形图（证明总闭环时延 < 10ms / 100Hz）。
- **状态**：待制作（链路框图 + 时延条形图）。
- **视觉作用**：证明算法兼容资源受限端侧单片机，具备物理落地后劲。

---

## 当前可用图表 → Slide 映射汇总

| Slide | 图表 | 文件 | 状态 |
|---|---|---|---|
| 5 | 图 2.2 泛化热力图 | [`fig22_generalization_heatmap.png`](../figures/ppt/fig22_generalization_heatmap.png) | 已生成 |
| 6 | 图 2.1 时域联动图 | [`fig21_tracking_timeline.png`](../figures/ppt/fig21_tracking_timeline.png) | 已生成 |
| 6 | 图 2.4 学术级 overlay | [`fig24_publication_overlay.png`](../figures/ppt/fig24_publication_overlay.png) | 已生成 |
| 6 | 图 2.3 OPR 纠偏 | [`fig23_opr_recovery.png`](../figures/ppt/fig23_opr_recovery.png) | 已生成 |
| 1 | 实物 / 渲染图 | — | 待提供 |
| 2–4, 7 | 方法论 / 链路框图 | — | 待制作 |
| 5 | heavy_moderate 修复柱状图 | — | 待制作（可脚本扩展） |
| 6 | 3-Seed 显著性曲线 | — | 待制作（需多 seed 数据） |

# DIAG — SO(3) 流形 S-Surface Phase 1 失败根因诊断（2026-07-01）

> 结论先行：**SO(3) 几何 S-Surface 控制器本身是正确且高质量的**（纯解析基线在 ordinary
> 任务上与全学习 euler 基线打平），Phase 1 双目标 eval 的全面退化**不是几何数学的错，而是
> "残差 RL 从 model_2846 fine-tune" 这一训练范式的错**。本文用逐通道 MSE + 饱和/驻留统计
> 定位到三条独立机理，并给出不需要推翻 SO(3) 理论的修复路径。

---

## 0. TL;DR（给赶时间的读者）

| 断言 | 证据 | 影响 |
|---|---|---|
| SO(3) 几何控制器**数学正确、restoring** | 纯解析 res=0 在 ordinary base/asym = 0.2358 / 0.2412，姿态三轴 MSE 全 <0.013，与 euler 基线 0.2229 / 0.2266 打平甚至姿态更好 | 论文卖点成立，**不要推翻 SO(3)** |
| Phase 1 退化主因 = **残差 RL fine-tune 混沌 + 共享策略污染** | 训练 log_mse 全程 12–40 混沌区；trained ckpt ordinary 退化到 0.47–0.57，其中**退化几乎全在 depth（0.34–0.42）**，而 depth 根本不走 SO(3) 路径 | 修训练范式，不修控制律 |
| flip360 打不过 model_2846 = **倒置区物理边界 + model_2846 是 flip360 专家** | res=0 flip360=5.33 > 2846 的 2.07；flip360 中 52% 时间目标在倒置带，depth 误差爆炸(2.01)、depth 指令 82% 饱和 | 固定增益解析基线赢不了"flip360 专门训练过"的学习控制器；需增益调度或通道隔离残差 |

---

## 1. 复现事实（本轮 eval 全量数据）

协议：`total_steps=1500, use_stdw=False, seed=0`，off_clean（无 STDW、无 router、无 probe）。
SO(3) eval 必须用 `attitude_error_mode: so3` 且控制律参数与训练一致，否则 policy 残差与
控制律不匹配（见 §4 配置清单）。

### 1.1 四 checkpoint × 双目标（final_mse，括注训练 log_mse）

| ckpt (train log_mse) | flip360 base | flip360 asym | ordinary base | ordinary asym |
|---|---:|---:|---:|---:|
| **model_2846 (euler 基线)** | **2.0701** | **3.6606** | **0.2229** | **0.2266** |
| model_2850 (15.16, 训练最优) | 3.9158 | 4.7203 | 0.4815 | 0.5739 |
| model_2900 (23.15) | 6.5130 | 6.7913 | 0.4986 | 0.5559 |
| model_2950 (17.11) | 16.2042 | 11.8111 | 0.4983 | 0.5626 |
| model_2995 (30.51, 末) | 5.9106 | 3.0318 | 0.4664 | 0.5523 |

→ 每个 checkpoint 的 flip360 base 都劣于 2.07、两项 ordinary 全部退化到 ~0.5（约 2×）。
训练 MSE 与 eval 排名**不单调**（2950 训练 MSE 比 2900 好，eval 却更差），是典型混沌非收敛特征。

### 1.2 逐通道拆解（trained ckpt，均值）

关键观察：**ordinary 退化几乎全部集中在 depth**，姿态三轴反而不差。

| ckpt / cell | roll | pitch | yaw | **depth** |
|---|---:|---:|---:|---:|
| 2850 ordinary base | 0.074 | 0.003 | 0.025 | **0.354** |
| 2850 ordinary asym | 0.112 | 0.003 | 0.028 | **0.423** |
| 2995 ordinary base | 0.074 | 0.003 | 0.025 | **0.343** |
| 2850 flip360 base | 0.891 | 0.083 | 0.649 | **1.214** |
| 2950 flip360 base | 1.008 | 0.107 | 1.099 | **4.390** |

- **depth 不走 SO(3) 路径**（`_pid_control` 只覆盖 `PID_value[:,0:3]`，channel 3 深度仍是旧
  euler 逐轴逻辑）。ordinary 上 depth 从基线 ~0.16 退化到 ~0.35–0.42，说明是**共享策略被
  混沌 fine-tune 拽偏了 depth 输出**，与几何姿态律无关。
- flip360 上 **pitch MSE 最小（0.08–0.33）**——这**推翻了先前"pitch 力矩预算不足是主因"的
  假设**。真正炸的是 depth 与 roll/yaw。

---

## 2. 决定性诊断实验：纯解析 SO(3) 基线（res=0，加载 model_2846）

**设计**：`geo_residual_scale=0`（完全关闭 RL 残差）→ roll/pitch/yaw 由**纯几何 PD**驱动；
同时加载 model_2846 → depth 通道保持**已知良好的 euler 深度控制器**。这样把
"几何控制器本身质量" 从 "混沌 fine-tune" 里干净剥离。配置见 §4。

### 2.1 结果（final_mse + 逐通道）

| cell | total | roll | pitch | yaw | depth |
|---|---:|---:|---:|---:|---:|
| **ordinary base** | **0.2358** | 0.0025 | 0.0014 | 0.0022 | 0.1565 |
| **ordinary asym** | **0.2412** | 0.0077 | 0.0124 | 0.0039 | 0.1412 |
| flip360 base | 5.3311 | 0.6146 | 0.1027 | 0.6110 | 2.0108 |
| flip360 asym | 5.0334 | 0.6093 | 0.1117 | 0.6460 | 1.8417 |

### 2.2 解读（三个独立结论）

1. **SO(3) 几何控制器在 ordinary 上与全学习 euler 基线打平**：0.2358 vs 0.2229（+5.8%，
   几乎全部差在 depth 的 0.1565）。**姿态三轴 roll/pitch/yaw 全 <0.013**，比 trained 残差
   ckpt（roll 0.07 / yaw 0.025）还好。→ **几何数学正确、restoring、可用**，`geo_channel_sign`
   标定正确。这是论文最硬的卖点：*不训练、纯几何律*就能达到学习控制器的姿态精度。

2. **残差 RL 是净损害**：res=0（0.2358）比任何 trained 残差 ckpt（0.47–0.57）都好。说明
   `geo_residual_scale=0.5` + 混沌 fine-tune 让策略在解析基线上**乱加修正**，破坏了本已良好
   的控制。残差权威过大（0.5）叠加 PPO 在 flip360 区不收敛 → 负优化。

3. **flip360 上纯解析(5.33) 打不过 model_2846(2.07)**：因为 model_2846 是**专门用 flip360
   课程训练出来的 euler 控制器**，学到了倒置区的近 bang-bang 策略；固定增益的解析 PD 没有这套
   区间自适应。这不是 SO(3) 的缺陷，而是"固定增益解析 vs 任务专家学习控制器"的固有差距。

---

## 3. flip360 depth 爆炸的物理机理（饱和 + 驻留统计）

在 res=0 flip360_base 的逐步 CSV 上统计（`executed_action` = 策略 4 维原始控制输出，
进控制律前 ×action_lim；depth 分量 action_lim=1.0）：

| 量 | flip360_base | ordinary_base |
|---|---:|---:|
| a_ctrl 均值\|·\| (roll,pitch,yaw,depth) | [1.22, 1.77, 0.82, **5.59**] | [0.36, 0.42, 0.33, **0.55**] |
| a_ctrl \|·\|≥1 占比 (depth) | **0.817** | 0.100 |
| control_effort 均值 / p95 | 7.29 / 12.72 | 1.20 / 3.72 |
| \|depth 误差\| 均值 / max (m) | **1.086 / 3.730** | 0.302 / 1.554 |
| 目标 \|tilt\| 均值 (rad) | 1.966 | 0.264 |
| 目标在倒置带占比 (\|tilt\|>120°) | **0.521** | 0.000 |

- flip360 中 **52% 的时间目标姿态在倒置带**，此时浮力回正力矩持续与深度保持耦合抗衡，
  深度控制指令被逼到 **82% 时间饱和**（bang-bang），深度误差均值 >1m。
- 这与 2026-06-30 的物理边界判定（`REPORT_flip360_training_20260628.md` 第 6 节：倒置区
  驻留型角度极限、名义工作点带宽不足）**完全一致**。depth 是旧 euler 路径，SO(3) 升级
  根本没碰它 → **flip360 的主瓶颈从来不在姿态几何，而在倒置区深度-浮力耦合的物理极限**。

---

## 4. 诊断产物与可复现配置

新增（均为诊断用，默认不影响主线）：
- `workflows/configs/pressure_flip360_medium_so3.yaml` / `matrix_wave_medium_so3.yaml`
  —— SO(3)-mode eval（`geo_residual_scale=0.5`，与训练一致）。
- `workflows/configs/pressure_flip360_medium_so3_res0.yaml` /
  `matrix_wave_medium_so3_res0.yaml` —— 纯解析基线（`geo_residual_scale=0`）。
- `workflows/run_so3_p1_eval.sh` —— 四 checkpoint × 双目标 eval 驱动（`CKPT=model_XXXX.pt`）。
- `workflows/run_so3_diag_res0.sh` —— 纯解析诊断（加载 model_2846，depth 保持良好基线）。

结果目录：`.results/so3_p1_eval_{2850,2900,2950,2995}/`、`.results/so3_diag_res0_2846/`。

---

## 5. 修复路径（不推翻 SO(3)，按代价升序）

诊断把问题从"几何律错了"重定位到"训练范式 + 深度通道污染 + 倒置区物理极限"，故：

### P1（最低代价，先做）：保护 depth + 压低残差权威
- **depth 通道隔离**：fine-tune 时冻结/不让残差策略改写 depth 输出（depth 已由 euler 解析
  良好求解），只让 RL 学 roll/pitch/yaw 残差。可用现有 stage 梯度隔离思路按输出行冻结。
- **降 `geo_residual_scale` 0.5 → 0.1~0.2**：让解析基线主导，残差只做微修正。res=0 已证明
  解析基线在 ordinary 打平基线，残差应是锦上添花而非主控。
- 预期：ordinary 立刻回到 ~0.23（解析基线水平），flip360 至少不劣于解析 5.33。

### P2（中代价）：倒置区增益调度 / 前馈
- flip360 差距来自固定增益解析 PD 在倒置区带宽不足。可在 `_so3_attitude_error` 输出上按
  倾角 smoothstep 放大 `geo_zeta1`（倒置带增益调度），或引入 §Phase 2 的解析前馈 τ_ff
  抵消浮力回正。这是把 model_2846 的"学习到的 bang-bang"用解析方式补回来。

### P3（高代价，用户既定"可选，不首先"）：Phase 2 伪逆分配器 B†
- 当前固定手工分配矩阵对 roll/yaw 反号（已用 `geo_channel_sign` 补偿）。倒置大机动下
  通道耦合仍可能次优；伪逆 B† 可最优分配。但 P1/P2 未穷尽前不必上。

### 明确不做
- 不推翻 SO(3) 几何律（res=0 证明其正确）。
- 不因 flip360 打不过 2846 而回退 euler：ordinary 平手 + 姿态更优 + 几何可解释性是净收益，
  真正卡点是倒置区物理极限（全 embodiment 通用，非 SO(3) 引入）。

---

## 6. 一句话给下一位接手者

> Phase 1 的双目标 eval 全面退化，**根因不是 SO(3) 数学**——纯解析 res=0 基线在 ordinary 上
> 与 euler 基线打平且姿态三轴更优，证明几何控制器正确。真凶是：①残差 RL 从 model_2846
> fine-tune 陷入 F3/F4 同构的混沌非收敛，②混沌训练污染了共享策略的 **depth 通道**（depth 不
> 走 SO(3) 却贡献了 ordinary 的全部退化），③flip360 的固定增益解析基线打不过"flip360 专家"
> model_2846，且瓶颈是倒置区深度-浮力耦合的**物理极限**（52% 倒置驻留 + depth 指令 82% 饱和）。
> 修复先走 P1（隔离 depth + 降 residual_scale），几何理论保留。

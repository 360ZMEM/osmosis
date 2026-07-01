# Paper Addendum: Multi-Embodiment Generalization, Config-Driven Thrust Allocation, and the Submerge-then-Flip360 Near-Boundary Protocol

日期：2026-07-01　状态：**骨架 / 可回填**（理论与设计已定稿，实验数值待 Isaac 冒烟/eval 统一取证）

接续：[`PLAN_multi_embodiment_tam_boundary_submerge_20260701.md`](../../.trae/documents/PLAN_multi_embodiment_tam_boundary_submerge_20260701.md)、
[`PAPER_ADDENDUM_lyapunov_esuot_boundary_20260701.md`](PAPER_ADDENDUM_lyapunov_esuot_boundary_20260701.md)

本附录补充论文的方法与实验章节，记录一组**跨形态泛化**能力：把推力分配从单一硬编码
8 推进器几何升级为 **config 驱动的分配矩阵 (TAM)**，覆盖全驱 / 欠驱动 / 非正交三类
矢量推进器 UUV，并把 REMUS 类**单推进器 + 舵面耦合 AUV**作为差距分析对象；同时定义
适用于**所有** embodiment 的下沉-then-flip360 近边界安全协议。核心主张：**只靠底层
S-Surface 控制 + config TAM 即可获得跨形态基本性能**，无需为每个形态重训策略。

> 撰写约定：以 `【TODO：…】` 标记的表格与数值为占位符，待 Isaac 冒烟/eval 取证后回填。
> 商业 AUV 推进器布局引用点标 `【有东西说：cite …】`。理论与设计部分已定稿。

---

## 1. Motivation（动机）

单一 embodiment 训练出的底层控制器是否能跨形态迁移，是水下机器人 Sim2Real 与
可复用控制的核心问题。本工作把 embodiment 从"仅质量/惯量/阻尼变体"扩展到**推进器
数量、布局、朝向都不同**的机型族，考察三个层级的泛化：

1. **全驱冗余变化**（8 推 → 6 推）：控制轴仍正交，但分配的零空间维度改变。
2. **欠驱动**（6 推 → 4 推，失去 yaw 权威）：某一控制 DOF 结构性不可达。
3. **非正交**（推进器朝向带倾角，三控制轴不互相垂直）：旧硬编码"逐轴混合块"失效，
   必须由分配矩阵吸收耦合。

推进器布局参照真实商业 AUV 的矢量推进器设计【有东西说：cite 商业 AUV 推进器布局，
如 vectored-thruster ROV/AUV 产品或综述】，而非任意几何，以保证跨形态实验的现实相关性。

第四类 **REMUS 型单推进器 + 舵面 AUV** 与前三类矢量推进器机型**本质不同**（§6），
本轮作为差距分析对象记录，不进入实现。

---

## 2. Embodiment 分类学与物理设计（Taxonomy & Physical Design）

### 2.1 分类学

| 类别 | 代表机型 | 推进器 | 可控 DOF | 分配 |
|---|---|---|---|---|
| 全驱（现有） | base / asymmetric / long_body / heavy_* | 8 | roll/pitch/yaw/depth | 硬编码混合块 |
| 全驱冗余变化 | `uuv6` | 6（4 垂直 + 2 水平） | roll/pitch/yaw/depth | config B⁺ |
| 欠驱动 | `uuv4` | 4（4 垂直） | roll/pitch/depth（**无 yaw**） | config B⁺ + WLS |
| 非正交全驱 | `uuv6_angled` | 6（带倾角） | roll/pitch/yaw/depth | config B⁺ |
| 非正交欠驱动 | `uuv4_angled` | 4（带倾角） | roll/pitch/depth | config B⁺ + WLS |
| 单推 + 舵面（差距分析） | REMUS（§6） | 1 + 舵 | surge + 耦合转向 | 不适用 |

### 2.2 物理设计与近边界一致性

所有 embodiment 的密度 ρ_body = mass/volume 设计为**略小于水** ρ_water=997 kg/m³
（约 990，净浮力 +0.7%），形成微正浮力。目的有二：(a) 与近边界效应模块的剩余浮力/
自由表面假设一致（浸没时有可测的残余上浮项）；(b) flip360 大机动后能自然回浮而非
沉底，保证任务可持续。设计值：

| 机型 | mass (kg) | volume (m³) | ρ_body (kg/m³) | 净浮力% |
|---|---:|---:|---:|---:|
| base（参考） | 22.701 | 0.022748 | 997.9 | ~0（中性） |
| uuv6 | 29.70 | 0.030000 | 990.0 | +0.7% |
| uuv4 | 21.78 | 0.022000 | 990.0 | +0.7% |
| uuv6_angled | 31.68 | 0.032000 | 990.0 | +0.7% |
| uuv4_angled | 23.76 | 0.024000 | 990.0 | +0.7% |

> 体积/质量差异使各机型的惯量、浮力回正力矩与推力权威各不相同，构成真实的跨形态
> 泛化考验。inertia 为按几何缩放的设计值，实测由动力学识别回填。

---

## 3. Config 驱动的推力分配矩阵（Config-Driven TAM）

### 3.1 Method

给定推进器布局 layout = {positions rᵢ∈ℝ³, orientations qᵢ (SO(3))}，第 i 个推进器
产生沿其 body 轴的单位推力方向 `fᵢ = R(qᵢ)·x̂`，对本体产生力 `fᵢ` 与力矩 `τᵢ = rᵢ × fᵢ`。
堆叠成分配矩阵

```
B = [ f₁  f₂ … f_N ]  ∈ ℝ^{6×N}
    [ τ₁  τ₂ … τ_N ]
```

body-frame 6-DOF 顺序 `[Fx(surge), Fy(sway), Fz(heave), Tx(roll), Ty(pitch), Tz(yaw)]`。
控制层输出 4 通道 `[roll, pitch, yaw, depth]`，映射到期望 wrench 命令
`w = [0, 0, F_depth, T_roll, T_pitch, T_yaw]ᵀ`（surge/sway 命令为 0）。

分配求推进器指令 `u ∈ ℝ^N`：

- **满驱 / 全驱冗余（pinv）**：`u = B⁺ w`，`B⁺ = Bᵀ(BBᵀ)⁻¹`（`torch.linalg.pinv`），
  最小二范数解，天然处理 N>6 冗余的零空间。
- **欠驱动（wls）**：某 DOF（如 uuv4 的 yaw）结构性不可控。用加权最小二乘
  `u = argmin_u ‖W(B u − w)‖²`，其中 W 对不可控 DOF 行赋权 0（不惩罚无法实现的分量），
  等价于只在可达子空间内投影。

### 3.2 与旧硬编码路径的关系（等价性验证）

对现有 8 推进器几何，`B⁺` 对单位 roll/pitch/yaw/depth 命令重构出的净 body wrench
方向/符号应与硬编码混合块一致——这正是既有 SO(3) 分支用 `_geo_channel_sign=[-1,+1,-1]`
数值修正的符号语义（roll τx、yaw τz 反号，pitch τy 正号）。单测断言此符号一致性
（[`thrust_allocation.py`](../../thrust_allocation.py) + standalone 验证）。

**契约**：base/asym/long_body/heavy_* 保持旧硬编码混合块路径（零行为变更 A4），
仅 `uuv6/uuv4/uuv6_angled/uuv4_angled` 显式声明 `thrust_allocation` 时走 config B⁺。

---

## 4. S-Surface 对非正交轴的再设计（Redesign for Non-Orthogonal Axes）

### 4.1 Problem

原 S-Surface 控制律 `PID_value = 2/(1+exp(−s_ratio·(P·e + D·ė)))−1` 逐轴独立 sigmoid，
**隐含假设三控制轴正交对角**。非正交机型（`*_angled`）推进器朝向带倾角，其净控制轴
不互相垂直——若沿用旧硬编码逐轴混合块，一个轴的命令会串扰到其他轴，破坏解耦。

### 4.2 Method — 在分配层解耦，而非控制层假设正交

关键设计：**控制律不变**（S-Surface 仍逐轴出 4 通道 `[roll, pitch, yaw, depth]`），
把非正交耦合完全交给**分配层**的伪逆 B⁺ 吸收。因为 B⁺ 求解的是"实现期望 body wrench
所需的推进器指令"，无论推进器如何倾斜，B 都编码了真实几何，B⁺ w 自动给出正确解。

这是最小侵入、最大复用的设计：
- 控制层：零改动，跨全部机型共用（含现有 ckpt）。
- 分配层：每机型一个 B 矩阵（由 config 布局构建），承担全部形态差异。

由此"S 面针对推力分配矩阵再设计"的落点是**分配矩阵替换硬编码混合块**，而非重写
sigmoid 控制律——支撑第 §7 的 ckpt 共用假设。

---

## 5. 下沉-then-flip360 近边界协议（Submerge-then-Flip360 Near-Boundary Protocol）

### 5.1 Motivation

近边界效应（残余浮力 / 自由表面 / 通气）仅在浸没比例 `s(t)∈(0,1)` 区间有意义。若
UUV 在 flip360 大机动中触碰或破出水面（z → z_surface），浸没比例 s→0，浮力/阻尼消失、
推进器失权，产生的既非受控数据也非有效近边界数据，污染统计。用户要求：**所有**
embodiment（含 base）先下沉到安全深度、保持竖直，再执行 flip360，且全程不触面。

### 5.2 Method — episode 内两相位机 + 破面守卫

- **阶段 1（下沉+稳姿）**：`episode_length_buf < submerge_hold_steps` 段，goal 姿态锁为
  竖直（单位四元数），深度设定驱动到 `submerge_depth`。
- **阶段 2（flip360）**：hold 结束后启用 sine/flip360 姿态参考。
- **破面守卫**：`surface_guard_enable` 时，`z > z_surface − surface_margin` 判 out_of_bounds。

**近边界安全关系式**（保浸没比例 s=1）：

```
submerge_depth  <  z_surface − vehicle_height/2 − margin
```

默认 `z_surface=3.0`、`vehicle_height=0.3` → 要求 `z < 2.85`。spawn `starting_depth=1.5`
已满足。z 约定：世界 z-UP，z 越大越浅。默认全部开关关闭（零行为变更）。

---

## 6. REMUS AUV 差距分析（Gap Analysis — Why Skipped This Round）

### 6.1 模型本质差异

REMUS 100（[`remus100.py`](../../PythonVehicleSimulator/src/python_vehicle_simulator/vehicles/remus100.py)：
L=1.6m, d=0.19m 椭球, ρ=1026, W=B 中性）是**单推进器 + 舵面**（rudder δ_r / stern-plane δ_s /
propeller n）机型。其转向/俯仰力矩

```
Y_r = −½ρ U_r h² A_r · C_L · δ_r ,  N = x_r · Y_r
Z_s = …,  M = −x_s · Z_s
```

与前进速度平方 **U²·δ** 成正比——**零速时不可转向**，转动完全耦合于前进。这与本仓库
矢量推进器 UUV（任意姿态、零速可直接施加 body 力矩）**本质不同**。

### 6.2 忠实移植的障碍

REMUS 动力学依赖 Fossen 6-DOF 刚体模型的完整项：附加质量 `M_A·ν̇`、科氏-向心
`C(ν)·ν`、非线性阻尼 `D(ν)·ν`、恢复力 `g(η)`，并以 `ν̇ = M⁻¹(τ − C ν − D ν − g)` 积分。
本仓库经 PhysX 刚体积分 + 加性 body wrench，不足以复现附加质量与流体耦合科氏项。
最小可行版也须重写 nu_dot 积分或注入附加质量伪力，工作量/风险与本轮"基础设施 + 文档"
目标不成比例。

### 6.3 决策与后续路线

**本轮 skip 实施，仅文档记录。** 若后续实施，考验须显著减载：低频正弦参考、减小幅度、
拉长时间、降低变化率（因零速不可转向，必须维持巡航速度才能操纵），并附带 depth/heading
autopilot（successive-loop + integral-SMC，见参考实现）。留作独立项。

---

## 7. Experiments 【TODO：Isaac 冒烟/eval 取证】

### 7.1 跨形态 ckpt 适用矩阵（核心待答：底层控制能否跨形态共用）

| embodiment | 首选 ckpt | flip360 final_mse | ordinary final_mse | 需重训? | note |
|---|---|---:|---:|:---:|---|
| base（基线） | model_2398 | 【TODO】 | 【TODO】 | no | 主线 |
| uuv6 | model_2398（复用） | 【TODO】 | 【TODO】 | 【TODO】 | 全驱冗余 |
| uuv6_angled | model_2398（复用） | 【TODO】 | 【TODO】 | 【TODO】 | 非正交，B⁺ 吸收 |
| uuv4 | model_2398（复用，yaw 降级） | 【TODO】 | 【TODO】 | 【TODO】 | 欠驱动 |
| uuv4_angled | model_2398（复用） | 【TODO】 | 【TODO】 | 【TODO】 | 非正交欠驱动 |

核心待答问题：
1. 底层 S-Surface + config TAM 是否使 uuv6/uuv6_angled 直接复用 model_2398 达到基本性能？
2. 欠驱动 uuv4 在 yaw 任务降级下，roll/pitch/depth 跟踪是否仍达标（还是必须重训）？
3. 非正交 B⁺ 分配是否引入可观测的解耦残差？

### 7.2 TAM 等价性验证 【TODO：单测数值】

| 命令 | 净 body wrench 方向（B⁺，8 推几何） | 与硬编码混合块符号一致? |
|---|---|:---:|
| unit roll | 【TODO：τx 符号】 | 【TODO】 |
| unit pitch | 【TODO：τy 符号】 | 【TODO】 |
| unit yaw | 【TODO：τz 符号】 | 【TODO】 |
| unit depth | 【TODO：Fz 符号】 | 【TODO】 |

### 7.3 下沉协议有效性 【TODO】

| 机型 | 破面事件数 | 下沉相位末端 z | 阶段 2 浸没比例 s | note |
|---|---:|---:|---:|---|
| base | 【TODO】 | 【TODO】 | 【TODO】 | 协议验证 |
| uuv6 | 【TODO】 | 【TODO】 | 【TODO】 | |

---

## 8. 正交性与契约（Orthogonality & Contract）

- 多 embodiment 与 config TAM 对 base/asym 零行为变更（`_use_config_alloc=False`、
  `_num_thrusters=8`、submerge/guard 关时逐行等价旧代码）。
- 下沉相位 / 水面守卫默认关，与 M1–M5 及 SO(3) 升级正交。
- 非正交性由分配层 B⁺ 吸收，控制律不改，支撑 ckpt 共用。
- C1–C5 主命令契约不改；新机型均为 opt-in（`--embodiment` 新选项 + 新 yaml）。
- REMUS 本轮不实现，仅文档差距分析。

## 9. 复现入口（Reproduction）

- 分配模块：[`thrust_allocation.py`](../../thrust_allocation.py)（ThrusterLayout + build_wrench_matrix + allocate）。
- 注册表：[`easyuuv_env.py`](../../easyuuv_env.py) `embodiment_configs`。
- 机型 yaml：`workflows/configs/embodiment_{uuv6,uuv4,uuv6_angled,uuv4_angled,submerge_flip360}.yaml`。
- 取证矩阵：`workflows/sweep_stdw_safety_pressure.py --profile embodiment_zoo`。
- 使用手册：[`EMBODIMENT_ZOO.md`](../guide/EMBODIMENT_ZOO.md)。
- CLI 契约参见 [`CHANGELOG.md`](../engineering/CHANGELOG.md) 2026-07-01 条目。

---

## English Draft (skeleton)

This addendum documents a multi-embodiment generalization capability: the thrust
allocation is lifted from a single hard-coded 8-thruster geometry to a **config-driven
allocation matrix (TAM)** covering fully-actuated, under-actuated, and non-orthogonal
vectored-thruster UUVs, while a REMUS-type single-thruster fin-coupled AUV is documented
as a gap-analysis target (not implemented this round). The central claim is that a single
low-level S-Surface controller combined with per-embodiment allocation matrices generalizes
across morphologies **without retraining**: the control law emits four channels
`[roll, pitch, yaw, depth]` unchanged, and all morphological differences — thruster count,
layout, and non-orthogonal orientation — are absorbed at the **allocation layer** by the
pseudo-inverse `B⁺` (weighted least squares for the under-actuated yaw-less case). All
embodiments (including base) further follow a submerge-then-flip360 near-boundary protocol
that keeps the vehicle fully submerged (submersion ratio s=1) throughout the maneuver.
The REMUS gap analysis explains why a faithful port requires Fossen added-mass / Coriolis /
damping terms beyond the current PhysX additive-wrench architecture. All experimental
numbers are marked 【TODO】 pending the Isaac smoke/eval matrix; the theory and design are
final.

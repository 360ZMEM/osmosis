# Paper Addendum: Lyapunov Redefinition, Directional Guard, Near-Boundary Effects, and Prior-Free Domain Adaptation

日期：2026-07-01　状态：**骨架 / 可回填**（理论已定稿，实验数值待 SO(3) 收敛后统一取证）

接续：[`PLAN_lyapunov_esuot_boundary_20260630.md`](../../.trae/documents/PLAN_lyapunov_esuot_boundary_20260630.md)、
[`PAPER_ADDENDUM_dynamic_identification_20260613.md`](PAPER_ADDENDUM_dynamic_identification_20260613.md)、
[`PAPER_ADDENDUM_followup_validation_20260613.md`](PAPER_ADDENDUM_followup_validation_20260613.md)

本附录补充论文的方法与实验章节，记录四个正交、可拆装、默认零行为变更的机制：
(M1) STDW 慢环 Lyapunov 势函数 V 的重定义；(M2) 建立在 V 之上的方向性硬约束；
(M4) 近边界（自由表面 / 地效 / 剩余浮力）Sim2Real 效应；
(M5) 无物理先验的 E-SUOT 域适应，作为 OPR 物理先验路径的对照后端。

> 撰写约定：所有以 `【TODO：…】` 标记的表格与数值为占位符，待 M6 聚焦矩阵
> （`workflows/sweep_stdw_safety_pressure.py`）取证后回填。理论与设计部分已定稿。

---

## 1. M1 — Lyapunov V 重定义（Redefining the Lyapunov Candidate）

### 1.1 Problem

STDW 慢环原始势函数为纯位姿二次型 `V = 0.5·Σ P·e²`
（[`easyuuv_stdw_wrapper.py`](../../easyuuv_stdw_wrapper.py#L196-L226) `_compute_lyapunov_mask`），
存在三处已定位缺陷：(a) 只含位姿误差、不含速率项，`dV<0` 只说明本步误差变小，
不区分"因正确控制收敛"与"因边界回正力矩偶然压小误差"；
(b) 误差用 Euler `wrap_to_pi`（标量绕 π），与 reward/metrics 已采用的 SO(3)
`quat_error_magnitude` 口径不一致，flip360 大角度下产生非单调跳变；
(c) mask 仅做样本加权，不拒绝更新（此点由 M2 修复）。

### 1.2 Method — 四种可对照的 V 定义

`--lyapunov_v_mode`（默认 `pose_quadratic` = 现状，零行为变更）：

- **V-A `pose_quadratic`（基线）**：`V = 0.5·Σ P·eᵢ²`，`e = [roll, pitch, yaw, depth]` 误差。
- **V-B `so3_consistent`（口径修正）**：姿态误差改用 SO(3)
  `e_att = quat_error_magnitude(q_goal, q_curr)`，depth 单列保留；
  `V = 0.5·(w_att·e_att² + w_depth·depth_err²)`。消除 Euler/SO(3) 口径不一致与 flip360 跳变。
- **V-C `energy_with_rate`（ISS 型）**：`V = 0.5·(eᵀP e + ėᵀQ ė)`，`ė` 由 body 角速度与
  depth 速度近似。把 V 从位置势能升级为位置+动能候选，可区分真收敛与瞬时压误差。
- **V-D `control_lyapunov`（CLF）**：在 V-C 基础上要求 `dV ≤ −α·V`（`--lyapunov_decay_alpha`），
  mask=1 当且仅当满足指数收敛率，而非仅 `dV<0`。

配套：`--lyapunov_q_diag`（速率权重 Q）、`--lyapunov_decay_alpha`（α）。

### 1.3 Experiments 【TODO：M6 `lyapunov_v_ablation` group，asym + OPR off】

| lyapunov_v_mode | off_clean final_mse | stdw_default final_mse | delta_vs_off | block_count | note |
|---|---:|---:|---:|---:|---|
| pose_quadratic | 【TODO】 | 【TODO】 | 【TODO】 | 【TODO】 | baseline |
| so3_consistent | 【TODO】 | 【TODO】 | 【TODO】 | 【TODO】 | 口径修正 |
| energy_with_rate | 【TODO】 | 【TODO】 | 【TODO】 | 【TODO】 | +动能项 |
| control_lyapunov | 【TODO】 | 【TODO】 | 【TODO】 | 【TODO】 | CLF gate |

预期问题（待数据回答）：口径修正（V-B）单独能否稳定 flip360 大角度下的 mask？
动能项（V-C）是否降低 asym 下 STDW 反作用？CLF（V-D）是否过保守导致更新过少？

---

## 2. M2 — 方向性硬约束（Directional Hard Constraint）

### 2.1 Problem

M1 的 mask 只对慢环损失 `L_src / L_tgt` 做样本加权，不拒绝更新、不约束 drift 方向
（`PLAN_stdw_hard_constraints` §3 已记录）。因此在 asymmetric + OPR off 下，即便个别样本
Lyapunov 上升，批次整体仍可能推着 target 锚朝反下降方向漂移。

### 2.2 Method

`--stdw_dir_guard`（默认 `off` = 旧纯软加权），建立在 M1 的 dV 符号 mask 之上，
把慢环更新从"降权"升级为"拒绝"：

- `pass_rate`：批次 Lyapunov 下降样本占比 < `--stdw_dir_guard_min_pass_rate`（默认 0.5）→ reject。
- `descent_align`：target 净拉力方向与 Lyapunov 下降方向的对齐分数 < `--stdw_dir_guard_align_margin`
  （默认 0.0 = 仅拒绝净反下降批次）→ reject。
- `both`：两条件同时满足才接受。

实现经 `stdw_dir_guard.DirGuardConfig` + `stdw_dir_guard.evaluate`；
诊断落盘 `stdw_dir_guard_pass_rate` / `stdw_dir_guard_align` / `stdw_update_rejected` /
`stdw_acceptance_reason`。

### 2.3 Experiments 【TODO：与 M1 联跑，三档 guard vs off】

| stdw_dir_guard | final_mse | pass_vs_off | rejected_count | note |
|---|---:|---:|---:|---|
| off | 【TODO】 | 【TODO】 | 0 | 旧软加权 |
| pass_rate | 【TODO】 | 【TODO】 | 【TODO】 | |
| descent_align | 【TODO】 | 【TODO】 | 【TODO】 | |
| both | 【TODO】 | 【TODO】 | 【TODO】 | 最严格 |

---

## 3. M4 — 近边界效应（Near-Boundary Sim2Real Effects）

### 3.1 Motivation

标称仿真的浮力为常值、近中性（`F_b = ρVg ≈ 222.4N ≈ mg`），无深度依赖、无自由表面、
无池底、无通气。真实浅水 UUV 会遭遇自由表面分段浮力/阻尼突变、推进器吸气、近地吸力与
剩余浮力。这些是 STDW 要在线适应的真实域漂移来源。M4 将其建模为可拆装的加性 body-frame
wrench（`boundary_effects.py`），默认 off，是唯一有意耦合动力学的模块。

### 3.2 Method — preset 与子效应

`--boundary_effect`（默认 `None` = off），preset 模式串或 JSON 覆盖
`BoundaryEffectModels` 字段：

- `residual_buoyancy`（B1）：`F_b ×= ρ_residual/ρ` 或加常值 `ΔB=(0.5%~2%)·mg`。
- `free_surface`（B2）：浸没比例 `s(t)=clip((z_surface−(z−H/2))/H, 0, 1)`，浮力/阻尼 ×s(t)。
- `ground_effect`（B4）：近池底叠加 `F_ground = F_nom·(D/h_dist)^γ` 吸力。
- `nonlinear_restoring`（B5）：显式 `τ=(R r_B)×F_B+(R r_G)×F_G`，可调 COG/COB 双偏置强化
  180° 失稳分叉。
- `full`：以上组合。

注入时机：`ctrl_mismatch` 之后、首次 reset 之前，经 `env.apply_boundary_effect`。
`compute_boundary_wrench` 返回 info 字典（键 `submersion_ratio` / `residual_dB` / `ground_mag`），
在 `_compute_dynamics` 缓存到 `env._last_boundary_info`。

诊断落盘（off 时全 NaN）：`boundary_submersion_ratio` / `boundary_residual_dB` /
`boundary_ground_mag`。可视化 `stdw_integration/plots.plot_boundary_wrench`。

> 已知局限：通气（ventilation）因子在 env 侧已应用但暂未存入 `_last_boundary_info`，
> 故绘图列 `boundary_vent_min` 当前恒不命中，保留占位待 env 侧补充。

### 3.3 Experiments 【TODO：M6 `boundary_effect_pressure` group】

| boundary_effect | off_clean final_mse | stdw_default final_mse | delta_vs_off | note |
|---|---:|---:|---:|---|
| off | 【TODO】 | 【TODO】 | — | baseline（无边界力） |
| residual_buoyancy | 【TODO】 | 【TODO】 | 【TODO】 | |
| free_surface | 【TODO】 | 【TODO】 | 【TODO】 | |
| full | 【TODO】 | 【TODO】 | 【TODO】 | |

---

## 4. M5 — 无物理先验的 E-SUOT 域适应（Prior-Free Domain Adaptation）

### 4.1 Motivation

OPR 路径（drift-router + micro-probe + pseudo-action/J_inv）依赖 COM-COB 物理先验读取
`_base_com_to_cob_offsets`、`_read_jacobian_inv_diag`。用户核心目的是验证"无需显式物理估计
即可达到无偏域适应"。M5 用熵正则半对偶最优传输（E-SUOT，按 [`ref/ICML.md`](../../ref/ICML.md)
公式自研，不侵入论文私有结构），纯从状态-动作分布计算 target 锚。

### 4.2 Method — 双后端

`--domain_adapt_backend`（默认 `opr` = 现状，零行为变更），与 OPR 单选互斥：

- **ES-A `esuot_full`**：忠于 Algorithm 1 的 dual potential `w_φ` + 传输映射 `T_θ` +
  熵正则 ε + f-散度非平衡惩罚（Eq.18 λ1/λ2，呼应推进器推力不守恒→unbalanced）。
- **ES-B `esuot_light`**：去双网络，用熵正则 Sinkhorn / barycentric 直接把 source 控制样本
  运输到 target 支撑域，不估计任何物理参数。
- **ES-C `none`**：关闭域自适应，下界对照。

配套：`--esuot_eps`（ε）、`--esuot_eta`（η，传输代价 `1/(2η)‖·‖²`）、
`--esuot_lambda1/2`（Eq.18 非平衡权重）、`--esuot_divergence`（f-散度，KL=Table 3 最优）、
`--esuot_inner_iters` / `--esuot_num_steps`（ES-A）、`--esuot_sinkhorn_iters`（ES-B）。

**无先验性保证**：esuot_* 路径不读 `com_to_cob` / `J_inv` / micro-probe，
target 锚完全由分布传输得到。经 `esuot.DomainAdaptAdapter`。

**不侵入声明**：所有公式按 ref 文档自研，命名用通用 OT 术语
（dual_potential / transport_map / entropic_semidual），不复制论文私有代码或专有名。
双轨保留：OPR 不删除，与 E-SUOT 并存、单选切换。

### 4.3 Experiments 【TODO：M6 `esuot_vs_opr` group，asym + boundary on】

| domain_adapt_backend | final_mse | delta_vs_opr | reads_physical_prior | note |
|---|---:|---:|:---:|---|
| opr | 【TODO】 | — | yes | 物理先验基线 |
| esuot_full | 【TODO】 | 【TODO】 | no | Algorithm 1 |
| esuot_light | 【TODO】 | 【TODO】 | no | Sinkhorn barycentric |
| none | 【TODO】 | 【TODO】 | no | 下界对照 |

核心待答问题：无先验 esuot_* 能否达到甚至超过 OPR 的 final_mse？
轻量 ES-B 相对完整 ES-A 的性能差距是否可接受（验证"无显式估计即可无偏"）？

---

## 5. 正交性与契约（Orthogonality & Contract）

- M1/M2/M4/M5 互相正交，默认参数 = 旧行为
  （`pose_quadratic` / `off` / `None` / `opr`）。
- M4 是唯一有意耦合（改动力学=新目标域），通过默认 off 完全旁路。
- M5 与 OPR 单选互斥（`domain_adapt_backend ∈ {opr, esuot_full, esuot_light, none}`）。
- C1–C5 主命令契约不改；四机制均为 optional plug-and-play flag。

## 6. 复现入口（Reproduction）

- 取证矩阵：`workflows/sweep_stdw_safety_pressure.py`（M6 聚焦 group）。
- 可视化：`stdw_integration/plots.py`（`plot_lyapunov_guard` / `plot_boundary_wrench`）、
  `workflows/tools/plot_safety_pressure_matrix.py`（跨 run 对照）。
- CLI 契约参见 [`CHANGELOG.md`](../engineering/CHANGELOG.md) 2026-07-01 条目。

---

## English Draft (skeleton)

This addendum documents four orthogonal, plug-and-play mechanisms added to the STDW slow
loop, all defaulting to the legacy behaviour (zero default-behaviour change):
(M1) a redefinition of the Lyapunov candidate `V` with four selectable modes
(pose-quadratic baseline, SO(3)-consistent, energy-with-rate, and control-Lyapunov);
(M2) a directional hard constraint built on top of the descent mask that *rejects* rather
than merely down-weights slow-loop updates lacking Lyapunov-descent evidence;
(M4) near-boundary Sim2Real effects (residual buoyancy, free surface, ground effect,
nonlinear restoring) modelled as additive body-frame wrenches;
(M5) a prior-free E-SUOT domain-adaptation backend that computes the target anchor purely
from the state-action distribution, contrasted against the physics-prior OPR path.
All experimental numbers are marked 【TODO】 pending the M6 focused matrix and are to be
back-filled once the SO(3) manifold upgrade converges. The theory and design are final.

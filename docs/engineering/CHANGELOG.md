# Changelog

本仓库不使用 git，但仍按时间倒序记录实质变更。日期 ISO 8601 (UTC+8)。

---

## 2026-07-01 — 多 Embodiment + 配置化推力分配 (TAM) + 下沉后 flip360 近边界（基础设施）

**背景**
- 需要跨形态泛化能力：从"仅质量/惯量变体"扩展到**推进器数量/布局/朝向都不同**的机型族
  （全驱 6 推、欠驱动 4 推、非正交变种），并把 REMUS 型单推 + 舵面 AUV 作差距分析。
- 推力分配此前硬编码在三处（`_pid_control` 混合块、`get_thruster_com_and_orientations`
  几何、散布的 `8` 字面量），无法承载非正交/任意 N 推进器。核心设计：控制律不变，
  非正交与形态差异全部由**分配层**的伪逆 B⁺ 吸收——最小侵入、最大复用现有 ckpt。

**两处拍板**
- **REMUS 本轮 skip 实施**，仅文档记录设计 + 差距分析（∝U²·δ 零速不可转向，与矢量推进器
  本质不同；忠实移植需 Fossen 附加质量/科氏/阻尼，PhysX 刚体积分不足）。
- **TAM 采用 config 驱动 B⁺，base/asym 保持旧硬编码路径**（保 A4 零行为变更与 model_2398 复现）。

**新增（文档，本轮先落地）**
- `docs/guide/EMBODIMENT_ZOO.md`：面向用户的多机型手册（9 节：机型总览/物理参数设计表/
  推进器排布/下沉后 flip360 协议/差异化任务/REMUS 文档态/ckpt 适用矩阵/命令示例/
  M1–M6 默认启用速查表）。INDEX guide 区补第 8 行。
- `docs/principles/PAPER_ADDENDUM_multi_embodiment_20260701.md`：面向论文的附录骨架（9 节
  + 英文草稿，实验数值全 `【TODO】` 占位，商业 AUV 引用点标 `【有东西说：cite …】`）。
  INDEX principles 区补第 8 行。

**新增（代码/config，opt-in，默认零行为变更）**
- `thrust_allocation.py`：`ThrusterLayout` + `build_wrench_matrix(layout)->B(6,N)` +
  `allocate(B, wrench_cmd, mode)`（pinv / wls）。body 6-DOF 顺序
  `[Fx,Fy,Fz,Tx,Ty,Tz]`，控制通道 `[roll,pitch,yaw,depth]`→`[Tx,Ty,Tz,Fz]`。
- `easyuuv_env.py`：`embodiment_configs` 新增 `uuv6/uuv4/uuv6_angled/uuv4_angled`（可选键
  `thrust_allocation`/`volume`/`action_lim_vec`/`num_thrusters`）；`apply_embodiment_config`
  增 config 分配分支（构建 `self._alloc_B`/`_alloc_mode`/`_alloc_weight`，
  `_num_thrusters` 传入 `DynamicsFirstOrder`）；分配路径分叉 `self._use_config_alloc`
  （默认 False→base/asym 走旧混合块）；`8` 字面量改读 `self._num_thrusters`（默认 8）；
  推进器效率/失配 buffer 随 N 重建，避免 4/6 推运行期形状错配。
- `easyuuv_env.py`：下沉-then-flip360 两相位机（`submerge_phase_enable`/`submerge_depth`/
  `submerge_hold_steps`，默认关）+ 水面破面守卫（`surface_guard_enable`/`surface_margin`，
  默认关）。安全关系式 `submerge_depth < z_surface_guard − vehicle_height/2 − surface_margin`（默认 <2.7）。
- `workflows/configs/embodiment_{uuv6,uuv4,uuv6_angled,uuv4_angled,submerge_flip360}.yaml`：
  差异化 reference（欠驱动 yaw 幅置 0）+ submerge/surface_guard + wave block。
- `workflows/play_stdw_adapt.py`：`--embodiment` choices 追加四新机型。
- `workflows/sweep_stdw_safety_pressure.py`：新增 profile `embodiment_zoo`。

**验证（本轮非 Isaac）**
- `python3 -m py_compile thrust_allocation.py easyuuv_env.py workflows/play_stdw_adapt.py workflows/sweep_stdw_safety_pressure.py` 全过。
- `python3 workflows/tools/test_thrust_allocation.py` 全过：8 推几何 `B` 与运行期 `quat_apply + cross`
  一致；config B⁺ 通道解耦；旧硬编码混合块净力矩符号与 `geo_channel_sign=[-1,+1,-1]`
  一致；uuv4 WLS 屏蔽 yaw。
- `python3 workflows/sweep_stdw_safety_pressure.py --profile embodiment_zoo --dry_run --limit 3 --total_steps 10`
  冒烟通过；5 个 `embodiment_*.yaml` 均通过 `yaml.safe_load` 静态解析。

**契约**
- C1–C5 主命令契约不改。新机型 + 下沉相位 + 水面守卫均为 opt-in，默认旧 baseline 行为。

---

## 2026-07-01 — SO(3) 流形 S-Surface 升级 Phase 1（力矩解封 + 几何姿态误差，默认零行为变更）

**背景**
- Flip360（满 ±π 后空翻）此前经 F1–F4 判定触及架构边界，两个真凶：
  (1) 力矩预算被算法阉割——固定 `action_lim=0.15` + S 面 sigmoid 截断把 pitch 可用力矩
      卡在 10.43Nm < asym 最坏浮力回正 15.89Nm；(2) 欧拉奇异 + 三通道解耦在大机动下破产。
- 方案：把 S-Surface 从 1D 逐轴解耦升级到 SO(3) 流形 3D 向量化（保留 sigmoid 内核）。

**新增（`easyuuv_env.py`，全部 cfg 开关，默认 = 旧 euler 行为）**
- `EasyUUVEnvCfg` 新增字段：`attitude_error_mode`（默认 `"euler"`）、
  `action_lim_vec`（默认 `[0.15, 0.15, 0.3, 1.0]` = 旧硬编码）、`geo_zeta1/geo_zeta2`
  （1.0/0.5）、`geo_residual_scale`（0.0）、`geo_channel_sign`（`[-1.0, 1.0, -1.0]`）。
- `_so3_attitude_error()`：body-frame 姿态误差 `e_R = 2·sgn(w)·vec(q_curr⁻¹⊗q_goal)`；
  角速度误差 `e_omega = ω_desired − ω_body`，`ω_desired` 由相邻两步 goal 四元数有限差分
  得到（首步 `_goal_prev==goal` → 0），规避欧拉率在 ±π 附近奇异。
- `_pid_control` so3 分支：`PID_value[0:3] = (2/(1+exp(-s_ratio·(ζ1·e_R+ζ2·e_omega
  +res·a)))−1)·geo_channel_sign`；depth 通道（3）保持旧逐轴逻辑。大姿态误差处
  `s_ratio·ζ1·e_R` 自动饱和 sigmoid → 满 PWM 权威，解除 10.43Nm 软封印。
- `_goal_prev` 缓冲 + reset/`_update_reference` 快照（供有限差分求 ω_d）。
- `action_lim` 改为读 `cfg.action_lim_vec`；`_geo_channel_sign` 在 init 缓存。

**关键修正（数值标定分配矩阵符号）**
- `/tmp/alloc_sign_check.py` 用真实推进器几何算出单位命令的净 body 力矩：
  roll τx=−0.84、pitch τy=+0.516、yaw τz=−0.56。固定手工分配矩阵对 roll/yaw 通道**反号**。
- 故 SO(3) S 面输出须乘 `geo_channel_sign=[-1,+1,-1]` 才是 restoring；否则纯解析基线
  在 roll/yaw 上反阻尼（发散），会毁掉从 model_2846 的 fine-tune。诊断报告原式
  `e_ω=ω_body−ω_desired` 配 + 号亦为反阻尼，已改为 `ω_desired−ω_body`。

**新增配置**
- `workflows/configs/train_flip360_so3_p1.yaml`：`attitude_error_mode: so3`、
  `action_lim_vec: [0.6, 0.6, 0.3, 1.0]`、`geo_zeta1/2: 1.0/0.5`、`geo_residual_scale: 0.5`，
  镜像 F4 的 `mixed_sine_flip360` 参考 + JONSWAP 扰动，从 model_2846 fine-tune 150 iter。

**验证**
- `python3 -m py_compile easyuuv_env.py` 通过。
- `/tmp/so3_math_verify.py`（standalone，复刻精确张量运算）：
  (1) 200 个随机 goal 上解析基线全部 restoring（τ·e_R>0）；
  (2) 错误符号 `[1,1,1]` → 152/200 反阻尼，证明符号向量必要；
  (3) ω_d 有限差分正确（0.5 rad/s goal 自旋 → e_omega=[0,0,0.5]）；
  (4) 首步 ω_d≈0。
- 零行为变更：默认 `attitude_error_mode="euler"` 时 so3 分支不执行，
  `_so3_attitude_error`/`_geo_channel_sign`/`_goal_prev` 均不被消费，
  `action_lim_vec` 默认值 = 旧硬编码，故 model_2846 复现不受影响。

**训练命令（待运行，Phase 1 验证用）**
```bash
bash custom_workflows/run_with_isaac_env.sh workflows/train_meta.py \
  --task EasyUUV-Direct-Parametric-v1 --num_envs 512 \
  --meta_stage 0 --resume True --resume_load_optimizer False \
  --load_run 2026-06-29_23-50-50_flip360_curric_b_full_pi_stage0 \
  --checkpoint model_2846.pt \
  --workflow_config workflows/configs/train_flip360_so3_p1.yaml \
  --headless
```
验收：flip360 base/asym 低于 model_2846（2.0701/3.6606）且 ordinary base/asym 不退化。

**训练与双目标 eval 结果（Phase 1，未胜出）**
- 训练已完成：run `2026-07-01_20-15-47_flip360_so3_p1_stage0`，150 iter（2846→2995），
  exit 0，checkpoints model_2850/2900/2950/2995。`log_mse` 全程落在 F3/F4 同构的 12–40
  混沌非收敛区间（起点 iter 2846=6.46 反而最优，此后再未回落）。
- 修复：运行首次崩溃 `AttributeError: 'EasyUUVEnv' object has no attribute
  '_geo_channel_sign'`——init 块（`self._geo_channel_sign`）又一次未落盘（编辑器缓存 vs 磁盘
  分叉），重新按 `self.action_lim` 锚点插入 easyuuv_env.py line 388-392，grep 裸磁盘 +
  py_compile 验证后重跑。
- 新增 SO(3)-mode eval 配置（控制律参数必须与训练一致，否则 policy 残差与控制律不匹配）：
  `workflows/configs/pressure_flip360_medium_so3.yaml`、`matrix_wave_medium_so3.yaml`；
  驱动脚本 `workflows/run_so3_p1_eval.sh`（沿用 F3/F4 off_clean 协议：total_steps=1500,
  use_stdw=False, seed=0）。
- 四 checkpoint × 双目标 eval（final_mse，baseline model_2846 括注）：

  | ckpt (train log_mse) | flip360 base | flip360 asym | ordinary base | ordinary asym |
  |---|---:|---:|---:|---:|
  | model_2846 (baseline) | 2.0701 | 3.6606 | 0.2229 | 0.2266 |
  | model_2850 (15.16) | 3.9158 | 4.7203 | 0.4815 | 0.5739 |
  | model_2900 (23.15) | 6.5130 | 6.7913 | 0.4986 | 0.5559 |
  | model_2950 (17.11) | 16.2042 | 11.8111 | 0.4983 | 0.5626 |
  | model_2995 (30.51) | 5.9106 | 3.0318 | 0.4664 | 0.5523 |

- **结论：Phase 1 未胜出，保留 model_2846 为最佳。** 每个 checkpoint 的 flip360 base 均劣于
  2.07、两项 ordinary 均退化到 ~0.5（约 2×），仅 model_2995 的 flip360 asym（3.03<3.66）单点
  改善，不构成占优。ordinary 退化说明 so3 控制律路径未能干净地涵盖旧 euler 行为。训练 log_mse
  落入 F3/F4 混沌区间的警示与 eval 结论一致。
- 下一步待用户拍板（用户既定优先级）：调 action_lim / geo_zeta / residual_scale 重训；或
  扩 18 维；或开 Phase 2（解析前馈 τ_ff + 伪逆分配器 B†）。

**契约**
- C1–C5 主命令契约不改。Phase 2（解析前馈 τ_ff + 伪逆分配器 B†）为可选，未实施。

---

## 2026-06-30 — Flip360 F4 难区间容差整形 + 物理力矩边界判定（未胜出，确认架构边界）

**新增**
- `easyuuv_env.py`：F4 难区间（接近倒置）姿态跟踪容差整形。新增 cfg 字段
  `flip_tol_relax`（默认 0.0，零行为变更）、`flip_tol_band_lo`（120°）、`flip_tol_band_hi`（180°）；
  `_get_rewards` 内当 goal 倾角进入难区间时用 smoothstep 放宽 `rew_ang` 高斯宽度
  （÷(1+relax·smooth)），只影响训练 reward，不影响 eval MSE 指标。
- `workflows/configs/train_flip360_f4_tolshape.yaml`：从 `model_2846` fine-tune，
  `flip_tol_relax: 2.0`，150 iter，lr 8e-5。
- `workflows/analyze_flip360_torque_budget.py`：物理力矩边界定量分析工具（可复现），
  对比逐轴控制力矩预算 vs 倒置区浮力回正力矩，回答“是否物理无解”。

**实验结果（双目标 eval, total_steps=1500, use_stdw=False, seed0, freq0.05）**
- F4 model_2850（训练 MSE 最低 ckpt）：flip360 base 2.0520 / asym 3.6747；
  ordinary base 0.2229 / asym 0.2268。
- 对比 `model_2846`：flip360 base 2.0701 / asym 3.6606；ordinary base 0.2229 / asym 0.2266。
- **四项全面持平，无胜出无遗忘。** 训练 `log_mse` 段初从 ~6.1 立即脱离、全程在 14–42
  混沌区间震荡、再未回到 6.1，与 F3 完全同构 —— 倒置区训练不收敛对 reward 容差整形不敏感。

**物理力矩边界判定（`analyze_flip360_torque_budget.py`）**
- F_buoy ≈ 222.49 N；倒置区浮力回正力矩最坏值：base offset(z=0.01) ≈ 2.22 Nm，
  asymmetric offset(xy=0.05) ≈ 15.89 Nm。
- 执行器硬上限（PWM=1.0）：roll ≈ 114.55 Nm、pitch ≈ 70.37 Nm —— 超 asym 最坏回正约 4–7×，
  故满 ±π 后空翻**非严格物理无解**。
- 但名义工作点（S 面 action_lim=0.15, a≈1）：roll ≈ 16.97 Nm、pitch ≈ 10.43 Nm；
  其中 **pitch 10.43 Nm < asym 最坏回正 15.89 Nm**，克服回正须把动作压进 sigmoid 饱和尾部（≈bang-bang）。
- **主导边界 = 倒置区不稳定平衡 + 欧拉轴控制奇异 + 名义工作点可学习带宽不足**，
  三者不可由 reward 整形 / 课程移除。`model_2846` 已逼近该架构上限。

**文档**
- `REPORT_flip360_training_20260628.md` 新增第 6 节「F4 难区间容差整形 + 物理边界判定」。
- `PLAN_stdw_hard_constraints_20260628.md` Section 12 Step F4 记录执行/结果/物理边界判定与执行顺序更新。
- `.results/flip360_freq_sweep_analysis.md` 末尾追加「物理力矩边界判定」节。

**契约**
- C1-C5 主命令契约不改。

---

## 2026-06-30 — Flip360 F3 平滑幅度爬坡课程（未胜出，保留 model_2846）

**新增**
- F3 三段课程配置：`train_flip360_f3_s1_halfpi.yaml`（±π/2, flip_prob0.3, lr1e-4）、
  `train_flip360_f3_s2_3qpi.yaml`（±3π/4, 0.4, 9e-5）、`train_flip360_f3_s3_fullpi.yaml`（±π, 0.5, 8e-5）。
- 三段从 A3 `model_2398` 顺序 fine-tune（`--meta_stage 0 --resume_load_optimizer False`），
  run 目录 `2026-06-30_22-11-54/22-14-39/22-19-46_flip360_f3_s{1,2,3}_*_stage0`。

**实验结果（双目标 eval, total_steps1500, use_stdw=False, seed0, freq0.05）**
- F3 model_2755（S3 末段）：flip360 base 3.2019 / asym 4.0127；ordinary base 0.2246 / asym 0.2755。
- F3 model_2650（S3 训练 MSE 最低 23.27）：flip360 base 3.1214 / asym 3.5536。
- 对比 curric `model_2846`：flip360 base 2.0701 / asym 3.6606。
- **结论：F3 未超过 model_2846，保留 2846 为最佳。** 根因：S3 进入满 ±π 后训练 `log_mse`
  从 ~6 爆到 20–85 混沌区间且全程不收敛，瓶颈在满倒置区训练不收敛（与 F1/F2 角度极限一致），
  而非课程跳变幅度。model_2755 另有 ordinary asym 轻微遗忘。

**文档**
- `REPORT_flip360_training_20260628.md` 新增第 5 节「F3 结果」。
- `PLAN_stdw_hard_constraints_20260628.md` Section 12 记录 F3 执行/结果与执行顺序更新；
  下一步建议转向 F4（不放慢 + 难区间容差整形），等待用户拍板。

**契约**
- C1-C5 主命令契约不改。

---

## 2026-06-30 — Flip360 物理极限取证（F1 keep + F2 速度扫描）与报告中文化

**新增**
- `workflows/analyze_flip360_limits.py`：读取逐步 `stdw_output_*.csv`，按 |参考角| 与
  |参考角速度| 分箱输出姿态误差、误差–角速度相关、`control_effort` 近饱和占比，并给出
  `RATE-LIMITED` / `ANGLE-LIMITED` 自动判据。
- keep / 速度扫描配置：`pressure_flip360_keep_slow.yaml`（freq 0.005）、
  `pressure_flip360_freq0p025.yaml`、`pressure_flip360_freq0p1.yaml`。

**实验结果（model_2846，flip360，只改 ref_sine_freq）**
- RMSE 随参考变快单调下降：keep(0.005) base 1.742 → 0.025 1.001 → 0.05 0.915 → 0.1 0.792。
- 准静态 keep 下 base 推力近饱和步占比 15.2%（其余频率 <0.3%）。
- 判定为**驻留时间惩罚型角度极限**（非 rate-limited）：难区间（接近倒置）浮力回正力矩持续反向，
  驻留越久越饱和。**“放慢参考”假设被数据否定**。
- 分析产物：`.results/flip360_keep_analysis.md`、`.results/flip360_freq_sweep_analysis.md`。

**文档**
- `REPORT_flip360_training_20260628.md` 全文中文化，新增「课程现状 / 更平滑训练流程 /
  物理极限分析（含脚本）/ 下一步建议」，并按 F1/F2 结论更新第 4 节。
- `PLAN_stdw_hard_constraints_20260628.md` Section 12 记录 F1/F2 取证结果与执行顺序更新。

**契约**
- C1-C5 主命令契约不改。

---

## 2026-06-29 — Flip360 连续参考修复与幅度课程调优

**修复**
- `easyuuv_env._update_reference`：`flip360_sine` 此前只在 reset 设一次目标、控制步内不推进，
  实为“静态大姿态保持”。现每步推进 roll/pitch 参考；Round 1 `model_2550` 旧数据作废。

**新增**
- `easyuuv_env.py`：`reference_mode=mixed_sine_flip360`（按 `ref_mix_flip_prob` 抽 flip / ordinary sine）
  及 `ref_mix_*` cfg 字段。
- `workflows/train_meta.py`：`--meta_stage 0` fine-tune 模式（不做梯度隔离 / 不覆盖 env）、
  `--resume_load_optimizer`、`--ref_mix_*` 透传。
- `workflows/sweep_stdw_safety_pressure.py`：窄 profile `flip360_only` / `ordinary_only`。
- 课程配置：`train_flip360_curric_a.yaml`（±π/2）、`train_flip360_curric_b.yaml`（±π）、
  `train_flip360_mixed_replay.yaml`。

**实验结果**
- 幅度课程（A ±π/2 → B ±π，均混 40% ordinary）产出 `model_2846`：连续 flip360 base
  `2.8675 -> 2.0701`、asym `5.5360 -> 3.6606`，且 ordinary medium 无遗忘
  （base `0.2257 -> 0.2229`，asym `0.2263 -> 0.2266`）。首个 Pareto 占优 A3 的 checkpoint。

---

## 2026-06-28 — STDW 安全压测入口与 360 度参考轨迹

**新增**
- `easyuuv_stdw_wrapper.py` 新增 Lyapunov gate mode：`sample_mask`、`strict_sample_mask`、
  `guarded_drift`，并输出 `stdw_dV`、滚动 pass rate、safety block 等诊断字段。
- `workflows/play_stdw_adapt.py` 新增 `--lyapunov_gate_mode`、margin/window/pass-rate、
  `--lyapunov_guard_action`，可在安全压测中跳过慢环、冻结 drift 或将 drift 退回 0。
- `easyuuv_env.py` / `workflows/train_meta.py` 新增 `reference_mode=flip360_sine`，用于
  roll/pitch ±π 连续后空翻评估与后续训练入口。
- `workflows/configs/pressure_flip360_medium_full.yaml` 与
  `workflows/sweep_stdw_safety_pressure.py`：小而硬专项矩阵，覆盖 asymmetric Lyapunov
  门控、1% 控制器 mismatch、360 度 eval，并强制关闭 OPR/router/probe。

**契约**
- C1-C5 主命令契约不改；`COMMAND_CONTRACT.md` 只新增 optional safety pressure tests。

**实验结果**
- 已完成 32-cell `small_hard` 正式矩阵：
  `.results/stdw_safety_pressure_20260628_221237/`。
- 结论见 `docs/engineering/REPORT_stdw_safety_pressure_20260628.md`：默认 STDW 在
  `asymmetric + OPR off` 下比 clean off 差约 +139% 到 +141%；`strict_sample_mask`
  不修复；`guarded_drift + zero_drift` 可回到 clean-off 水平。360 度后空翻 eval
  当前失败，应进入训练课程设计。
- 新增专项修复上下文：`docs/engineering/PLAN_stdw_hard_constraints_20260628.md`，
  固定硬约束设计空间与后续 step-by-step 路线：先修 `pass_vs_off` 容差，再做
  batch-level update acceptance / rollback，最后再处理 pseudo-action 通道门控与 360 训练课程。
- OPR/domain-transfer 线索已记录并保留；当前切换到 flip360 训练课程。第一轮使用
  `flip360_ft200_from_a3_stage1` 短训验证链路，再用 flip360 eval 对照 A3 stage2 baseline。
- 新增 `docs/engineering/REPORT_flip360_training_20260628.md`。`model_2550.pt` 是本轮
  最佳 flip360 checkpoint：base `3.366683 -> 2.237671`，asymmetric `5.895537 -> 2.184740`；
  但普通 medium/asymmetric 从 `0.226305` 劣化到 `0.286082`，需要 mixed curriculum /
  two-objective early stopping。

---

## 2026-06-28 — 命令契约与接手记忆收敛到 A3 stage2

**文档收敛**
- `docs/guide/COMMAND_CONTRACT.md` 更新为当前可执行契约：Isaac 相关命令统一使用
  `bash custom_workflows/run_with_isaac_env.sh <script.py> ...`，并以 A3 stage2
  `model_2398.pt` 作为主线复现 ckpt。
- `docs/guide/AGENT_HANDOFF.md` 的接手记忆从 2026-06-07 试运行 ckpt 更新到
  2026-06-08 A3 stage1/stage2 主线 ckpt，并修正 guide/principles 文档路径。
- `README.md` 的命令摘要同步为当前 launcher 与 `model_2398.pt` 路径，详细命令仍以
  `COMMAND_CONTRACT.md` 为准。
- `docs/principles/RESEARCH_ANALYSIS.md` 的复现命令从旧 `--policy/--output_dir/--run_dir`
  口径更新为当前 `--policy_path/--out_root/--matrix_dir`。

**核对来源**
- 对齐 `docs/engineering/REPORT_full_matrix_20260608_a3_stage2.md` 与
  `docs/engineering/控制器稳定调节记录.md` §7.6-7.7。

---

## 2026-06-24 — 干净环境 Fossen 6-DOF 部署链路冒烟

**新增**
- 在被 `.gitignore` 忽略的 `.results/clean_env_fossen_demo/` 下落地外部样例：
  `fossen6dof_wrapper.py`、`make_dummy_policy.py`、`run_fossen_chain.py`、
  `run_clean_smoke.sh`、`requirements-clean.txt`。
- 新增小白向教程
  [`docs/guide/CLEAN_ENV_STDW_FOSSEN_DEMO.md`](../guide/CLEAN_ENV_STDW_FOSSEN_DEMO.md)，
  说明从干净 Python venv 准备依赖、生成 dummy 12D->8D TorchScript 策略、运行
  Fossen 6-DOF 虚拟实物、产出 replay CSV 并调用 `eval.deploy_eval` 的完整命令链。
- `docs/INDEX.md` 的用户指南区加入该教程入口。

**修复**
- `__init__.py` 的顶层 Gym 注册导入改为同时容忍缺少 `gymnasium` 和 `omni.*`。
  否则在非 IsaacLab 干净环境中执行 `from easyuuv_stdw.eval import ...` 会先触发
  顶层包初始化并因缺少 `gymnasium` 失败，破坏 `eval/` 的 Isaac-independent 部署承诺。

**验证**
- `.results/clean_env_fossen_demo/run_clean_smoke.sh` 已通过：系统 Python 缺少
  `python3.12-venv` 时 fallback 到不含 Isaac 路径的 conda base Python，创建干净 venv，
  安装 `numpy`、`pyyaml`、CPU-only `torch-2.3.1+cpu`。
- 验证 venv 中 `omni_spec=None`，依赖来自 `.venv-clean/lib/python3.11/site-packages`。
- 链路输出：`obs_dim=12`、`action_dim=8`、`backend=torchscript`，
  `fossen_replay.csv`、`deploy_eval_out.csv`、`chain_summary.json` 均写出。

---

## 2026-06-07 — 第二期改良（TAG + 分阶段训练）+ 8D 阶段一重训 2500 iter

**新功能（已 py_compile + 功能冒烟通过）**
- 模块一 TAG（Triggered Adaptation Gating）：`play_stdw_adapt.py` 新增
  `--enable_trigger_gate`(默认 True) / `--trigger_threshold`(默认 0.05 rad)；
  仅当滤波复合误差 `filt_err >= threshold` 才激活慢环梯度更新，否则静默并计数。
  新增 CSV 列 `trigger_gate_silenced`、终端日志 `[STDW-Slow] Triggered: ...`、
  summary.json 字段 `enable_trigger_gate/trigger_threshold/gate_silenced_count`。
- 模块二 分阶段元控制训练：`train_meta.py` 新增 `--meta_stage`(1/2,默认1)、
  `--stage1_checkpoint`、`--stage2_cob_offset_xyz`(默认 0.03,0.03,0.01)、
  `--stage2_wave_mode`(默认 jonswap)。阶段一只训控制头（`identity_init=True` +
  增益头输出行[4:8]梯度清零）；阶段二加载阶段一权重、冻结控制头(行[0:4])+主干
  `requires_grad=False`、重建 Adam 只含可训练参数、开启中强度 COM-COB 漂移 + JONSWAP。
- 绘图增强：`plots.py` 在 `stdw_losses.png` 用灰色阴影标示 TAG 静默期；新增第 7 张图
  `stdw_tracking_overlay.png`（roll/pitch/yaw/depth 的 Target vs Actual + per-channel MSE）。
- CSV 契约：每个评估 cell 落盘原始跟踪 CSV 到 artifact_dir（`stdw_output_*.csv`）；
  `sweep_full_matrix.py` 透传 `--enable_trigger_gate/--trigger_threshold`，
  矩阵 CSV 新增 `gate_silenced_count` 列。

**验证**
- py_compile：`play_stdw_adapt.py` / `train_meta.py` / `sweep_full_matrix.py` / `plots.py` 全过。
- plots 功能冒烟（合成 DataFrame）：新/旧 CSV 均生成 7 张图，灰色阴影与"旧 CSV 无
  trigger_gate_silenced 列"回归均无报错。
- TAG 运行期冒烟受阻：`play_stdw_adapt.py` 进入步进循环前强制加载 checkpoint，
  当时仓库无 8D 模型 → 因此先做阶段一重训产出 ckpt。

**8D 阶段一重训（C1, meta_stage=1）**
- 命令：`train_meta.py --task EasyUUV-Direct-Parametric-v1 --num_envs 512
  --max_iterations 2500 --meta_stage 1 --logger tensorboard`。
- 用时 ≈ 11 min（15:08→15:20），≈45k steps/s，0.27 s/iter（RTX 4060 Laptop 8GB）。
- 阶段一隔离已确认生效：日志含 `[Stage1] force env_cfg.identity_init = True`
  + `[Stage1] freeze actor output rows [4:8] (gain head)`。
- 产物：`logs/rsl_rl/easyuuv_parametric/2026-06-07_15-08-41_stage1/model_2499.pt`
  （+ 每 50 iter checkpoint、compact_log.jsonl、mse_curve.jsonl、tfevents）。
- 收尾指标：reward `log_mse` ≈ 30–33（越大越好的 reward 项，非真实误差）、
  vloss ≈ 0.20、ploss ≈ -0.0017、`noise_std` ≈ 2.96（adaptive schedule 推高，偏大，
  说明策略仍在较高探索噪声；后续可关注是否需降 init/调 schedule）。

**8D 阶段二训练（C1, meta_stage=2, 试运行 800 iter）**
- 命令：`train_meta.py --task EasyUUV-Direct-Parametric-v1 --num_envs 512
  --max_iterations 800 --meta_stage 2 --stage1_checkpoint
  logs/rsl_rl/easyuuv_parametric/2026-06-07_15-08-41_stage1/model_2499.pt --logger tensorboard`。
- 迭代计数从 2499 续到 3298（800 iter）。
- 阶段二隔离已确认生效：`[Stage2] com_to_cob_offset_xyz_range = [0.03, 0.03, 0.01]`
  + `disturbance_cfg.mode = jonswap` + `loading stage-1 checkpoint`
  + `freeze actor backbone + output rows [0:4] (ctrl head)`
  + `rebuilt Adam over 9 trainable tensors (18961 params)`。
- 产物：`logs/rsl_rl/easyuuv_parametric/2026-06-07_15-23-41_stage2/model_3298.pt`。

**TAG 运行期验证（用阶段二 ckpt，轻量短跑）**
- 命令：`play_stdw_adapt.py ... --checkpoint model_3298.pt
  --load_run 2026-06-07_15-23-41_stage2 --slow_loop_interval 60 --batch_size 64`（400 步）。
- 触发逻辑完美：step 120/180/240/300/360 全部 `Triggered: True ... gate=on`，
  filt_err ≈ 5–7（远超 thr=0.05），慢环 loss 正常。
- 产物核对通过：
  - CSV `stdw_output.csv` 含 `trigger_gate_silenced` 列（第 20 列）。
  - summary.json 含 `enable_trigger_gate=true / trigger_threshold=0.05 /
    gate_silenced_count=0 / slow_loop_triggers=5`。
  - plots 生成全部 7 张图（含 `stdw_tracking_overlay.png`）。
  - artifact 镜像存在：`artifacts/.../2026-06-07_15-23-41_stage2/model_3298/stdw_new/`
    下 7 png + `stdw_output_*.csv` + `summary_*.json`。

**修复（本次）**
- `play_stdw_adapt.py` 行 606：`ppo_runner.load(resume_path)` → 
  `ppo_runner.load(resume_path, load_optimizer=False)`。根因：阶段二 ckpt 的 optimizer
  仅含 9 个可训练 tensor（重建 Adam），与 play 默认 `load_optimizer=True` 的全参数
  optimizer param group 大小不匹配；而 play 下游本就自建 optimizer（行 612），
  无需 ckpt optimizer state。已查 rsl_rl 源码确认 `load(path, load_optimizer=True)` 支持该参数。

**待办**
- 48-cell C4 全矩阵复评。
- 阶段二仅试运行 800 iter，如需正式产物可加大 `--max_iterations`。

---

## 2026-06-07 — 迁移阻塞项修复 + C3/C4/C5 冒烟通过

**修复（迁移遗漏）**
- 恢复 `custom_workflows/`（`cli_args.py` / `run_with_isaac_env.sh` / `workflow_config.py`
  / `workflow_paths.py`）——`train_meta.py` 与 `play_stdw_adapt.py` 依赖但迁移时遗漏。
- 恢复 `utils/__init__.py` + `utils/stdw_buffer.py`——`play_stdw_adapt.py` 经 importlib 加载。
- `sweep_full_matrix.py` 行34-37：过时路径 `workflows_new_stdw/` / 裸 `custom_workflows/`
  改为 `workflows/` + `workflows/configs/` + `custom_workflows/run_with_isaac_env.sh`。
- `play_stdw_adapt.py` 行316-324：迁移 sed 误把 bare import 加上 `easyuuv_stdw.` 前缀，
  但 sys.path 只含包目录本身（REPO_ROOT），报 `ModuleNotFoundError: No module named 'easyuuv_stdw'`。
  改回 bare import（`easyuuv_stdw_wrapper` / `stdw_integration` / `stdw_integration.plots`）。

**文档对齐**
- `COMMAND_CONTRACT.md` + `README.md` 的 C4/C5 flag 与代码对齐：
  `--policy/--output_dir/--run_dir` → `--policy_path/--out_root/--matrix_dir`。

**冒烟（全部 PASS）**
- C3 off（calm/base/identity，100 步）：`final_mse=7.679`，`nonfinite_guard_count=0`。
- C3 on（calm/base/full，200 步，drift 40-180）：`final_mse=6.29`，`final_mse_after_drift=4.27`。
- C4 sweep（1 cell）：`success=1`，`full_matrix.csv` 写出。
- C5 aggregate：`summary_aggregated.{json,csv}` + `stdw_pairwise.csv` 写出。

**已知差异**
- 冒烟用的迁移源 ckpt `experiment_name=warpauv_parametric`（非注册表的 `easyuuv_parametric`）。
  需用 `--experiment_name warpauv_parametric` 加载；新训练后回归 `easyuuv_parametric`。

---

## 2026-06-06 — 仓库独立化

**新增**
- 从 `direct/isaac-auv-env-new/` 重构出独立的 `direct/easyuuv_stdw/` 仓库。
- 注册 Gym 任务 `EasyUUV-Direct-v1` (4D) 和 `EasyUUV-Direct-Parametric-v1` (8D)。
- `eval/` 子模块（不依赖 Isaac Lab）：`wrappers.py` / `policy_loader.py` / `train_loop.py`
  / `deploy_eval.py` / `examples/`。
- `docs/` 文档体系（9 个 md）：INDEX / INTERFACE / COMMAND_CONTRACT / ARCHITECTURE
  / ERROR_CASES / RESEARCH_ANALYSIS / EVAL_SOP / AGENT_HANDOFF / CHANGELOG。
- 6 个核心函数加 doxygen 注释：`_pre_physics_step` / `_apply_action` / `_compute_dynamics`
  / `get_current_fluid_velocity` / `_refresh_domain_randomization_defaults` / `_reset_idx`。
- 命令契约：5 条契约（C1–C5）收敛到 `docs/COMMAND_CONTRACT.md`。

**变更**
- 类名重命名：`WarpAUVEnv` → `EasyUUVEnv`，`WarpAUVStdwWrapper` → `EasyUUVStdwWrapper`，
  `WarpAUVPPORunnerCfg` → `EasyUUVPPORunnerCfg` 等。
- `experiment_name`：`warpauv_direct` / `warpauv_parametric` → `easyuuv_direct` / `easyuuv_parametric`。
- import 路径：`from warpauv_stdw_wrapper` → `from easyuuv_stdw.easyuuv_stdw_wrapper`，
  其他 cross-module 导入同步修复。

**保留（不要改）**
- `WARPAUV_CFG` 常量名与 `data/warpauv/` 目录名（USD 内部对资产路径有硬引用）。
- `_legacy/` 历史文档（论文复现 anchor）。

---

## 2026-06-06 — 全矩阵 STDW 评估完成

- 跑完 48-cell 全矩阵评估（3 wave × 4 emb × 2 tune × 2 stdw × 1 seed），48/48 成功。
- 关键发现：
  - STDW 在 base / long_body / asymmetric 上稳定改善 final_mse 28%–47%。
  - 跨 calm / medium / storm 三档 wave 收益几乎不变（−38.6 ± 0.1%）。
  - `heavy_moderate` 反向：calm × full × on 劣化 +75%。
  - `tune=full` vs `identity` 仅 +1.9%（统计噪声内）—— 8D meta-control head 未收敛。
- 完整报告：[`_legacy/REPORT_full_matrix_20260606.md`](../archive/REPORT_full_matrix_20260606.md)。

---

## 2026-06-06 — JONSWAP yaml 注入 bug 修复

- 修复 `play_stdw_adapt.py` 中 `_set_initial_disturbance` 用 CLI 默认值覆盖 yaml 注入的三层 bug。
- 给 `apply_runtime_domain_shift` 加 6 个 jonswap kwargs，变更时 `_wave_manager` 失效以重建。
- 验证：calm/medium/storm step 100 处 fluid_vx 分别为 0.058 / 0.093 / 0.187 m/s。

---

## 2026-06-05 — 极简训练日志接入

- `train_meta.py` 加 `_attach_compact_logger`，monkey-patch `runner.log` 写两份 JSONL：
  - `compact_log.jsonl`：每行 `{iter, reward, ep_len, vloss, ploss, timesteps, noise_std}`。
  - `mse_curve.jsonl`：每行 `{iter, log_mse, n}`。
- 不再依赖 train.log regex 解析。

---

## 2026-06-05 — 8 维 meta-control 钩子整套补回

- 之前一轮迭代时丢失的整定钩子整套补回 `easyuuv_env.py`：
  - `__init__` 末尾增 `_tune_gains_enabled`/`_a_gain_buf`/`_zeta_nominal`/`_zeta_runtime`
    /`_last_pe`/`_last_deadzone`/`_sim_time_s`，并条件实例化 `_gain_tuner`。
  - `_refresh_domain_randomization_defaults` 中捕获 `_zeta_nominal = PID_args[:,:,0].clone()`
    + `tuner.reset()`。
  - `_pre_physics_step` 拆 8 维：后 4 存到 `_a_gain_buf`。
  - `_apply_action` 调 `tuner.step()` 路由到 `PID_args[:,:,col]`。
  - `_reset_idx` 末尾 `tuner.reset(env_ids)` + sim_time/a_gain reset。

---

## 2026-06-05 — 72-cell 训练 sweep

- 跑完 72-cell training sweep。报告 [`_legacy/REPORT_full_train_72cell_stdw_20260605.md`](../archive/REPORT_full_train_72cell_stdw_20260605.md)。

---

## 早于 2026-06-05

历史变更见 `_legacy/` 内的各份报告。

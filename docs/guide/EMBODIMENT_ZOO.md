# EMBODIMENT ZOO — 多机型仿真使用手册

本文档面向**使用者**，说明 EasyUUV-STDW 新增的多 embodiment 仿真能力：如何选择机型、每类机型的物理设计与差异化任务、下沉后 flip360 的近边界安全协议、ckpt 适用矩阵，以及 M1–M6 各机制的默认启用状态。

> 设计原则：新 embodiment 全部 **opt-in**，`base`/`asymmetric` 保持原硬编码路径与数值行为（零行为变更）。数值列先给**设计值**，Isaac 冒烟/eval 后回填**实测值**（标 `【TODO：实测】`）。
>
> 具体 smoke test、训练命令、验收标准与失败处理见 [`EMBODIMENT_ZOO_RUNBOOK.md`](EMBODIMENT_ZOO_RUNBOOK.md)。

---

## 1. 机型总览

| `--embodiment` | 类型 | 推进器 | 可控 DOF | 分配路径 | 状态 |
|---|---|---|---|---|---|
| `base` | 全驱 8 推 | 8（4 垂直 + 4 水平） | roll/pitch/yaw/depth | 旧硬编码混合块 | 现有，默认 |
| `asymmetric` | 全驱 8 推（COB 偏移） | 8 | roll/pitch/yaw/depth | 旧硬编码混合块 | 现有 |
| `long_body`/`heavy_*` | 全驱 8 推变惯量 | 8 | roll/pitch/yaw/depth | 旧硬编码混合块 | 现有 |
| `uuv6` | 全驱 6 推 | 6（4 垂直 + 2 水平） | roll/pitch/yaw/depth | config 驱动 B⁺ | **新增** |
| `uuv4` | **欠驱动** 4 推 | 4（4 垂直） | roll/pitch/depth（**无 yaw**） | config 驱动 B⁺（WLS 屏蔽 yaw） | **新增** |
| `uuv6_angled` | 全驱 6 推**非正交** | 6 | roll/pitch/yaw/depth | config 驱动 B⁺ | **新增** |
| `uuv4_angled` | 欠驱动 4 推**非正交** | 4 | roll/pitch/depth | config 驱动 B⁺（WLS 屏蔽 yaw） | **新增** |
| `remus`（AUV） | 单推进器 + 舵面 | surge + 舵耦合转向 | — | **本轮仅文档**（见 §6） |

---

## 2. 物理参数设计表

密度约束：所有机型 ρ_body = mass/volume **略小于水** ρ_water=997 kg/m³，形成微正浮力，使近边界（残余浮力/自由面）效应假设成立且 flip360 后能自然回浮，不至沉底。

| 机型 | mass (kg) | volume (m³) | ρ_body (kg/m³) | 净浮力% | inertia [Ix,Iy,Iz]（设计） |
|---|---|---|---|---|---|
| base（参考） | 22.701 | 0.022748 | 997.9 | ~0（中性） | [0.37, 0.97, 1.19] |
| uuv6 | 29.70 | 0.030000 | 990.0 | +0.7% | [0.55, 1.45, 1.78] |
| uuv4 | 21.78 | 0.022000 | 990.0 | +0.7% | [0.36, 0.94, 1.15] |
| uuv6_angled | 31.68 | 0.032000 | 990.0 | +0.7% | [0.60, 1.55, 1.90] |
| uuv4_angled | 23.76 | 0.024000 | 990.0 | +0.7% | [0.40, 1.02, 1.25] |

> inertia 为按几何/质量比缩放的设计值；实测由动力学识别回填。净浮力% = (ρ_water − ρ_body)/ρ_water。

---

## 3. 推进器排布（config 驱动）

推进器排布进 embodiment 注册表（Python，`easyuuv_env.py:embodiment_configs`），因几何是结构性的。分配矩阵 B∈ℝ^{6×N} 由排布几何构建，命令经伪逆 B⁺（或欠驱动加权最小二乘 WLS）分配到 N 个推进器。body-frame 6-DOF 顺序：`[Fx(surge), Fy(sway), Fz(heave), Tx(roll), Ty(pitch), Tz(yaw)]`；控制通道 `[roll, pitch, yaw, depth]` → `[Tx, Ty, Tz, Fz]`，surge/sway 命令置 0。

- **uuv6**（全驱 6 推）：4 垂直推进器（承 roll/pitch/depth）+ 2 水平推进器构成 yaw 力偶。三控制轴**互相正交**，B⁺ 天然解耦。
- **uuv4**（欠驱动 4 推）：仅 4 垂直推进器 → roll/pitch/depth 三维；**无水平推进器 → yaw 不可控**。`controllable_dofs` 屏蔽 Tz，WLS 忽略 yaw 行。
- **uuv6_angled / uuv4_angled**（非正交变种）：推进器朝向带 roll/pitch/yaw 倾角，使净控制轴**不互相垂直**。此时旧硬编码混合块失效——必须靠 B⁺ 吸收非正交耦合（这正是「S 面针对推力分配矩阵再设计」的落点：控制层仍出 4 通道，非正交性在**分配层**由 B⁺ 解耦，控制律不变）。

排布精确坐标见注册表；参照真实商业 AUV 的矢量推进器布局（引用见论文附录）。

---

## 4. 下沉后 flip360 近边界安全协议

用户要求：**所有** embodiment（含 base）先下沉，再 flip360，且深度不触水面。经 config 开关（默认关，保旧行为）：

| cfg 字段 | 默认 | 说明 |
|---|---|---|
| `submerge_phase_enable` | `false` | 开启 episode 内两相位：阶段 1 下沉+保持竖直，阶段 2 flip360 |
| `submerge_depth` | — | 阶段 1 目标下沉深度（世界 z，越小越深） |
| `submerge_hold_steps` | — | 阶段 1 持续步数 |
| `surface_guard_enable` | `false` | 开启破面守卫 |
| `surface_margin` | — | 破面判据裕度 |

**安全关系式**（必须满足以保证近边界效应有效）：
```
submerge_depth  <  z_surface − vehicle_height/2 − 裕度
```
默认 `z_surface=3.0`、`vehicle_height=0.3` → 完全浸没（submersion_ratio s=1）要求 `z < 2.85`。spawn `starting_depth=1.5` 已满足。破面守卫在 `z > z_surface − surface_margin` 时判出界，避免破面数据污染近边界统计。

z 约定：**世界 z-UP，z 越大越浅**，水面在 `z_surface=3.0`。

---

## 5. 差异化任务考验

不同 embodiment 施加不同难度（欠驱动/物理受限机型减载）：

| 机型 | 参考模式 | roll/pitch 幅 | yaw 幅 | 备注 |
|---|---|---|---|---|
| uuv6 | flip360_sine | ±π（全角） | 正常正弦 | 全驱，完整考验 |
| uuv4 | flip360_sine | ±π（全角） | **0**（yaw 幅置 0） | 欠驱动，不做 yaw 机动 |
| uuv6_angled | flip360_sine | ±π | 正常 | 非正交，考验 B⁺ 分配 |
| uuv4_angled | flip360_sine | ±π | **0** | 非正交 + 欠驱动 |
| remus（文档态） | 低频正弦 | 减小 | 减小 | 拉长时间、降变化率（见 §6） |

---

## 6. REMUS AUV（本轮仅文档）

REMUS 是**单推进器 + 舵面**（rudder / stern-plane）模型，与本仓库的矢量推进器 UUV 本质不同：

- 转向/俯仰力矩 ∝ **U²·δ**（前进速度平方 × 舵偏角），**零速时不可转向**——耦合于前进。
- 忠实移植需 Fossen 附加质量 MA·ν̇、科氏力 C(ν)、阻尼 D、恢复力 g(η)，**PhysX 刚体积分不足以复现**。

**本轮决策：skip 实施，仅记录设计与差距分析**（详见论文附录 §6）。后续若实施，任务考验应显著减载：低频正弦参考、减小幅度、拉长时间、降低变化率（因零速不可转向，必须维持巡航速度才能操纵）。参考实现见 [`PythonVehicleSimulator/.../remus100.py`](../../PythonVehicleSimulator/src/python_vehicle_simulator/vehicles/remus100.py)。

---

## 7. ckpt 适用矩阵

**第一要义：只靠 S-Surface 底层控制即可获得基本性能**，尽量共用现有 ckpt，不轻易重训。

| 机型 | 首选 ckpt | 是否需重训 | 理由 |
|---|---|---|---|
| base/asymmetric | `model_2398.pt` / SO(3) 系 | 否 | 现有主线 |
| uuv6 | `model_2398.pt`（复用） | 否（假设） | 全驱 + B⁺ 提供等价控制权威 |
| uuv6_angled | `model_2398.pt`（复用） | 否（假设） | 非正交由 B⁺ 吸收，控制律不变 |
| uuv4 | `model_2398.pt`（复用，yaw 任务降级） | 候选 | 欠驱动缺 yaw 权威，若底层不达标再登记重训 |
| uuv4_angled | `model_2398.pt`（复用） | 候选 | 同上 |

> 「假设/候选」由 Isaac 冒烟/eval 拍板，实测回填。仅当**底层控制器绝对不能**或结构性需要时才重训。

---

## 8. 命令示例

```bash
# 6 推全驱，下沉后 flip360
bash custom_workflows/run_with_isaac_env.sh workflows/play_stdw_adapt.py \
  --headless --task EasyUUV-Direct-Parametric-v1 --num_envs 1 \
  --embodiment uuv6 \
  --workflow_config workflows/configs/embodiment_uuv6.yaml \
  --load_run 2026-06-08_13-48-14_stage2 --checkpoint model_2398.pt \
  --total_steps 3000 --use_stdw True --target_drift 0.05

# 欠驱动 4 推（yaw 幅=0）
#   --embodiment uuv4 --workflow_config workflows/configs/embodiment_uuv4.yaml

# base 下沉后 flip360 协议验证
#   --embodiment base --workflow_config workflows/configs/embodiment_submerge_flip360.yaml

# 机型矩阵（dry-run 预览）
python3 workflows/sweep_stdw_safety_pressure.py --profile embodiment_zoo --dry_run
```

---

## 9. M1–M6 默认启用速查表

| 机制 | 开关 | 默认值 | 默认启用？ |
|---|---|---|---|
| M1 Lyapunov V 重定义 | `--lyapunov_v_mode` | `pose_quadratic` | mask 在 STDW 路径生效，V 定义默认=旧 `pose_quadratic`；其余模式默认关 |
| M2 方向性硬约束 | `--stdw_dir_guard` | `off` | **否** |
| M3 控制器 mismatch 放大 | `--pid_multipliers`/`--ctrl_mismatch` | 空 | **否**（诊断注入） |
| M4 近边界效应 | `--boundary_effect` | `None` | **否**（零修正） |
| M5 E-SUOT 域适应 | `--domain_adapt_backend` | `opr` | 默认走 **OPR**；E-SUOT 默认关 |
| M6 实验矩阵 | `--profile` | — | 驱动器，非运行期机制 |
| SO(3) S-Surface | `attitude_error_mode` | `euler` | **否**（旧欧拉逐轴） |
| 多 embodiment | `--embodiment` | `base` | 新机型全 opt-in |
| 下沉相位 / 水面守卫 | `submerge_phase_enable`/`surface_guard_enable` | `false` | **否** |

> **结论：默认全部为旧 baseline 行为**（euler S-Surface + OPR + 无边界效应 + 无方向守卫 + V=pose_quadratic + base 机型 + 无下沉相位）。所有新机制/机型均 opt-in。

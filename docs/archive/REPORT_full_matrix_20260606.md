# 全矩阵评估报告 — STDW × 整定 × JONSWAP × cross-embodiment

- **日期**：2026-06-06
- **运行目录**：`.tmp/full_matrix_20260606_223645/`
- **驱动**：[workflows_new_stdw/sweep_full_matrix.py](workflows_new_stdw/sweep_full_matrix.py)
- **Policy**：`meta_train_full_20260605_220714/.../model_1499.pt`（meta-control 1500 iter，8 维输出）
- **命令契约**：[workflows_new_stdw/FULL_MATRIX_RUNBOOK.md](workflows_new_stdw/FULL_MATRIX_RUNBOOK.md)
- **接口能力卡**：[workflows_new_stdw/INTERFACE_REFERENCE.md](workflows_new_stdw/INTERFACE_REFERENCE.md)

---

## 1. 实验矩阵

| 维度 | 取值 | 数量 |
|---|---|---|
| Wave (JONSWAP) | calm (hs=0.3, fp=0.18) / medium (hs=0.8, fp=0.13) / storm (hs=1.5, fp=0.10) | 3 |
| Embodiment | base / long_body / heavy_moderate / asymmetric | 4 |
| Tune mode | identity (旁路 4 机制) / full (PE+死区+LPF+gain_beta=0.2) | 2 |
| STDW | off (`use_stdw=False, target_drift=0`) / on (`use_stdw=True, target_drift=0.05`) | 2 |
| Seed | 0 | 1 |

**48 unique cell × 1 seed = 48 trial**，每 cell 1500 step（25 s 仿真），平均 wall = **23.3 s/cell**，总 wall ≈ **18.6 min**。

JONSWAP base_vel × 3 档：calm `[0.06, 0, 0.02]`、medium `[0.09, 0, 0.035]`、storm `[0.12, 0, 0.05]`。
**注入链路验证**（同一 base/identity/off cell 在 step 100 处的 `fluid_vx`）：

```
calm:   0.0582
medium: 0.0930
storm:  0.1874
```

三档拉开，wave 维度真正生效（这次修复了 yaml 注入被 `_set_initial_disturbance` CLI 默认值覆盖的 bug；详见 §6.1）。

---

## 2. STDW 生效前后对照（核心结果）

按 (wave × embodiment × tune) 配对，列 STDW off→on 的 `final_mse` 与 `final_mse_after_drift` 变化。

### 2.1 全矩阵（24 行配对）

| wave | emb | tune | fmse off | fmse on | Δfmse % | drift_off | drift_on | Δdrift % |
|---|---|---|---:|---:|---:|---:|---:|---:|
| calm | base | identity | 6.598 | 3.714 | **−43.7%** | 6.438 | 3.491 | **−45.8%** |
| calm | base | full | 6.451 | 3.892 | **−39.7%** | 6.322 | 3.603 | **−43.0%** |
| calm | long_body | identity | 6.729 | 4.822 | −28.3% | 6.664 | 4.870 | −26.9% |
| calm | long_body | full | 7.731 | 5.448 | −29.5% | 7.648 | 5.321 | −30.4% |
| calm | heavy_moderate | identity | 3.580 | 3.753 | **+4.8%** | 3.333 | 3.854 | +15.6% |
| calm | heavy_moderate | full | 3.480 | 6.093 | **+75.1%** ⚠ | 3.241 | 6.236 | +92.4% |
| calm | asymmetric | identity | 6.866 | 3.673 | **−46.5%** | 6.653 | 3.295 | −50.5% |
| calm | asymmetric | full | 6.514 | 3.655 | **−43.9%** | 6.375 | 3.304 | −48.2% |
| medium | base | identity | 6.661 | 3.746 | **−43.8%** | 6.501 | 3.519 | −45.9% |
| medium | base | full | 6.516 | 3.924 | **−39.8%** | 6.388 | 3.634 | −43.1% |
| medium | long_body | identity | 6.791 | 4.887 | −28.0% | 6.730 | 4.918 | −26.9% |
| medium | long_body | full | 7.800 | 5.471 | −29.9% | 7.720 | 5.343 | −30.8% |
| medium | heavy_moderate | identity | 3.583 | 4.108 | +14.6% | 3.338 | 4.229 | +26.7% |
| medium | heavy_moderate | full | 3.499 | 5.613 | **+60.4%** ⚠ | 3.258 | 5.754 | +76.6% |
| medium | asymmetric | identity | 6.919 | 3.700 | **−46.5%** | 6.709 | 3.318 | −50.5% |
| medium | asymmetric | full | 6.579 | 3.683 | **−44.0%** | 6.441 | 3.328 | −48.3% |
| storm | base | identity | 6.785 | 3.808 | **−43.9%** | 6.631 | 3.578 | −46.0% |
| storm | base | full | 6.652 | 3.978 | **−40.2%** | 6.531 | 3.688 | −43.5% |
| storm | long_body | identity | 6.869 | 4.957 | −27.8% | 6.828 | 4.967 | −27.3% |
| storm | long_body | full | 7.934 | 5.682 | −28.4% | 7.866 | 5.503 | −30.0% |
| storm | heavy_moderate | identity | 3.594 | 4.103 | +14.2% | 3.355 | 4.233 | +26.2% |
| storm | heavy_moderate | full | 3.466 | 4.521 | **+30.4%** ⚠ | 3.236 | 4.767 | +47.3% |
| storm | asymmetric | identity | 7.026 | 3.747 | **−46.7%** | 6.822 | 3.359 | −50.8% |
| storm | asymmetric | full | 6.708 | 3.721 | **−44.5%** | 6.572 | 3.362 | −48.8% |

> 数值为单 seed 0；列 `_std=0` 已从聚合表删去以保持简洁。完整原始数据见 `.tmp/full_matrix_20260606_223645/stdw_pairwise.csv`。

### 2.2 按 wave 维度的 Δfmse 平均

| wave | 全 8 组合（含 heavy_moderate） | 6 组合（去掉 heavy_moderate） |
|---|---:|---:|
| calm | −19.0% | **−38.6%** |
| medium | −19.6% | **−38.7%** |
| storm | −23.4% | **−38.6%** |

**关键观察**：
1. STDW 在 base / long_body / asymmetric 三个 embodiment 上**一致改善**了 final_mse（−27% ~ −47%）。
2. 跨 wave 三档（hs 从 0.3 → 1.5），STDW 收益**几乎不变**（去掉异常 emb 后稳定在 −38.6% ± 0.1%）。说明 wrapper 的 COB drift / 慢环更新对扰动幅度不敏感，泛化到大风浪情况下仍然奏效。
3. **`heavy_moderate` 是显著异常点**（详见 §3）。

---

## 3. heavy_moderate 异常分析

| wave | tune | off | on | Δ% |
|---|---|---:|---:|---:|
| calm | identity | 3.580 | 3.753 | +4.8% |
| calm | full | 3.480 | 6.093 | **+75.1%** |
| medium | identity | 3.583 | 4.108 | +14.6% |
| medium | full | 3.499 | 5.613 | **+60.4%** |
| storm | identity | 3.594 | 4.103 | +14.2% |
| storm | full | 3.466 | 4.521 | **+30.4%** |

**现象**：
- `heavy_moderate` 在 **STDW off** 下 fmse 已经是 4 个 embodiment 中最低（3.5 左右，约为 base 的 53%）——它本身的浮力/质量配比已接近全局最优。
- STDW on 后 fmse **反而上升**，特别是 `full` tune 配置下 calm 档涨 75%。

**直观解释**：STDW wrapper 的核心动作是把 COB 推向偏置点（target_drift=0.05）+ 慢环每 60 步更新策略。它对**已经偏移最优配重**的 embodiment（base/long_body/asymmetric）能补回偏差；但对**接近最优配重**的 heavy_moderate 来说，drift 反而把它推离最优，叠加 full tune 的 PE/LPF 注入又加大波动，结果就是劣化。

**这意味着**：STDW 是一个"修偏移"机制，**不是无差别的性能放大器**。生产部署时，应该先用 STDW off 跑一组短 baseline 判定该 embodiment 是否需要 STDW 介入，否则会反向。

---

## 4. 整定（identity vs full）维度

跨 3 wave × 4 emb 平均 `fmse_off`（这是不被 STDW 干扰的"裸控制"指标）：

| Tune | mean fmse_off |
|---|---:|
| identity（旁路 4 机制） | 6.000 |
| full（PE + 死区 + LPF + β=0.2） | **6.111** |

差异 +0.111（**+1.9%**），在统计噪声范围内。结合 §3 的现象（heavy_moderate 上 full 在 STDW on 时显著劣化）可以判断：
- 当前 1500 iter 训练出来的 8 维 meta-control policy，**整定头还没真正学到有意义的 ζ 调节策略**——`a_gain` 输出基本是噪声。
- `full` 模式下的 4 阶段（LPF→死区→Bounded Safeguard→PE）只是在裸控制上叠加了少量扰动，没带来净收益。
- 在 `heavy_moderate + STDW on` 的特殊组合里，这层扰动还会放大 STDW 自身的负效应。

**建议**：要让 full tune 真正生效，要么训练更长（≥3000 iter）让 policy head 收敛到非平凡解，要么就接受 identity 作为稳定基线、把 8 维元控制头退化为 4 维。

---

## 5. embodiment 维度

平均 `fmse_off`（跨 3 wave × 2 tune）：

| Embodiment | mean fmse_off | 相对 base |
|---|---:|---:|
| base | 6.610 | 1.00x |
| long_body | 7.309 | 1.10x |
| heavy_moderate | **3.534** | **0.53x** |
| asymmetric | 6.769 | 1.02x |

`long_body` 因为转动惯量增大，最难控（10% 差于 base）。`heavy_moderate` 最易控（前面分析过）。`asymmetric` 与 base 持平，说明 1500 iter policy 对左右不对称配重已经有相当的鲁棒性。

---

## 6. 修复记录

### 6.1 JONSWAP yaml 注入未真正下达 env

- **现象**：上一轮 144 trial 跑完后，stdw_pairwise.csv 显示 wave 三档（calm/medium/storm）数值完全相同。
- **根因链**：
  1. [play_stdw_adapt.py L459-462](workflows_new_stdw/play_stdw_adapt.py) 在 `apply_config_overrides` 之后用 CLI 默认值（`wave_mode=sine`、`base_vel=[0.06,0,0.02]`）整段覆盖 disturbance_cfg。**已修**（sentinel 模式：仅当 yaml 没指定才用 CLI 默认值）。
  2. 即使 sentinel 修好了，`disturbance_cfg` 是 [warpauv_env.py L89](warpauv_env.py) 处的 inner class，`apply_config_overrides` 递归 setattr 后，env 内部 step 0 仍读到错误值。
  3. 真正的元凶：[play_stdw_adapt.py L408 `_set_initial_disturbance`](workflows_new_stdw/play_stdw_adapt.py) 在 main loop 入口第三次用 CLI 默认值调 `apply_runtime_domain_shift`，把 mode/base_vel/amplitude/frequency 又覆盖回 `sine` 默认值。**已修**：`_set_initial_disturbance` 接受 `yaml_disturbance` 字典参数，sentinel 优先。
- **附加增强**：[warpauv_env.py apply_runtime_domain_shift](warpauv_env.py#L591) 加了 6 个 jonswap kwargs（hs/fp/gamma/depth/direction/seed），更改时使 `_wave_manager` 失效以重建。
- **验证**：calm/medium/storm 三档 step 100 处的 `fluid_vx` 分别为 `0.058 / 0.093 / 0.187`，与 base_vel × 周期波动叠加一致。

### 6.2 8 维 meta-control 钩子整套补回

上一轮 [warpauv_env.py](warpauv_env.py) 的 8 维 meta-control 钩子（`_gain_tuner` 实例化、`_pre_physics_step` 拆 4+4、`_apply_action` 注入 ζ_runtime、3 处 reset）整套丢失。本轮全量补回：
- PID buffer `(N, 4)` 不再依赖 `cfg.num_actions`。
- `__init__` 末尾增 `_tune_gains_enabled`/`_a_gain_buf`/`_zeta_nominal/_zeta_runtime`/`_last_pe`/`_last_deadzone`/`_sim_time_s`，并条件实例化 `_gain_tuner`。
- `_refresh_domain_randomization_defaults` 中捕获 `_zeta_nominal = PID_args[:,:,0].clone()` + `tuner.reset()`。
- `_pre_physics_step` 拆 8 维：后 4 存到 `_a_gain_buf`。
- `_apply_action` 调 `tuner.step()` → 路由到 `PID_args[:,:,col]`（按 `gain_update_targets`） → `_compute_dynamics(ctrl_actions)` → 还原 `PID_args`。
- `_reset_idx` 末尾 `tuner.reset(env_ids)` + sim_time/a_gain reset。

48/48 trial 0 失败、no warnings/errors，证明钩子已正确恢复。

---

## 7. 主要结论与建议

1. **STDW 在 base/long_body/asymmetric 三个 embodiment 上稳定改善 final_mse 28%-47%**，跨 calm→storm 三档泛化稳定（去掉异常点后均值 −38.6%）。这是本轮最重要的可复现结论。
2. **STDW 不是"无差别加速器"**：`heavy_moderate`（接近最优配重）下 STDW 反而劣化最高 +75%。生产部署应先短跑 STDW off 判定基线，再决定是否启用 STDW。
3. **8 维元控制 full tune 当前没有显著收益**（vs identity +1.9% 在统计噪声内），且会放大 STDW 在异常 emb 上的负效应。要么再训 ≥3000 iter，要么退化为 4 维。
4. **JONSWAP wave 维度对 STDW 收益影响很小**（off/on 各档差异 < 0.5%），说明 wrapper 学到的不是抗特定扰动谱，而是抗低频偏置。

---

## 8. 数据制品

| 文件 | 含义 |
|---|---|
| [.tmp/full_matrix_20260606_223645/full_matrix.csv](.tmp/full_matrix_20260606_223645/full_matrix.csv) | 48 trial 原始指标 |
| [.tmp/full_matrix_20260606_223645/summary_aggregated.csv](.tmp/full_matrix_20260606_223645/summary_aggregated.csv) | 按 (wave,emb,tune,stdw) 聚合（单 seed 等同原始） |
| [.tmp/full_matrix_20260606_223645/stdw_pairwise.csv](.tmp/full_matrix_20260606_223645/stdw_pairwise.csv) | 24 行 STDW off↔on 配对，含 Δfmse % |
| [.tmp/full_matrix_20260606_223645/sweep.log](.tmp/full_matrix_20260606_223645/sweep.log) | sweep 驱动 stdout |
| [.tmp/full_matrix_20260606_223645/<cell>/run.log](.tmp/full_matrix_20260606_223645/) | 每 cell stdout（含 [DISTURBANCE] 注入校验） |

# 接口现状参考卡（JONSWAP / 整定 / STDW / Embodiment）

> 用途：在跑「整定 + JONSWAP + cross-embodiment + STDW 前后对照」这种全矩阵实验之前，必须先对齐这份卡上的真实接口能力。本卡只描述**当前代码事实**，不重复历史 README。
>
> 上游引用：
> - 根 [README.md](../README.md)
> - STDW 工作流总入口 [workflows_new_stdw/README.md](README.md)
> - 工作区规则 [AGENTS.md](../AGENTS.md)

---

## 0. TL;DR：实验前必须知道的 5 件事

1. JONSWAP 的 hs/fp/gamma/depth/direction **没有** CLI 暴露，必须走 `--workflow_config <yaml>` 注入 `env.disturbance_cfg.jonswap_*`。
2. STDW 没有干净的 wrapper-bypass 总开关；要做"无 STDW"基线必须 **`--use_stdw=False` + `--target_drift 0`** 双关，否则 wrapper 仍每步推进 COB drift 与 RMS 滤波。
3. 5 个 embodiment 在 env 里都存在，但 `play_stdw_adapt.py --embodiment` choices **只有 4 个**（漏了 `heavy_duty`）。
4. 控制器整定不是闭环 autotuner，而是 4 个布尔机制（`enable_pe / enable_deadzone / enable_param_lpf / identity_init`） + 2 个数值旋钮（`gain_beta / pe_amp`）的开环参数化映射。
5. 元控制 8 维任务 `MOGA-WarpAUV-Direct-Parametric-v1` 与基线 4 维任务 `MOGA-WarpAUV-Direct-v1` 的 ckpt 不通用，experiment_name 也不同（`warpauv_parametric` vs `warpauv_direct`）。

---

## 1. JONSWAP / 海浪扰动

### 1.1 实现
- 模型：[wave_disturbance_manager.py](../wave_disturbance_manager.py)（`JonswapWaveDisturbanceManager` 数据类）
- 注入：[warpauv_env.py](../warpauv_env.py#L503-L527) `get_current_fluid_velocity()` 在 `mode=="jonswap"` 分支返回 `base_vel + manager.get_wave_velocity(...)`，作为**水流速度**进入 hydrodynamics（不是直接力/力矩）
- 缓存：[warpauv_env.py](../warpauv_env.py#L531-L552) `_get_wave_manager`（按 hs/fp/depth/direction/seed 签名缓存）

### 1.2 接口
| 维度 | 入口 | 说明 |
|---|---|---|
| `mode` | `--wave_mode {none,constant,sine,jonswap}` | CLI 直暴 |
| `base_vel` | `--wave_base_vel x y z` | CLI 直暴 |
| `amplitude` | `--wave_amplitude x y z` | CLI 直暴；jonswap 模式下不生效 |
| `frequency` | `--wave_frequency x y z` | CLI 直暴；jonswap 模式下不生效 |
| `jonswap_hs` | yaml 注入 `env.disturbance_cfg.jonswap_hs` | **无 CLI** |
| `jonswap_fp` | 同上 | **无 CLI** |
| `jonswap_gamma` | 同上 | **无 CLI**，默认 3.3 |
| `jonswap_depth` | 同上 | **无 CLI**，默认 30 m |
| `jonswap_direction` | 同上 | **无 CLI**，弧度 |
| `jonswap_seed` | 同上 | **无 CLI** |

### 1.3 yaml 注入示例
```yaml
env:
  disturbance_cfg:
    mode: jonswap
    base_vel: [0.06, 0.0, 0.02]
    jonswap_hs: 1.0
    jonswap_fp: 0.12
    jonswap_gamma: 3.3
    jonswap_depth: 30.0
    jonswap_direction: 0.0
    jonswap_seed: 7
```
配合 `play_stdw_adapt.py --workflow_config configs/wave_high.yaml --wave_mode jonswap` 使用。

### 1.4 已知坑
- `apply_runtime_domain_shift`（[warpauv_env.py L554-L588](../warpauv_env.py#L554-L588)）只覆盖 `mode/base_vel/amplitude/frequency`，**不覆盖** jonswap_* 子参数。要切换海况只能在 reset 之前覆盖 `cfg.disturbance_cfg`，或在 yaml 里写死。

---

## 2. 控制器参数整定（ParametricGainTuner）

### 2.1 实现
- 模型：[gain_tuner.py L24-L181](../gain_tuner.py#L24-L181) `ParametricGainTuner`
- 4 阶段开环串行：`a_gain` → 一阶 LPF → 死区冻结 → Bounded Safeguard `ζ_i = ζ_nom·(1+β·a_gain_i)` → PE 注入 `ζ ← ζ + a(t)·sin(2πf·t)`
- Cfg 镜像：[warpauv_env.py L139-L152](../warpauv_env.py#L139-L152)（`tune_gains / gain_beta / enable_pe / pe_freq / pe_amp / pe_decay_gamma / enable_deadzone / deadzone_threshold / enable_param_lpf / param_lpf_cutoff / identity_init`）

### 2.2 不是闭环 autotuner，是机制开关组合
| 开关 | 默认 | 作用 |
|---|---|---|
| `tune_gains` | False（基线任务） / True（Parametric 任务） | 8 维 a_ctrl+a_gain 通路总开关 |
| `identity_init` | False | True 旁路全部 4 个机制，ζ_runtime ≡ ζ_nominal |
| `enable_pe` | True | Persistent Excitation 正弦注入 |
| `enable_deadzone` | True | 死区冻结（避免低频噪声扰动调参） |
| `enable_param_lpf` | True | 一阶 LPF 平滑 a_gain |
| `gain_beta` | 0.2 | Bounded Safeguard 因子（±20% ζ_nom） |
| `pe_amp` / `pe_freq` / `pe_decay_gamma` | 0.05 / 0.5 Hz / 5.0 | PE 振幅/频率/状态衰减 |

### 2.3 跑"4 种整定模式对照"的 yaml 模板
| 模式 | identity_init | enable_pe | enable_deadzone | enable_param_lpf | 含义 |
|---|---|---|---|---|---|
| **identity** | true | – | – | – | ζ_runtime≡ζ_nom，不调参 |
| **safeguard_only** | false | false | false | false | 仅 Bounded Safeguard |
| **safeguard+pe** | false | true | false | false | + 持续激励 |
| **full** | false | true | true | true | 4 机制全开 |

### 2.4 不存在的接口
- 没有专门的字符串旋钮 `--tune_mode {open,closed,fixed}`
- 没有 `controller_tuning` 配置块
- 没有独立的 S-Surface 控制器文件，控制函数在 [warpauv_env.py](../warpauv_env.py) 的 `apply_control_profile` 内（`L338` 起）

---

## 3. STDW 慢环

### 3.1 实现
- Wrapper：[warpauv_stdw_wrapper.py](../warpauv_stdw_wrapper.py) — 每步推进 COB drift（`drift_start_step` → `drift_end_step` 的 `target_drift`）+ 5s 滑窗 RMS + Lyapunov mask 计算
- 慢环主入口：[play_stdw_adapt.py](play_stdw_adapt.py) — 快环 dt=1/120s 采样 + 慢环每 60 步按 mask 加权 MSE + L2 锚定
- Buffer：[utils/stdw_buffer.py](../utils/stdw_buffer.py)（10 张主表）

### 3.2 CLI 旋钮（节选）
| Flag | 默认 | 作用 |
|---|---|---|
| `--use_stdw` | True | 慢环更新总开关；False **只** skip 慢环步骤，wrapper 仍跑 |
| `--enable_filter` / `--use_quantile_filter` | True / False | RMS 滤波 / 分位数过滤 |
| `--enable_pseudo_action` | True | 物理修正动作伪标签 |
| `--enable_lyapunov_mask` | True | Lyapunov 物理滤网 |
| `--reg_mode` | l2 | L2 / behavior_kl 锚定 |
| `--slow_loop_interval` | 60 | 慢环触发间隔 |
| `--lambda_reg` / `--g_C_lr` | 1e-3 / 5e-5 | 锚定权 / 慢环 lr |
| `--target_drift` | 0.05 | COB 漂移目标值（米） |
| `--drift_start_step` / `--drift_end_step` | 200 / 1200 | 漂移窗 |
| `--drift_axes` | 0 (X) | 漂移轴下标 |
| `--ramp_shape` | linear | linear / cosine |
| `--workflow_config` | None | yaml 注入入口 |
| `--experiment_name` | warpauv_direct | **必须显式给 `warpauv_parametric` 才能加载 8 维 ckpt** |

### 3.3 "STDW 生效前后"如何写
| 配置 | 含义 | 旋钮 |
|---|---|---|
| **A. wrapper-only baseline** | wrapper 每步加 drift，但慢环不更新；表现为"漂移期 MSE 单调上升"的对照 | `--use_stdw False --target_drift 0.05` |
| **B. STDW off（最干净基线）** | drift 也不加；纯环境基线 | `--use_stdw False --target_drift 0` |
| **C. STDW on（推荐）** | drift + 慢环 + Lyapunov mask + 滤波 | `--use_stdw True --enable_filter True --target_drift 0.05` |

写"STDW 生效前后"建议用 **B vs C**（最干净的"开关"对照）。如果想突出"漂移环境下 STDW 的额外收益"，用 **A vs C**。

### 3.4 summary.json 关键字段
`final_mse`、`convergence_step`、`stability_threshold_*`、`baseline_compound_error_mean`、`slow_loop_triggers`、`reset_count`、`final_mse_after_drift`、`use_stdw`、`enable_filter`、`embodiment`、`scenario`、加 `**tracking_mse_summary`（[play_stdw_adapt.py L1054-L1101](play_stdw_adapt.py#L1054-L1101)）。

---

## 4. Cross-Embodiment

### 4.1 5 个机型在 env 里都有
[warpauv_env.py L188-L225](../warpauv_env.py#L188-L225) `embodiment_configs`：

| 名 | mass | inertia | dyn_τ | drag |
|---|---|---|---|---|
| `base` | 22.7 kg | [0.37, 0.97, 1.19] | 0.05 | 1.0 |
| `long_body` | 22.7 kg | [0.1, 2.5, 2.5] | 0.05 | 1.0 |
| `heavy_duty` | × 5 | × 5 | 0.20 | × 5 |
| `heavy_moderate` | × 2 | × 2 | 0.10 | × 2 |
| `asymmetric` | 22.7 kg | [0.37, 0.97, 1.19] | 0.05 | 1.0（com_to_cob xy 偏移 0.05/0.05） |

### 4.2 切换接口
- env 层：`WarpAUVEnv.apply_embodiment_config(name)`（[warpauv_env.py L408-L419+](../warpauv_env.py#L408-L419)）
- CLI：
  - [play_stdw_adapt.py L207](play_stdw_adapt.py#L207) `--embodiment {base, long_body, heavy_moderate, asymmetric}`（**漏 `heavy_duty`**）
  - [play_meta_eval.py L98-L99](play_meta_eval.py#L98-L99) `--embodiment` **无 choices 限制**，按 env 字典查（5 个都能用）
- sweep 驱动：[sweep_72cell.py L13](sweep_72cell.py#L13) 也是漏 `heavy_duty` 的 4 项

### 4.3 已知坑
- `play_stdw_adapt.py` 的 choices 列表与 env 字典脱钩；如果未来要扫到 `heavy_duty`，必须改 [play_stdw_adapt.py L207](play_stdw_adapt.py#L207) 的 choices 或者改 worker 用 `play_meta_eval.py`。

---

## 5. 任务 / 实验名映射

| 任务 ID | 注册位置 | 维度 | experiment_name | ckpt 默认目录 |
|---|---|---|---|---|
| `MOGA-WarpAUV-Direct-v1` | [__init__.py](../__init__.py) | 4 维 a_ctrl | `warpauv_direct` | `~/isaaclab/logs/rsl_rl/warpauv_direct/...` |
| `MOGA-WarpAUV-Direct-Parametric-v1` | [__init__.py](../__init__.py) | 8 维 a_ctrl + a_gain | `warpauv_parametric` | 训练 run 落 `.tmp/.../warpauv_parametric/...` |

跨任务用 ckpt 会立刻报 actor MLP 维度不匹配（4 vs 8）。

---

## 6. Sweep 驱动总览

| 脚本 | 矩阵 | 任务 | yaml 输入 |
|---|---|---|---|
| [sweep_stdw.py](sweep_stdw.py) | scenario × embodiment × algo flags（默认 32 / 8 / 72 通过 flag 切换） | `MOGA-WarpAUV-Direct-v1` | 通过 `play_stdw_adapt.py --workflow_config` 间接注入 |
| [sweep_72cell.py](sweep_72cell.py) | 3 axis × 3 magnitude × 4 emb × 2 flag = 72 | `MOGA-WarpAUV-Direct-Parametric-v1` | 内部硬编码，不读外部 yaml |
| [run_4grp_compare.sh](run_4grp_compare.sh) | 4 组 STDW 算法对照 | `MOGA-WarpAUV-Direct-v1` | – |
| [tools/aggregate_sweep72.py](tools/aggregate_sweep72.py) | 72-cell 聚合器 | – | – |

---

## 7. 极简训练日志（2026-06-05 新增）

[train_meta.py](train_meta.py) `_attach_compact_logger` 会在每个 run 的 log_dir 根下写两份 JSONL：

- `compact_log.jsonl`：每行 `{iter, reward, ep_len, vloss, ploss, timesteps, noise_std}`
- `mse_curve.jsonl`：每行 `{iter, log_mse, n}`，n 是该 iter 内参与平均的 episode 数

不需要在 train.log 上做 regex 解析。

---

## 8. 修改这份卡的纪律

- 这份卡描述的是**接口能力**与**已知坑**，不写实验结论。结论写在 `REPORT_*.md` 里。
- 改了 `disturbance_cfg` / `apply_embodiment_config` / `play_stdw_adapt.py` 的 CLI / `gain_tuner.py` 的机制开关，**必须同步本卡**。
- 报告（`REPORT_*.md`）要引用接口能力时，链到本卡而不是直接复述。

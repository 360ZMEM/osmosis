# EasyUUV-STDW Interface Reference

> 用途：执行任何"整定 + JONSWAP + cross-embodiment + STDW 前后对照"实验之前必须先对齐这份卡。
> 只描述**当前代码事实**，不重复历史 README。源自 `_legacy/INTERFACE_REFERENCE_legacy.md`，命名重构为 EasyUUV*。

---

## 0. TL;DR：实验前必须知道的 5 件事

1. JONSWAP 的 `hs/fp/gamma/depth/direction` **没有** CLI 暴露，必须走 `--workflow_config <yaml>` 注入
   `env.disturbance_cfg.jonswap_*`。
2. STDW 没有干净的 wrapper-bypass 总开关；要做"无 STDW"基线必须 **`--use_stdw=False` + `--target_drift 0`** 双关，
   否则 wrapper 仍每步推进 COB drift 与 RMS 滤波。
3. 5 个 embodiment 在 env 里都存在，但 `play_stdw_adapt.py --embodiment` choices **只有 4 个**（漏 `heavy_duty`）。
4. 控制器整定不是闭环 autotuner，而是 4 个布尔机制
   （`enable_pe / enable_deadzone / enable_param_lpf / identity_init`）+ 2 个数值旋钮（`gain_beta / pe_amp`）的开环参数化映射。
5. 元控制 8 维任务 `EasyUUV-Direct-Parametric-v1` 与基线 4 维任务 `EasyUUV-Direct-v1` 的 ckpt 不通用，
   experiment_name 也不同（`easyuuv_parametric` vs `easyuuv_direct`）。

---

## 1. JONSWAP / 海浪扰动

### 1.1 实现
- 模型：[`wave_disturbance_manager.py`](../../wave_disturbance_manager.py)（`JonswapWaveDisturbanceManager`）
- 注入：[`easyuuv_env.py`](../../easyuuv_env.py) `get_current_fluid_velocity()` 在 `mode=="jonswap"`
  分支返回 `base_vel + manager.get_wave_velocity(...)`，作为**水流速度**进入 hydrodynamics（不是直接力/力矩）。
- 缓存：同文件 `_get_wave_manager()` 按 `(hs/fp/depth/direction/seed)` 签名缓存。

### 1.2 接口
| 维度 | 入口 | 说明 |
|---|---|---|
| `mode` | `--wave_mode {none,constant,sine,jonswap}` | CLI 直暴 |
| `base_vel` | `--wave_base_vel x y z` | CLI 直暴；单位 m/s |
| `amplitude` | `--wave_amplitude x y z` | CLI；jonswap 模式下不生效 |
| `frequency` | `--wave_frequency x y z` | CLI；jonswap 模式下不生效，单位 Hz |
| `jonswap_hs` | yaml 注入 `env.disturbance_cfg.jonswap_hs` | **无 CLI**；m |
| `jonswap_fp` | 同上 | **无 CLI**；Hz |
| `jonswap_gamma` | 同上 | **无 CLI**；默认 3.3 |
| `jonswap_depth` | 同上 | **无 CLI**；默认 30，单位 m |
| `jonswap_direction` | 同上 | **无 CLI**；rad |
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
- `apply_runtime_domain_shift` 在 `easyuuv_env.py` 中只覆盖 `mode/base_vel/amplitude/frequency`，
  **不覆盖** `jonswap_*` 子参数。要切换海况要么 reset 之前覆盖 `cfg.disturbance_cfg`，
  要么在 yaml 里写死。具体 bug 与修复在 [`ERROR_CASES.md`](ERROR_CASES.md#case-1-jonswap-yaml-注入失效) §1。

---

## 2. 控制器参数整定（ParametricGainTuner）

### 2.1 实现
- 模型：[`gain_tuner.py`](../../gain_tuner.py) `ParametricGainTuner`
- 4 阶段开环串行：`a_gain` → 一阶 LPF → 死区冻结 → Bounded Safeguard `ζ_i = ζ_nom·(1+β·a_gain_i)` →
  PE 注入 `ζ ← ζ + a(t)·sin(2πf·t)`
- Cfg 镜像在 [`easyuuv_env.py`](../../easyuuv_env.py) 内（`tune_gains / gain_beta / enable_pe / pe_freq /
  pe_amp / pe_decay_gamma / enable_deadzone / deadzone_threshold / enable_param_lpf / param_lpf_cutoff /
  identity_init`）。

### 2.2 不是闭环 autotuner，是机制开关组合

| 开关 | 默认 | 作用 |
|---|---|---|
| `tune_gains` | False（基线任务）/ True（Parametric 任务） | 8 维 a_ctrl+a_gain 通路总开关 |
| `identity_init` | False | True 旁路全部 4 个机制，ζ_runtime ≡ ζ_nominal |
| `enable_pe` | True | Persistent Excitation 正弦注入 |
| `enable_deadzone` | True | 死区冻结（避免低频噪声扰动调参） |
| `enable_param_lpf` | True | 一阶 LPF 平滑 a_gain |
| `gain_beta` | 0.2 | Bounded Safeguard 因子（±20% ζ_nom），无量纲 |
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
- 没有独立 S-Surface 控制器文件，控制函数在 `easyuuv_env.py` 的 `apply_control_profile` 内

---

## 3. STDW 慢环

### 3.1 实现
- Wrapper：[`easyuuv_stdw_wrapper.py`](../../easyuuv_stdw_wrapper.py) — 每步推进 COB drift
  （`drift_start_step` → `drift_end_step` 的 `target_drift`）+ 5s 滑窗 RMS + Lyapunov mask
- 慢环主入口：[`workflows/play_stdw_adapt.py`](../../workflows/play_stdw_adapt.py) — 快环 dt=1/120s 采样 +
  慢环每 60 步按 mask 加权 MSE + L2 锚定
- Buffer：[`utils/stdw_buffer.py`](../../utils/stdw_buffer.py)（10 张主表）

### 3.2 CLI 旋钮（节选）

| Flag | 默认 | 作用 / 单位 |
|---|---|---|
| `--use_stdw` | True | 慢环更新总开关；False **只** skip 慢环步骤，wrapper 仍跑 |
| `--enable_filter` / `--use_quantile_filter` | True / False | RMS 滤波 / 分位数过滤 |
| `--enable_pseudo_action` | True | 物理修正动作伪标签 |
| `--enable_lyapunov_mask` | True | Lyapunov 物理滤网 |
| `--reg_mode` | l2 | L2 / behavior_kl 锚定 |
| `--slow_loop_interval` | 60 | 慢环触发间隔（step） |
| `--lambda_reg` / `--g_C_lr` | 1e-3 / 5e-5 | 锚定权 / 慢环 lr |
| `--target_drift` | 0.05 | COB 漂移目标值，单位 m |
| `--drift_start_step` / `--drift_end_step` | 200 / 1200 | 漂移窗（step） |
| `--drift_axes` | 0 (X) | 漂移轴下标 |
| `--ramp_shape` | linear | linear / cosine |
| `--workflow_config` | None | yaml 注入入口 |
| `--experiment_name` | easyuuv_direct | **必须显式给 `easyuuv_parametric` 才能加载 8 维 ckpt** |

### 3.3 "STDW 生效前后"如何写

| 配置 | 含义 | 旋钮 |
|---|---|---|
| **A. wrapper-only baseline** | wrapper 每步加 drift，但慢环不更新；表现"漂移期 MSE 单调上升"的对照 | `--use_stdw False --target_drift 0.05` |
| **B. STDW off（最干净基线）** | drift 也不加；纯环境基线 | `--use_stdw False --target_drift 0` |
| **C. STDW on（推荐）** | drift + 慢环 + Lyapunov mask + 滤波 | `--use_stdw True --enable_filter True --target_drift 0.05` |

写"STDW 生效前后"建议用 **B vs C**（最干净的"开关"对照）。
如果想突出"漂移环境下 STDW 的额外收益"，用 **A vs C**。

### 3.4 summary.json 关键字段
`final_mse`（m²）、`convergence_step`、`stability_threshold_*`、
`baseline_compound_error_mean`、`slow_loop_triggers`、`reset_count`、`final_mse_after_drift`（m²）、
`use_stdw`、`enable_filter`、`embodiment`、`scenario`、加 `**tracking_mse_summary`（详见
`workflows/play_stdw_adapt.py` 末尾段）。

---

## 4. Cross-Embodiment

### 4.1 5 个机型在 env 里都有

`easyuuv_env.py` 的 `embodiment_configs`：

| 名 | mass | inertia | dyn_τ | drag |
|---|---|---|---|---|
| `base` | 22.7 kg | [0.37, 0.97, 1.19] | 0.05 | 1.0 |
| `long_body` | 22.7 kg | [0.1, 2.5, 2.5] | 0.05 | 1.0 |
| `heavy_duty` | × 5 | × 5 | 0.20 | × 5 |
| `heavy_moderate` | × 2 | × 2 | 0.10 | × 2 |
| `asymmetric` | 22.7 kg | [0.37, 0.97, 1.19] | 0.05 | 1.0（com_to_cob xy 偏移 0.05/0.05 m） |

### 4.2 切换接口
- env 层：`EasyUUVEnv.apply_embodiment_config(name)`
- CLI：
  - `play_stdw_adapt.py --embodiment {base, long_body, heavy_moderate, asymmetric}`（**漏 `heavy_duty`**）
  - `play_meta_eval.py --embodiment` **无 choices 限制**，按 env 字典查（5 个都能用）
- sweep 驱动：`workflows/sweep_72cell.py` 同样漏 `heavy_duty` 的 4 项

### 4.3 已知坑
`play_stdw_adapt.py` 的 choices 列表与 env 字典脱钩；如果未来要扫到 `heavy_duty`，
必须改 choices 或者改 worker 用 `play_meta_eval.py`。

---

## 5. 任务 / 实验名映射

| 任务 ID | 注册 | 维度 | experiment_name | ckpt 默认目录 |
|---|---|---|---|---|
| `EasyUUV-Direct-v1` | [`__init__.py`](../../__init__.py) | 4 维 a_ctrl | `easyuuv_direct` | `~/isaaclab/logs/rsl_rl/easyuuv_direct/...` |
| `EasyUUV-Direct-Parametric-v1` | 同上 | 8 维 a_ctrl + a_gain | `easyuuv_parametric` | 训练 run 落 `.tmp/.../easyuuv_parametric/...` |

跨任务用 ckpt 会立刻报 actor MLP 维度不匹配（4 vs 8），见 [`ERROR_CASES.md`](ERROR_CASES.md#case-2-actor-mlp-维度不匹配)。

---

## 6. Sweep 驱动总览

| 脚本 | 矩阵 | 任务 | yaml 输入 |
|---|---|---|---|
| `workflows/sweep_stdw.py` | scenario × embodiment × algo flags | `EasyUUV-Direct-v1` | 通过 `play_stdw_adapt.py --workflow_config` 间接注入 |
| `workflows/sweep_72cell.py` | 3 axis × 3 magnitude × 4 emb × 2 flag = 72 | `EasyUUV-Direct-Parametric-v1` | 内部硬编码 |
| `workflows/sweep_full_matrix.py` | 3 wave × 4 emb × 2 tune × 2 stdw × 1 seed = 48 | 同上 | yaml 注入 |
| `workflows/run_4grp_compare.sh` | 4 组 STDW 算法对照 | `EasyUUV-Direct-v1` | – |
| `workflows/tools/aggregate_sweep72.py` | 72-cell 聚合器 | – | – |
| `workflows/tools/aggregate_full_matrix.py` | 48-cell 聚合器 + STDW 配对 | – | – |

---

## 7. 极简训练日志

`workflows/train_meta.py` 的 `_attach_compact_logger` 会在每个 run 的 log_dir 根下写两份 JSONL：

- `compact_log.jsonl`：每行 `{iter, reward, ep_len, vloss, ploss, timesteps, noise_std}`
- `mse_curve.jsonl`：每行 `{iter, log_mse, n}`，n 是该 iter 内参与平均的 episode 数

不需要在 train.log 上做 regex 解析。

---

## 8. 修改这份卡的纪律

- 这份卡描述的是**接口能力**与**已知坑**，不写实验结论。结论写在 `RESEARCH_ANALYSIS.md` 与 `_legacy/REPORT_*.md`。
- 改了 `disturbance_cfg` / `apply_embodiment_config` / `play_stdw_adapt.py` 的 CLI / `gain_tuner.py` 的机制开关，
  **必须同步本卡 + [`COMMAND_CONTRACT.md`](COMMAND_CONTRACT.md)**。

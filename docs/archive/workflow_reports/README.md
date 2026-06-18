# workflows_new_stdw — STDW v3 渐进域自适应工作流

本目录是 [STDW模块化重构与渐进域自适应实施计划.md](../.trae/documents/STDW模块化重构与渐进域自适应实施计划.md) v3 落地的入口工作区。所有脚本与论文 *Stepwise Dynamic Osmosis*（NeurIPS 2025）相同的渐进域自适应思路对齐：在线对运行中的 WarpAUV PPO 策略做 STDW 风格的 fine-tune（不是蒸馏），并加入 Lyapunov 单调性 mask 作为物理滤网。

> **跑大矩阵前必读**：[INTERFACE_REFERENCE.md](INTERFACE_REFERENCE.md) 列出了 JONSWAP / 整定 / STDW / Embodiment 的真实 CLI / yaml 接口能力与已知坑（无 CLI 的子参数、STDW 没有干净的 wrapper-bypass 总开关、4 vs 5 个 embodiment 列表脱钩等）。

> 适用任务：`MOGA-WarpAUV-Direct-v1`（在 [__init__.py](../__init__.py) 注册）。
> PPO 实验名：`warpauv_direct`（见 [agents/rsl_rl_ppo_cfg.py](../agents/rsl_rl_ppo_cfg.py)）。
> Checkpoint：默认 `~/isaaclab/logs/rsl_rl/warpauv_direct/<load_run>/<checkpoint>`。

---

## 1. 仓库速览（仅与本工作流相关）

| 路径 | 角色 |
|---|---|
| [warpauv_env.py](../warpauv_env.py) | Gym 注册指向的基线环境；line 893-897 处会缓存底层 PID 自适应增量 `PID_value_add` 到 `_pid_value_add_buf`，供伪标签使用 |
| [warpauv_stdw_wrapper.py](../warpauv_stdw_wrapper.py) | `gym.Wrapper` 子类。负责：① 线性 com↔cob drift 调度（`drift_start_step`/`drift_end_step`/`target_drift`/`drift_axes`）；② 5s 滑窗 RMS 误差滤波；③ Lyapunov `V=½ eᵀ P e` 与 ΔV mask 计算 |
| [utils/stdw_buffer.py](../utils/stdw_buffer.py) | 环形 Replay Buffer，10 张主表（含 `pseudo_actions / stdw_masks / lyapunov_V / domain_tags`），提供 `sample / sample_pair / save / load` |
| [workflows_new_stdw/play_stdw_adapt.py](play_stdw_adapt.py) | STDW v3 在线微调主入口：快环 dt=1/120s 采样 + 慢环每 60 步按 mask 加权 MSE + L2 锚定 |
| [workflows_new_stdw/run_stdw_smoke.py](run_stdw_smoke.py) | 20 步冒烟跑（默认 `--cpu --headless`），用于验证 CSV/PNG/buffer.pt/summary.json 写盘链路 |
| [workflows_new_stdw/sweep_stdw.py](sweep_stdw.py) | 72 组扫参驱动；默认 `--limit_combinations 4` 用于快速验证 |
| [workflows_new_stdw/run_4grp_compare.sh](run_4grp_compare.sh) | 本目录提供的 4 组对照 launcher（plan §5 最小集），串行 1400 step |
| [custom_workflows/run_with_isaac_env.sh](../custom_workflows/run_with_isaac_env.sh) | bash launcher，激活 `isaaclab` conda 环境并 `source setup_python_env.sh` |
| [custom_workflows/experiment_runner.py](../custom_workflows/experiment_runner.py) | 通用扫参/任务调度，`SCRIPT_BY_KIND["stdw"]` 已指向本目录的 `play_stdw_adapt.py` |
| [workflows/stdw_integration/plots.py](../workflows/stdw_integration/plots.py) | 旧 / 新两个工作流共享的 7 张诊断图函数 |

### 1.1 物理修正动作（伪标签）

慢环训练时不能直接用 buffer 中的"漂移期错误动作"作监督，而要用：

```
a_pseudo[t] = a_executed[t] + pseudo_gain · (J_inv_diag · Δu(t))
```

其中 `Δu(t) = env.unwrapped._pid_value_add_buf`（4 通道：roll/pitch/yaw/depth），`J_inv_diag = ones(4)`（v3 简化）。该路径在 [play_stdw_adapt.py](play_stdw_adapt.py) §3.4.2.5 实现。

### 1.2 梯度路径

不能用 `policy.act_inference`（被 `@torch.no_grad()` 装饰），慢环走 `policy.actor(obs)` 子模块原始前向，确保 `requires_grad=True`。helpers：`_policy_forward_eval / _policy_forward_train`。

### 1.3 Lyapunov 物理滤网

每步计算 `V[k] = ½ · sum(P_diag · e²)`，仅当 `ΔV = V[k] − V[k−1] < lyapunov_eps` 时 `mask=1`，其余 `mask=0`。冷启动首步 mask=0。慢环按样本加权 MSE。

---

## 2. 运行前置条件

1. conda 环境 `isaaclab` 已配置，参考 [run_with_isaac_env.sh](../custom_workflows/run_with_isaac_env.sh)。
2. 已存在 PPO checkpoint，路径：

   ```
   ~/isaaclab/logs/rsl_rl/warpauv_direct/SS4/model_500.pt
   ```

3. 默认 shell 已切到 bash（zsh 在 `setup_python_env.sh` 上 BASH_SOURCE 不兼容）。

---

## 3. 推荐运行命令

### 3.1 冒烟跑（20 步，CPU，headless）

```bash
bash custom_workflows/run_with_isaac_env.sh \
  workflows_new_stdw/run_stdw_smoke.py
```

期望产物：`results_root` 下 `summary.json` + `buffer.pt` + 7 张 PNG + `output.csv`，`lyapunov_V` 与 `stdw_mask` 列非空。

### 3.2 单次完整跑（1400 步）

```bash
bash custom_workflows/run_with_isaac_env.sh \
  workflows_new_stdw/play_stdw_adapt.py \
  --headless --cpu \
  --task MOGA-WarpAUV-Direct-v1 \
  --num_envs 1 \
  --load_run SS4 \
  --checkpoint model_500.pt \
  --total_steps 1400 \
  --use_stdw True --enable_filter True --use_quantile_filter True \
  --target_drift 0.05 \
  --drift_start_step 200 --drift_end_step 1200 \
  --slow_loop_interval 60 --batch_size 256 \
  --g_C_lr 5e-5 --lambda_reg 1e-3
```

### 3.3 4 组对照（plan §5 最小集，本目录 launcher）

```bash
bash workflows_new_stdw/run_4grp_compare.sh
```

落盘：`<repo>/.tmp/stdw_4grp_<YYYYMMDD_HHMMSS>/{baseline,stdw_only,stdw_filter,stdw_full}/`

| 组别 | use_stdw | enable_filter | use_quantile_filter | 期望观察 |
|---|---|---|---|---|
| baseline | False | False | False | 漂移期 MSE 单调上升的对照基线 |
| stdw_only | True | False | False | 仅渐进域 + Lyapunov mask 的纯 STDW |
| stdw_filter | True | True | False | + 5s RMS 滤波抑制非平稳噪声 |
| stdw_full | True | True | True | + quantile 置信度过滤 |

其余参数走 [play_stdw_adapt.py](play_stdw_adapt.py) 默认（`drift_start=200, drift_end=1200, slow_loop_interval=60, batch_size=256, g_C_lr=5e-5, target_drift=0.05, drift_axes=0`，Lyapunov 默认开）。

### 3.4 sweep dry-run（4 组裁剪）

```bash
python workflows_new_stdw/sweep_stdw.py --dry_run
```

### 3.5 完整扫参（耗时长，谨慎使用）

`sweep_stdw.py` 自 2026-06-04 起支持两套不同的扫参矩阵：

| 矩阵 | 触发开关 | 默认含义 | 组数 |
|---|---|---|---|
| `DEFAULT_MATRIX` | `--full_grid` / `--full_matrix` | 8 个 scenario × 4 个 embodiment（new 渐进注入扩展） | 32 |
| `DEFAULT_MATRIX` 子集 | `--scenarios_only` | 8 个 scenario × `embodiment=base` | 8 |
| `DEFAULT_MATRIX` 子集 | `--embodiments_only` | 4 个 embodiment × `scenario=none` | 4 |
| `ALGO_MATRIX` | `--algo_grid` | 旧 sweep8/72 组算法网格（`use_stdw / enable_filter / use_quantile_filter / g_C_lr / target_drift` 笛卡尔积） | 72 |

> ⚠️ **回归提醒**：之前 `--full_matrix` 等价于 sweep72 算法网格。当前默认矩阵已切换为"场景 × 机型"，若需复现旧 sweep8 / sweep72 的算法网格，**必须显式加 `--algo_grid`**，否则会跑成新 28+ 组场景矩阵。建议在 [run_4grp_compare.sh](run_4grp_compare.sh) 与任何回归脚本里同时显式写出 `--algo_grid` 与 `--full_grid`，并在 CSV 输出路径里加上 `_algo` / `_grid` 后缀以避免覆盖。

```bash
# A. 新场景 × 机型 28+ 组（推荐：6000 步，验证 fault / cosine ramp / 跨机型）
python workflows_new_stdw/sweep_stdw.py --full_grid \
  --total_steps 6000 \
  --base_logs_root .tmp/stdw_full_grid_6k \
  --csv_out logs/stdw_full_grid_6k.csv

# B. 旧算法网格 72 组（sweep72 复现）
python workflows_new_stdw/sweep_stdw.py --algo_grid --full_matrix \
  --total_steps 1400 \
  --base_logs_root .tmp/stdw_algo_grid \
  --csv_out logs/stdw_algo_grid.csv
```

新矩阵一律会把 `scenario` / `embodiment` / `final_mse_after_drift` 三列写入聚合 CSV。`--algo_grid` 仍然用旧字段，保持 sweep8 / sweep72 报告兼容。

#### 进一步暴露的 CLI（2026-06-04）

| Flag | 含义 |
|---|---|
| `--ramp_shape {linear,cosine}` | 控制 drift / disturbance ramp 形状；cosine 用于 sine 等振荡场景，零斜率端点更平滑 |
| `--stability_threshold` | 收敛绝对阈值（默认 0.05） |
| `--stability_threshold_rel` | 相对阈值倍率，runtime 取 `max(abs, rel × baseline_compound_error_mean)`，0 关闭 |
| `--stability_window` | 收敛连续步数 |

---

## 4. 关键产物

每个 run 在其 `results_root/stdw_new_<timestamp>/` 下生成：

- `output.csv`：含 `step, mse_total, rho, drift_frac, action_*, lyapunov_V, stdw_mask, effective_batch_frac, ...`
- `summary.json`：`final_mse / convergence_step / config snapshot`
- `buffer.pt`：完整 buffer 序列化
- 7 张 PNG（由 [plots.py](../workflows/stdw_integration/plots.py) 生成）

`run_4grp_compare.sh` 额外写出 `<base>/index.csv`（列：`name,use_stdw,enable_filter,use_quantile_filter,returncode,started_at,ended_at`）。

---

## 5. 排错速查

- **未找到 checkpoint** → 检查 `--load_run` 与 `--checkpoint`；默认查找 `~/isaaclab/logs/rsl_rl/warpauv_direct/<load_run>/<checkpoint>`。
- **慢环报 `requires_grad=False`** → 说明 `_policy_forward_train` 走错路径或 `policy.actor` 还没 unfreeze；查 [play_stdw_adapt.py](play_stdw_adapt.py) helpers。
- **`_pid_value_add_buf` 缺失** → 确认 [warpauv_env.py L893-897](../warpauv_env.py#L893-L897) 那段 hook 没被回滚。
- **zsh 在 source 时报 BASH_SOURCE** → 切到 bash（这是本工作区强制要求）。

---

## 6. 文档与计划

- 实施计划（已批准 v3）：[STDW模块化重构与渐进域自适应实施计划.md](../.trae/documents/STDW模块化重构与渐进域自适应实施计划.md)
- 论文摘录：[stdw_paper.md](../stdw_paper.md)
- 工作区总规则：[AGENTS.md](../AGENTS.md)

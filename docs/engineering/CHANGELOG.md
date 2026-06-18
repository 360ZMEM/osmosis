# Changelog

本仓库不使用 git，但仍按时间倒序记录实质变更。日期 ISO 8601 (UTC+8)。

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

# EasyUUV-STDW

Slow-Time Domain Wrapper (STDW) 慢环自适应控制 + JONSWAP 海况扰动 + 8 维 meta-control
参数整定的独立研究仓库。基于 Isaac Lab Direct RL，从 `MOGA-WarpAUV-Direct-v1` 演进而来。

> **新成员从这里开始** → [`docs/INDEX.md`](docs/INDEX.md) → [`docs/guide/AGENT_HANDOFF.md`](docs/guide/AGENT_HANDOFF.md)

---

## 已注册的 Gym 任务

| Task ID | 维度 | experiment_name |
|---|---|---|
| `EasyUUV-Direct-v1` | 4D ctrl | `easyuuv_direct` |
| `EasyUUV-Direct-Parametric-v1` | 8D (ctrl + a_gain) | `easyuuv_parametric` |

## 命令契约（5 条，详见 [`docs/guide/COMMAND_CONTRACT.md`](docs/guide/COMMAND_CONTRACT.md)）

```bash
# C1. 训练（8D parametric，A3 stage2 baseline；两阶段完整命令见 COMMAND_CONTRACT）
bash custom_workflows/run_with_isaac_env.sh workflows/train_meta.py \
    --task EasyUUV-Direct-Parametric-v1 --num_envs 512 \
    --max_iterations 1200 --meta_stage 1 --headless --logger tensorboard

# C3. 慢环评估（STDW on）
bash custom_workflows/run_with_isaac_env.sh workflows/play_stdw_adapt.py \
    --task EasyUUV-Direct-Parametric-v1 \
    --experiment_name easyuuv_parametric --num_envs 1 --headless \
    --workflow_config workflows/configs/wave_storm.yaml --wave_mode jonswap \
    --use_stdw True --target_drift 0.05 \
    --load_run 2026-06-08_13-48-14_stage2 --checkpoint model_2398.pt

# C4. 全矩阵评估（48 cell ≈ 23 min）
bash custom_workflows/run_with_isaac_env.sh workflows/sweep_full_matrix.py \
    --policy_path logs/rsl_rl/easyuuv_parametric/2026-06-08_13-48-14_stage2/model_2398.pt \
    --out_root .results/sweep_a3_stage2_$(date +%Y%m%d_%H%M%S) --seeds 0

# C5. 聚合
bash custom_workflows/run_with_isaac_env.sh workflows/tools/aggregate_full_matrix.py \
    --matrix_dir .results/sweep_a3_stage2_<timestamp>

# Eval（不依赖 Isaac）
python -m easyuuv_stdw.eval.deploy_eval --policy ckpt.jit --replay log.csv \
    --output ./.results/eval_out.csv
```

**STDW 双关基线**（写报告必用）：
- clean off：`--use_stdw False --target_drift 0`
- on：`--use_stdw True --target_drift 0.05`

## 文档导航

docs 按 **理解原理 / 工程实践 / 用户指南 / 归档** 四区组织，详见 [`docs/INDEX.md`](docs/INDEX.md)。

| 入口 | 用途 |
|---|---|
| [`docs/INDEX.md`](docs/INDEX.md) | 总入口、四区导航、单位速查表 |
| [`docs/guide/INTERFACE.md`](docs/guide/INTERFACE.md) | JONSWAP / 整定 / STDW / Embodiment 接口能力卡 |
| [`docs/guide/COMMAND_CONTRACT.md`](docs/guide/COMMAND_CONTRACT.md) | 5 条命令契约 + flag 兼容性 |
| [`docs/guide/EVAL_SOP.md`](docs/guide/EVAL_SOP.md) | Isaac-independent 部署评估 SOP |
| [`docs/guide/ERROR_CASES.md`](docs/guide/ERROR_CASES.md) | 常见调试坑（必读） |
| [`docs/guide/AGENT_HANDOFF.md`](docs/guide/AGENT_HANDOFF.md) | **AI 续接指令** |
| [`docs/principles/ARCHITECTURE.md`](docs/principles/ARCHITECTURE.md) | 模块图 + 数据流 |
| [`docs/principles/RESEARCH_ANALYSIS.md`](docs/principles/RESEARCH_ANALYSIS.md) | 实证结论（含单位） + 改进路线 |
| [`docs/principles/PAPER_STDW_CN.md`](docs/principles/PAPER_STDW_CN.md) | 论文中文主稿 |
| [`docs/engineering/REPORT_full_matrix_20260608_a3_stage2.md`](docs/engineering/REPORT_full_matrix_20260608_a3_stage2.md) | **主线** 48-cell 全矩阵报告 |
| [`docs/engineering/CHANGELOG.md`](docs/engineering/CHANGELOG.md) | 变更记录 |

## 最重要的 5 件事

1. **JONSWAP 没有 CLI**，必须 yaml 注入 `env.disturbance_cfg.jonswap_*`。
2. **STDW off 必须双关** `--use_stdw False --target_drift 0`，否则 wrapper 仍推 COB drift。
3. **4D vs 8D ckpt 不通用**：`--task` 与 `--experiment_name` 必须同时匹配。
4. **STDW 不是无差别加速器**：`asymmetric`（浮心-质心 xy 偏移）上 STDW on 反向劣化最高 +158%，部署前先短跑判定 gating。
5. **eval/ 不依赖 Isaac**，可在板载 PC / CI / ONNX Runtime 上跑。

## 数据制品

- **主线**性能报告（48-cell A3 stage2 全矩阵）：[`docs/engineering/REPORT_full_matrix_20260608_a3_stage2.md`](docs/engineering/REPORT_full_matrix_20260608_a3_stage2.md)
- 最新动态参数辨识：[`docs/engineering/REPORT_dynamic_parameter_identification_20260613.md`](docs/engineering/REPORT_dynamic_parameter_identification_20260613.md)
- 论文图 fig1–9：[`docs/figures/`](docs/figures/)（绘图说明见 [`docs/engineering/REPORT_stdw_effect_figures_20260608.md`](docs/engineering/REPORT_stdw_effect_figures_20260608.md)）
- 历史报告（旧 1500-iter ckpt）：[`docs/archive/REPORT_full_matrix_20260606.md`](docs/archive/REPORT_full_matrix_20260606.md)
- 训练 metric：每个 run 目录下 `compact_log.jsonl` + `mse_curve.jsonl`

## 关键性能数据（A3 stage2 baseline，model_2398，含单位）

STDW 主效应按 embodiment 分层（mean over wave × tune，单位 m²）：

| Embodiment | fmse_off | fmse_on | Δ_STDW | 状态 |
|---|---:|---:|---:|:---:|
| base           | 0.2254 | 0.0726 | **−67.8%** | OK |
| long_body      | 0.2063 | 0.0719 | **−65.2%** | OK |
| heavy_moderate | 0.2664 | 0.2807 | +5.3% | 中性偏负 |
| asymmetric     | 0.2263 | 0.5847 | **+158%** | **异常（需 gating）** |

> Δ_STDW 在 base+long_body 上跨 calm/medium/storm 三档 wave 极其稳定（−66.5 ± 0.05%）。
> 相比旧 1500-iter ckpt（−38.6%），A3 stage2 把主效应改善到约 1.7×，并首次让 full tune 净增益 −8.8%。

## 安装（简版）

需求：Isaac Lab ≥ 2024.1（仅训练/sweep 路径需要）；eval 路径仅需 `numpy + torch`。

```bash
# 在 Isaac Lab 的 lab_tasks 扩展下挂载本目录（已在原仓库挂载）
# 然后：
pip install -e .   # 如果需要 standalone（可选）
```

详细安装见 Isaac Lab 官方文档。本仓库不重复其安装步骤。

## 不要做的事

1. 不要 rename `WARPAUV_CFG` 与 `data/warpauv/`（USD 资产硬引用）。
2. 不要 git init —— 暂未启用 git。
3. 不要把新结论写到 `docs/archive/` 里 —— 一律写到 `docs/principles/RESEARCH_ANALYSIS.md` 或 `docs/engineering/REPORT_*.md`。

## 许可

继承自 Isaac Lab WarpAUV / EasyUUV upstream license。

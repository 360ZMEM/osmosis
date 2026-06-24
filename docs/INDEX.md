# EasyUUV-STDW Documentation Index

EasyUUV-STDW 是从 Isaac Lab Direct RL 任务 `MOGA-WarpAUV-Direct-v1` 演进出来的独立仓库，
聚焦于 **Slow-Time Domain Wrapper (STDW)** 慢环自适应控制研究 + JONSWAP 海况扰动 + 8 维 meta-control
参数整定。本仓库不依赖外部「Easy*」品牌名，所有公开类/Gym ID 已统一为 EasyUUV* 命名。

---

## 文档分区

docs 按用途分为四个目录：

| 目录 | 定位 | 何时看 |
|---|---|---|
| [`principles/`](principles/) | **理解原理** — 系统模型、方法论、论文、科研分析 | 想搞懂 STDW 为什么这样设计、要写论文 |
| [`engineering/`](engineering/) | **工程实践** — 实验报告、诊断记录、变更日志 | 想知道某次实验结论、调参与回归历史 |
| [`guide/`](guide/) | **用户指南** — 命令契约、接口卡、评估/部署 SOP、续接包 | 想跑命令、对接接口、续接会话 |
| [`archive/`](archive/) | **归档** — 已过时但保留的历史文档 | 仅作历史溯源，结论以上面三区为准 |

实验图片统一放在 [`figures/`](figures/)（`fig1`–`fig9` STDW 效果主线图）。

---

## 理解原理 · principles/

| # | 文档 | 用途 |
|---|---|---|
| 1 | [`ARCHITECTURE.md`](principles/ARCHITECTURE.md) | 模块图 + 数据流 + 文件职责 |
| 2 | [`RESEARCH_ANALYSIS.md`](principles/RESEARCH_ANALYSIS.md) | 科研分析 + 改进建议（含主线实验结论与单位） |
| 3 | [`PAPER_STDW_CN.md`](principles/PAPER_STDW_CN.md) | **论文（中文）** — 学术汇报主稿 |
| 4 | [`PAPER_STDW_EN.md`](principles/PAPER_STDW_EN.md) | **论文（英文）** — 学术汇报主稿 |
| 5 | [`PAPER_ADDENDUM_dynamic_identification_20260613.md`](principles/PAPER_ADDENDUM_dynamic_identification_20260613.md) | 论文附录 — 动态参数辨识 |
| 6 | [`PAPER_ADDENDUM_followup_validation_20260613.md`](principles/PAPER_ADDENDUM_followup_validation_20260613.md) | 论文附录 — router/probe 追加验证 |

## 工程实践 · engineering/

| # | 文档 | 用途 |
|---|---|---|
| 1 | [`REPORT_full_matrix_20260608_a3_stage2.md`](engineering/REPORT_full_matrix_20260608_a3_stage2.md) | **主线** 48-cell A3 stage2 全矩阵（STDW −67.8%） |
| 2 | [`REPORT_stdw_effect_figures_20260608.md`](engineering/REPORT_stdw_effect_figures_20260608.md) | 论文图 fig1–9 绘图说明与叙事顺序 |
| 3 | [`REPORT_noise_ablation_20260613.md`](engineering/REPORT_noise_ablation_20260613.md) | 26-cell 噪声/延迟鲁棒性消融 |
| 4 | [`REPORT_stdw_strong_validation_20260613.md`](engineering/REPORT_stdw_strong_validation_20260613.md) | 32-cell 强扰动验证 |
| 5 | [`REPORT_stdw_strong_ablation_20260613.md`](engineering/REPORT_stdw_strong_ablation_20260613.md) | 25-cell 默认消融 |
| 6 | [`REPORT_stdw_followup_validation_20260613.md`](engineering/REPORT_stdw_followup_validation_20260613.md) | router/probe followup 验证 |
| 7 | [`REPORT_dynamic_parameter_identification_20260613.md`](engineering/REPORT_dynamic_parameter_identification_20260613.md) | **最新** 动态参数辨识（active_v3） |
| 8 | [`DIAG_p1_p2_p5_20260610.md`](engineering/DIAG_p1_p2_p5_20260610.md) | P1/P2/P5 诊断记录 |
| 9 | [`控制器稳定调节记录.md`](engineering/控制器稳定调节记录.md) | 参考轨迹 + 阻尼奖励调优记录 |
| 10 | [`CHANGELOG.md`](engineering/CHANGELOG.md) | 版本变更记录 |

## 用户指南 · guide/

| # | 文档 | 用途 |
|---|---|---|
| 1 | [`COMMAND_CONTRACT.md`](guide/COMMAND_CONTRACT.md) | 5 条命令契约（train/play/sweep/eval/aggregate）+ flag 兼容性 |
| 2 | [`INTERFACE.md`](guide/INTERFACE.md) | JONSWAP / 整定 / STDW / Embodiment 全部接口能力卡 |
| 3 | [`EVAL_SOP.md`](guide/EVAL_SOP.md) | Isaac-independent 部署评估 SOP |
| 4 | [`DEPLOY_SOP_realworld.md`](guide/DEPLOY_SOP_realworld.md) | 实物部署 SOP |
| 5 | [`CLEAN_ENV_STDW_FOSSEN_DEMO.md`](guide/CLEAN_ENV_STDW_FOSSEN_DEMO.md) | 干净 Python 环境 + Fossen 6-DOF 虚拟实物的最小部署链路教程 |
| 6 | [`ERROR_CASES.md`](guide/ERROR_CASES.md) | 常见调试坑（yaml 注入失效、CUDA / dim mismatch 等） |
| 7 | [`AGENT_HANDOFF.md`](guide/AGENT_HANDOFF.md) | **AI 续接指令** — 新 agent 接手时的最短重启包 |

---

## Subpackage entry points

- 训练 / 慢环工作流 → [`workflows/`](../workflows/)
- 评估子模块（不依赖 Isaac）→ [`eval/`](../eval/) 与 [`eval/README.md`](../eval/README.md)
- Gym 注册 → [`__init__.py`](../__init__.py)
- 物理资产 → [`assets/`](../assets/) + [`data/`](../data/)
- STDW 集成层 → [`stdw_integration/`](../stdw_integration/)

## Quantities & units (cheat sheet)

| 量 | 单位 | 出现位置 |
|---|---|---|
| `final_mse` / `final_mse_after_drift` | m²（位置误差平方均值） | summary.json、`REPORT_*.md` |
| `rmse_pos_m` | m | `eval/deploy_eval.py` 输出 |
| `target_drift` | m（COB 漂移目标值） | `play_stdw_adapt.py --target_drift` |
| `slow_loop_interval` | step（快环 dt=1/120s） | 同上 |
| `pe_freq` | Hz | `gain_tuner.py` |
| `gain_beta` | 无量纲（ζ_eff = ζ_nom·(1+β·a)） | 同上 |
| `jonswap_hs` | m（significant wave height） | `wave_disturbance_manager.py` |
| `jonswap_fp` | Hz（peak frequency） | 同上 |
| `jonswap_depth` | m | 同上 |
| `jonswap_direction` | rad | 同上 |
| reward shaping | 无量纲（−|pos_err|² − 0.5·|yaw_err| − 0.1·|a[:4]|² − 0.05·|ω_b|²） | `eval/wrappers.py` |

## 归档 · archive/（不再权威）

仅作为历史参考，结论以 `principles/RESEARCH_ANALYSIS.md` + 最新 `engineering/REPORT_*.md` 为准：

- [`REPORT_full_matrix_20260606.md`](archive/REPORT_full_matrix_20260606.md)
- [`REPORT_full_train_72cell_stdw_20260605.md`](archive/REPORT_full_train_72cell_stdw_20260605.md)
- [`INTERFACE_REFERENCE_legacy.md`](archive/INTERFACE_REFERENCE_legacy.md)
- [`FULL_MATRIX_RUNBOOK_legacy.md`](archive/FULL_MATRIX_RUNBOOK_legacy.md)
- [`AGENTS_legacy.md`](archive/AGENTS_legacy.md)
- [`workflow_reports/`](archive/workflow_reports/) — 更早期的工作流报告集

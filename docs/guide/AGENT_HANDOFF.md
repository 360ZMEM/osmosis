# AGENT_HANDOFF — AI 续接指令

> 本文是为**下一位 AI agent / 工程师**准备的"最短重启包"，避免重复探索。
> 任何在 EasyUUV-STDW 仓库内开新会话的 AI，应该首先读完本文，再读 [`INDEX.md`](../INDEX.md)。

---

## 0. 你接手的是什么

EasyUUV-STDW 是从 Isaac Lab Direct RL 任务 `MOGA-WarpAUV-Direct-v1` 重构出来的独立仓库，
聚焦 **Slow-Time Domain Wrapper (STDW) 慢环自适应控制 + JONSWAP 海况 + 8 维 meta-control 整定**。
对外命名一律 EasyUUV*；对内（`WARPAUV_CFG`、`data/warpauv/`）保留原名以避免 USD 资产断链。

**最重要的事实**：
1. 已注册的 Gym 任务是 `EasyUUV-Direct-v1`（4D） 与 `EasyUUV-Direct-Parametric-v1`（8D）。
   见 [`__init__.py`](../../__init__.py)。
2. STDW 双关基线写法：clean off = `--use_stdw False --target_drift 0`；
   on = `--use_stdw True --target_drift 0.05`。
3. JONSWAP 海况只能通过 yaml 注入 `env.disturbance_cfg.jonswap_*`，**没有 CLI**。
4. 当前主线是 A3 stage2 baseline：stage1 1200 iter 训练控制头，stage2 1200 iter
   从 stage1 ckpt 续训增益头；主线 ckpt 是
   `logs/rsl_rl/easyuuv_parametric/2026-06-08_13-48-14_stage2/model_2398.pt`。
5. A3 stage2 后 `tune=full` 首次进入有效区：STDW on 时 full 比 identity 净改善约 8.8%；
   旧 1500-iter `model_1499.pt` 结论只作历史对照。
6. STDW 不是无差别加速器：A3 stage2 在 `base`/`long_body` 上稳定改善约 65%-68%，
   `heavy_moderate` 已从旧 +75% 异常收敛到约 +5.3%（中性偏负），但 `asymmetric`
   仍会被朴素 drift 显著劣化（最高约 +158%，需要 gating/probe/routing）。
7. **第二期改良已落地（2026-06-07）**：TAG 在线自适应门限（`play_stdw_adapt.py`
   `--enable_trigger_gate/--trigger_threshold`）+ 分阶段训练（`train_meta.py`
   `--meta_stage 1/2`、`--stage1_checkpoint`、`--stage2_cob_offset_xyz`、
   `--stage2_wave_mode`）+ plots 第 7 张图 `stdw_tracking_overlay.png` + TAG 灰色阴影。
8. **最新主线 8D checkpoint（阶段一 + 阶段二）**：
   - 阶段一：`logs/rsl_rl/easyuuv_parametric/2026-06-08_13-04-27_stage1/model_1199.pt`
     （A3 baseline，1200 iter，控制头训练，观测为 12D）。
   - 阶段二：`logs/rsl_rl/easyuuv_parametric/2026-06-08_13-48-14_stage2/model_2398.pt`
     （从 stage1 续训 1200 iter，控制头+主干冻结、只训增益头，COM-COB 漂移 + JONSWAP）。
   - 评估/play 用 `--experiment_name easyuuv_parametric --load_run <run> --checkpoint <model.pt>`。
9. **play 加载分阶段 ckpt 必须用 `load_optimizer=False`**：分阶段训练重建了只含可训练
   子集的 Adam，其 optimizer state 与全参数 optimizer param group 不匹配；play 下游本就
   自建 optimizer，无需 ckpt optimizer state。`play_stdw_adapt.py` 已固定为
   `ppo_runner.load(resume_path, load_optimizer=False)`。
10. **命令契约以 [`COMMAND_CONTRACT.md`](COMMAND_CONTRACT.md) 为准**；README 只保留摘要。
    本机依赖 Isaac Lab 的命令应优先用
    `bash custom_workflows/run_with_isaac_env.sh <script.py> ...` 启动。

---

## 1. 入口文件读取顺序

```
__init__.py                  Gym 注册 → 决定哪些任务可被发现
easyuuv_env.py               主 env，1280 行，6 个关键函数都加了 doxygen
easyuuv_stdw_wrapper.py      gym.Wrapper（COB drift + RMS + Lyapunov mask）
gain_tuner.py                ParametricGainTuner（开环 4 阶段）
agents/rsl_rl_ppo_cfg.py     baseline + parametric 两个 PPORunnerCfg
workflows/train_meta.py      训练入口
workflows/play_stdw_adapt.py 慢环评估入口
workflows/sweep_full_matrix.py  48 cell sweep 驱动
eval/                        Isaac-independent 评估子模块
docs/INDEX.md                文档总入口
docs/guide/INTERFACE.md      接口能力卡
docs/guide/COMMAND_CONTRACT.md  5 条命令契约
docs/guide/ERROR_CASES.md    调试坑
docs/principles/RESEARCH_ANALYSIS.md  实证结论 + 改进路线
docs/guide/EVAL_SOP.md       部署 SOP
```

---

## 2. 你最容易踩的 3 个坑（必读）

1. **JONSWAP yaml 注入失效** — 详见 [`ERROR_CASES.md`](ERROR_CASES.md) §1。
   `apply_runtime_domain_shift` 不覆盖 jonswap_*，必须通过 yaml 在 reset 之前覆盖。
2. **STDW off 仍跑 wrapper** — `--use_stdw False` 只 skip 慢环，wrapper 的 COB drift 仍推进。
   clean baseline 必须 `False + target_drift=0` 双关。
3. **4D vs 8D ckpt 不通用** — `--task` 与 `--experiment_name` 必须同时匹配。

---

## 3. 已完成的迁移工作（不要重做）

迁移自 `direct/isaac-auv-env-new/` 时已完成：

- [x] 拷贝 8 个核心 .py + agents/ + assets/ + data/warpauv/ + stdw_integration/ + workflows/
- [x] sed 重命名：`MOGA-WarpAUV-Direct-*-v1` → `EasyUUV-Direct-*-v1`，`WarpAUVEnv` → `EasyUUVEnv` 等
- [x] import 路径修正：`from warpauv_stdw_wrapper` → `from easyuuv_stdw.easyuuv_stdw_wrapper`
- [x] 6 个关键函数加 doxygen 注释（`_pre_physics_step` / `_apply_action` / `_compute_dynamics`
      / `get_current_fluid_velocity` / `_refresh_domain_randomization_defaults` / `_reset_idx`）
- [x] 写完 eval/ 6 个文件 + py_compile 通过
- [x] 写完 docs/ 9 个 md
- [x] 复制 5 份 _legacy md
- [x] 写 README.md / .gitignore / .gitkeep

**不要做的事**：不要 rename `WARPAUV_CFG` 常量、不要 rename `data/warpauv/` 目录。

---

## 4. 下次会话开局应该做的事

```bash
# 1. 验证 Gym 注册没坏
bash custom_workflows/run_with_isaac_env.sh -c "import gymnasium as gym; import easyuuv_stdw; \
    print([s for s in gym.registry if 'EasyUUV' in s])"

# 2. py_compile 全部
find . -name '*.py' -not -path './docs/archive/*' | xargs python -m py_compile && echo OK

# 3. 看最近实验结论
sed -n '1,80p' docs/principles/RESEARCH_ANALYSIS.md
```

如果用户的需求是**新一轮全矩阵评估**：
- 按 [`COMMAND_CONTRACT.md`](COMMAND_CONTRACT.md) §C4 跑 `workflows/sweep_full_matrix.py`。
- 评估前先用 1 个 cell smoke：calm/base/identity/off。

如果用户的需求是**新功能 / 改 STDW 行为**：
- 先读 [`ARCHITECTURE.md`](../principles/ARCHITECTURE.md) §3 的耦合点表。
- 改完 `disturbance_cfg` / `play_stdw_adapt.py` CLI / `gain_tuner.py` 必须同步
  [`INTERFACE.md`](INTERFACE.md) + [`COMMAND_CONTRACT.md`](COMMAND_CONTRACT.md)。

如果用户的需求是**部署 / Real-vehicle**：
- 看 [`EVAL_SOP.md`](EVAL_SOP.md)，全程不需要 Isaac。
- `eval/policy_loader.py` 已支持 .pt / .jit / .onnx 三种 backend。

---

## 5. 文档维护纪律

| 改了什么 | 必须同步更新 |
|---|---|
| `disturbance_cfg` / `wave_disturbance_manager.py` | INTERFACE.md §1, COMMAND_CONTRACT.md §C3 |
| `gain_tuner.py` 机制开关 | INTERFACE.md §2, ARCHITECTURE.md §3 |
| `play_stdw_adapt.py` CLI | INTERFACE.md §3, COMMAND_CONTRACT.md §C3, ERROR_CASES.md |
| Gym ID / experiment_name | INTERFACE.md §5, COMMAND_CONTRACT.md, README.md |
| `eval/wrappers.py` obs/reward | EVAL_SOP.md §1, eval/README.md |
| 新增实验报告 | _legacy/REPORT_<date>.md + RESEARCH_ANALYSIS.md §1 |

---

## 6. 不要做的事

1. 不要把 `_legacy/` 里的旧报告删掉 — 它们是论文复现的 anchor。
2. 不要给 `_legacy/` 里的旧文档加新结论 — 新结论一律写到 RESEARCH_ANALYSIS.md。
3. 不要把 README_mit.md / README_final.md（在源仓库）的安装步骤直接搬过来 —
   本仓库现在是独立体系，安装步骤已经在 README.md 重写。
4. 不要在 main env 里加 print()/log。极简日志走 `compact_log.jsonl` 与 `mse_curve.jsonl`。
5. 不要 git init —— 用户表示**还不需要 git init**。

---

## 7. 会话结束应做

完成你的工作后：
- 把变更同步到 [`CHANGELOG.md`](../engineering/CHANGELOG.md)。
- 如果新增实验，写 `_legacy/REPORT_<date>.md` + 在 RESEARCH_ANALYSIS.md §1 加一行。
- 如果改了任何 CLI / yaml schema，同步 [`COMMAND_CONTRACT.md`](COMMAND_CONTRACT.md) +
  [`INTERFACE.md`](INTERFACE.md)。
- 如果碰到新坑，写到 [`ERROR_CASES.md`](ERROR_CASES.md) Case 7+。

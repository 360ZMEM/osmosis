# Command Contract

EasyUUV-STDW 的全部对外命令收敛到 5 条契约。这份文档是契约源（source of truth），
README 与 RUNBOOK 都引用此处。当前主线基线是 A3 stage2：
`logs/rsl_rl/easyuuv_parametric/2026-06-08_13-48-14_stage2/model_2398.pt`。

> 所有命令的工作目录都是 **EasyUUV-STDW 仓库根**，即包含 `easyuuv_env.py` 的目录。
> 在本工作站上，依赖 Isaac Lab 的命令统一用 `bash custom_workflows/run_with_isaac_env.sh <script.py> ...`
> 启动；如果当前 shell 已经正确激活 Isaac Lab 环境，也可以把这一前缀替换为 `python`。

---

## C1. Train — A3 stage2 meta-control 训练（8 维）

当前可复现主线是两阶段训练：stage1 训练 8D policy 的控制头，stage2 从 stage1 ckpt 续训增益头。

```bash
# stage1: A3 baseline，产物示例 model_1199.pt
bash custom_workflows/run_with_isaac_env.sh workflows/train_meta.py \
    --task EasyUUV-Direct-Parametric-v1 \
    --num_envs 512 \
    --max_iterations 1200 \
    --meta_stage 1 \
    --headless \
    --logger tensorboard

# stage2: 从 stage1 ckpt 续训增益头，产物示例 model_2398.pt
bash custom_workflows/run_with_isaac_env.sh workflows/train_meta.py \
    --task EasyUUV-Direct-Parametric-v1 \
    --num_envs 512 \
    --max_iterations 1200 \
    --meta_stage 2 \
    --stage1_checkpoint logs/rsl_rl/easyuuv_parametric/<stage1_run>/model_1199.pt \
    --headless \
    --logger tensorboard
```

落盘：
- `logs/rsl_rl/easyuuv_parametric/<run>_stage1/model_*.pt`
- `logs/rsl_rl/easyuuv_parametric/<run>_stage2/model_*.pt`
- 同目录下 `compact_log.jsonl` + `mse_curve.jsonl`（极简日志）

当前报告主线 ckpt：
- stage1: `logs/rsl_rl/easyuuv_parametric/2026-06-08_13-04-27_stage1/model_1199.pt`
- stage2: `logs/rsl_rl/easyuuv_parametric/2026-06-08_13-48-14_stage2/model_2398.pt`

注意：历史 README 里出现过单条 `--max_iterations 2400` 训练摘要，但当前 A3 stage2 主线复现必须使用上面的
`--meta_stage 1/2` 两阶段命令；4 维基线见 C2。

## C2. Train — baseline（4 维）

```bash
bash custom_workflows/run_with_isaac_env.sh workflows/train_meta.py \
    --task EasyUUV-Direct-v1 \
    --num_envs 512 \
    --max_iterations 1500 \
    --headless
```

## C3. Play — STDW 慢环评估（单 cell）

```bash
bash custom_workflows/run_with_isaac_env.sh workflows/play_stdw_adapt.py \
    --task EasyUUV-Direct-Parametric-v1 \
    --experiment_name easyuuv_parametric \
    --num_envs 1 --headless \
    --workflow_config workflows/configs/wave_storm.yaml \
    --wave_mode jonswap \
    --use_stdw True --enable_filter True \
    --target_drift 0.05 --drift_start_step 200 --drift_end_step 1200 \
    --embodiment base \
    --load_run 2026-06-08_13-48-14_stage2 \
    --checkpoint model_2398.pt
```

落盘：默认写入 `source/results/rsl_rl/<experiment>/<run>/<checkpoint>_play/`；
sweep cell 会通过 `--results_root` / `--artifacts_root` 覆盖到 cell 目录。

**STDW 双关基线对照**（写报告必用）：

| 对照项 | flags |
|---|---|
| B (off, clean) | `--use_stdw False --target_drift 0` |
| C (on)         | `--use_stdw True --target_drift 0.05` |

## C4. Sweep — 全矩阵评估

```bash
bash custom_workflows/run_with_isaac_env.sh workflows/sweep_full_matrix.py \
    --policy_path logs/rsl_rl/easyuuv_parametric/2026-06-08_13-48-14_stage2/model_2398.pt \
    --out_root .results/sweep_a3_stage2_$(date +%Y%m%d_%H%M%S) \
    --total_steps 1500 \
    --waves calm,medium,storm \
    --embodiments base,long_body,heavy_moderate,asymmetric \
    --tunes identity,full \
    --stdws off,on \
    --seeds 0
```

矩阵：3 wave × 4 emb × 2 tune × 2 stdw × 1 seed = **48 cell**，当前主线报告 wall ≈ 22.7 min。

> flag 取**逗号分隔列表**（非空格列表）。`--policy_path` 必须形如
> `<root>/logs/<exp>/<run>/<ckpt>.pt`，driver 由它解析 `--logs_root/--experiment_name/--load_run/--checkpoint`
> 并经 `custom_workflows/run_with_isaac_env.sh` 启动每个 cell。脚本默认 `--seeds 0,1,2`，
> 论文主线 48-cell 必须显式传 `--seeds 0`。

## C5. Aggregate — 报告数据制品

```bash
bash custom_workflows/run_with_isaac_env.sh workflows/tools/aggregate_full_matrix.py \
    --matrix_dir .results/sweep_a3_stage2_<timestamp>
```

落盘：
- `full_matrix.csv` / `full_matrix.json`（原始 trial 指标，由 C4 增量写入）
- `summary_aggregated.csv` / `summary_aggregated.json`（按 wave×emb×tune×stdw 聚合）
- `stdw_pairwise.csv`（STDW off↔on 配对，含 Δfmse %）

---

## Eval (Isaac-independent)

详细 SOP 见 [`EVAL_SOP.md`](EVAL_SOP.md)。

```bash
# 一次性 replay 评估（不需 Isaac Sim）
python -m easyuuv_stdw.eval.deploy_eval \
    --policy ckpt.jit \
    --replay log.csv \
    --output ./.results/eval_out.csv
```

---

## Optional. Safety Pressure Tests — 不改 C1-C5 的专项压测

用于 Lyapunov 门控、1% 初始控制器 mismatch、roll/pitch 360 度后空翻 eval。该入口是
专项实验，不替代 C4 full matrix。

```bash
# 只打印 smoke 命令，确认 OPR/router/probe 已关闭
bash custom_workflows/run_with_isaac_env.sh workflows/sweep_stdw_safety_pressure.py \
    --profile smoke --dry_run

# 小而硬完整矩阵
bash custom_workflows/run_with_isaac_env.sh workflows/sweep_stdw_safety_pressure.py \
    --profile small_hard --run \
    --results_root .results/stdw_safety_pressure_$(date +%Y%m%d_%H%M%S)
```

硬验收字段：
- `pressure_runs.csv` / `pressure_summary.json` 中的 `pass_vs_off`
- asymmetric 相关 cell 必须显式 `--auto_drift_router False --enable_micro_probe False --drift_router_mode off`
- clean off 基线必须是 `--use_stdw False --target_drift 0`

---

## Flag 兼容性总表

| Flag | C1/C2 train | C3 play | C4 sweep | 备注 |
|---|---|---|---|---|
| `--task` | ✓ | ✓ | (嵌套传递) | EasyUUV-Direct-v1 / EasyUUV-Direct-Parametric-v1 |
| `--num_envs` | ✓ | ✓ | – | sweep 内部固定 1 |
| `--max_iterations` | ✓ | – | – | – |
| `--logger` | ✓ | – | – | tensorboard / wandb / null |
| `--meta_stage` | ✓ | – | – | A3 主线用 1/2 两阶段训练 |
| `--stage1_checkpoint` | ✓ | – | – | `--meta_stage 2` 必填 |
| `--workflow_config` | – | ✓ | (yaml 列表) | yaml 配置注入入口 |
| `--wave_mode` | – | ✓ | – | none/constant/sine/jonswap |
| `--use_stdw` | – | ✓ | (sweep 自动遍历) | bool |
| `--target_drift` | – | ✓ | (随 stdw 翻转) | m |
| `--embodiment` | – | ✓ | (sweep 遍历) | 4-5 选 |
| `--lyapunov_gate_mode` | – | ✓ | (专项 sweep) | sample_mask / strict_sample_mask / guarded_drift |
| `--pid_multipliers` | – | ✓ | (专项 sweep) | JSON；用于控制器 mismatch 压测 |
| `--load_run` | – | ✓ | (从 `--policy_path` 解析) | 精确 run 目录名 |
| `--checkpoint` | – | ✓ | (从 `--policy_path` 解析) | ckpt 文件名，如 `model_2398.pt` |
| `--experiment_name` | – | ✓ | – | 必须与 task 维度匹配 |

---

## 命令风格规范

1. 所有路径用绝对路径或 `./` 前缀的相对路径，避免 `~/` 在 worker subprocess 下解析失败。
2. yaml 注入永远走 `--workflow_config`，**不要**用 `--cfg` 或 `--env_cfg`（不存在）。
3. STDW 双关默认值：`use_stdw=True --target_drift 0.05`。要做 clean baseline 必须 `False + 0`。
4. 单 cell 测试时 `--num_envs 1 --headless`；sweep 时由 driver 注入。
5. JONSWAP 的 `hs/fp/gamma/depth/direction/seed` 没有 CLI；必须由 `workflows/configs/*.yaml`
   写入 `env.disturbance_cfg.jonswap_*`。

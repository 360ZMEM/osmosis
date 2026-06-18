# Command Contract

EasyUUV-STDW 的全部对外命令收敛到 5 条契约。这份文档是契约源（source of truth），
README 与 RUNBOOK 都引用此处。

> 所有命令的工作目录都是 **EasyUUV-STDW 仓库根**，即包含 `easyuuv_env.py` 的目录。

---

## C1. Train — meta-control 训练（8 维）

```bash
python workflows/train_meta.py \
    --task EasyUUV-Direct-Parametric-v1 \
    --num_envs 512 \
    --max_iterations 1500 \
    --logger tensorboard
```

落盘：
- `~/isaaclab/logs/rsl_rl/easyuuv_parametric/<run>/model_*.pt`
- 同目录下 `compact_log.jsonl` + `mse_curve.jsonl`（极简日志）

兼容性：`--task EasyUUV-Direct-v1`（4 维基线）也可，对应 `experiment_name=easyuuv_direct`。

## C2. Train — baseline（4 维）

```bash
python workflows/train_meta.py \
    --task EasyUUV-Direct-v1 \
    --num_envs 512 \
    --max_iterations 1500
```

## C3. Play — STDW 慢环评估（单 cell）

```bash
python workflows/play_stdw_adapt.py \
    --task EasyUUV-Direct-Parametric-v1 \
    --experiment_name easyuuv_parametric \
    --num_envs 1 --headless \
    --workflow_config workflows/configs/wave_storm.yaml \
    --wave_mode jonswap \
    --use_stdw True --enable_filter True \
    --target_drift 0.05 --drift_start_step 200 --drift_end_step 1200 \
    --embodiment base \
    --checkpoint /abs/path/to/model_1499.pt
```

落盘：`./.results/<run>/summary.json`、`tracking_mse.csv`、`output.csv`。

**STDW 双关基线对照**（写报告必用）：

| 对照项 | flags |
|---|---|
| B (off, clean) | `--use_stdw False --target_drift 0` |
| C (on)         | `--use_stdw True --target_drift 0.05` |

## C4. Sweep — 全矩阵评估

```bash
python workflows/sweep_full_matrix.py \
    --policy_path /abs/path/to/model_1499.pt \
    --out_root .results/full_matrix_<date> \
    --waves calm,medium,storm \
    --embodiments base,long_body,heavy_moderate,asymmetric \
    --tunes identity,full \
    --stdws off,on \
    --seeds 0
```

矩阵：3 wave × 4 emb × 2 tune × 2 stdw × 1 seed = **48 cell**，平均 wall ≈ 23 s/cell。

> flag 取**逗号分隔列表**（非空格列表）。`--policy_path` 必须形如
> `<root>/logs/<exp>/<run>/<ckpt>.pt`，driver 由它解析 `--logs_root/--experiment_name/--load_run/--checkpoint`
> 并经 `custom_workflows/run_with_isaac_env.sh` 启动每个 cell。

## C5. Aggregate — 报告数据制品

```bash
python workflows/tools/aggregate_full_matrix.py \
    --matrix_dir .results/full_matrix_<date>
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

## Flag 兼容性总表

| Flag | C1/C2 train | C3 play | C4 sweep | 备注 |
|---|---|---|---|---|
| `--task` | ✓ | ✓ | (嵌套传递) | EasyUUV-Direct-v1 / EasyUUV-Direct-Parametric-v1 |
| `--num_envs` | ✓ | ✓ | – | sweep 内部固定 1 |
| `--max_iterations` | ✓ | – | – | – |
| `--logger` | ✓ | – | – | tensorboard / wandb / null |
| `--workflow_config` | – | ✓ | (yaml 列表) | yaml 配置注入入口 |
| `--wave_mode` | – | ✓ | – | none/constant/sine/jonswap |
| `--use_stdw` | – | ✓ | (sweep 自动遍历) | bool |
| `--target_drift` | – | ✓ | (随 stdw 翻转) | m |
| `--embodiment` | – | ✓ | (sweep 遍历) | 4-5 选 |
| `--checkpoint` | – | ✓ | (`--policy`) | 绝对路径 |
| `--experiment_name` | – | ✓ | – | 必须与 task 维度匹配 |

---

## 命令风格规范

1. 所有路径用绝对路径或 `./` 前缀的相对路径，避免 `~/` 在 worker subprocess 下解析失败。
2. yaml 注入永远走 `--workflow_config`，**不要**用 `--cfg` 或 `--env_cfg`（不存在）。
3. STDW 双关默认值：`use_stdw=True --target_drift 0.05`。要做 clean baseline 必须 `False + 0`。
4. 单 cell 测试时 `--num_envs 1 --headless`；sweep 时由 driver 注入。

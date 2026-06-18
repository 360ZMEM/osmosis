# 全实验矩阵命令契约 — 整定 + JONSWAP + cross-embodiment + STDW 前后

> 一份**最小可复现**的命令契约。读这一份就能跑全矩阵。
> 接口能力背景：[INTERFACE_REFERENCE.md](INTERFACE_REFERENCE.md)。
> 上一轮总结：[../REPORT_full_train_72cell_stdw_20260605.md](../REPORT_full_train_72cell_stdw_20260605.md)。

---

## 1. 矩阵定义（48 unique cell × 1 seed = 48 trial）

| 维度 | 取值 | 数量 |
|---|---|---|
| Wave (JONSWAP) | calm (hs=0.3) / medium (hs=0.8) / storm (hs=1.5) | 3 |
| Embodiment | base / long_body / heavy_moderate / asymmetric | 4 |
| Tune mode | identity / full | 2 |
| STDW | off / on | 2 |
| Seed | 0 | 1 |

- **Tune identity** = `identity_init=true`（旁路 4 机制；ζ_runtime ≡ ζ_nominal）。
- **Tune full** = `identity_init=false` + `enable_pe/enable_deadzone/enable_param_lpf=true` + `gain_beta=0.2`（默认）。
- **STDW off**（B 基线）= `--use_stdw False --target_drift 0`：wrapper 不推 COB drift，慢环不更新。最干净的"无 STDW"。
- **STDW on** = `--use_stdw True --target_drift 0.05`：wrapper 推 drift + 慢环每 60 步更新 + Lyapunov mask。
- 单 cell 1500 step（25 s 仿真 @ 60 Hz），每 cell ≈ 70 s wall。
- 总 wall ≈ 48 × 70 ≈ 56 min。

---

## 2. 命令契约（最小一行）

### 2.1 单 cell 命令模板

```
custom_workflows/run_with_isaac_env.sh \
  workflows_new_stdw/play_stdw_adapt.py \
  --headless \
  --task MOGA-WarpAUV-Direct-Parametric-v1 \
  --num_envs 1 \
  --experiment_name warpauv_parametric \
  --policy_path <CKPT> \
  --workflow_config workflows_new_stdw/configs/matrix_wave_<wave>_<tune>.yaml \
  --embodiment <emb> \
  --use_stdw <True|False> \
  --target_drift <0.05|0.0> \
  --total_steps 1500 \
  --seed <s> \
  --results_root <out_dir>
```

> 注：`--policy_path` 不在原 CLI 里，由 sweep 驱动改用 `--load_run` + `--checkpoint`；这里仅示意。

### 2.2 全矩阵驱动（一条命令）

```
python workflows_new_stdw/sweep_full_matrix.py \
  --policy_path .tmp/meta_train_full_20260605_220714/logs/warpauv_parametric/2026-06-05_22-08-02/model_1499.pt \
  --out_root .tmp/meta_full_matrix_$(date +%Y%m%d_%H%M%S) \
  --total_steps 1500 \
  --seeds 0
```

驱动会写：
- `<out_root>/<cell_id>/run.log`：每 cell 的 stdout
- `<out_root>/<cell_id>/results/.../summary.json`：play_stdw_adapt 原生产物
- `<out_root>/full_matrix.csv`：增量写入的聚合表（cell_id / wave / emb / tune / stdw / seed / final_mse / final_mse_after_drift / convergence_step / wall_seconds / returncode）
- `<out_root>/full_matrix.json`：JSON 副本

---

## 3. 配置目录约定

| yaml | 含义 |
|---|---|
| [configs/matrix_wave_calm_identity.yaml](configs/matrix_wave_calm_identity.yaml) | hs=0.3, identity_init=true |
| [configs/matrix_wave_calm_full.yaml](configs/matrix_wave_calm_full.yaml) | hs=0.3, full tune |
| [configs/matrix_wave_medium_identity.yaml](configs/matrix_wave_medium_identity.yaml) | hs=0.8, identity_init=true |
| [configs/matrix_wave_medium_full.yaml](configs/matrix_wave_medium_full.yaml) | hs=0.8, full tune |
| [configs/matrix_wave_storm_identity.yaml](configs/matrix_wave_storm_identity.yaml) | hs=1.5, identity_init=true |
| [configs/matrix_wave_storm_full.yaml](configs/matrix_wave_storm_full.yaml) | hs=1.5, full tune |

每份 yaml 内部**自带** `env.tune_gains`/`identity_init`/`disturbance_cfg.jonswap_*`，因此 sweep 驱动只需在 CLI 里覆盖 `--embodiment` / `--use_stdw` / `--target_drift` / `--seed` 就够了。

---

## 4. 跑后处理

```
python workflows_new_stdw/tools/aggregate_full_matrix.py \
  --matrix_dir <out_root>
```

- 输出 `summary_aggregated.json`：按 (wave, emb, tune, stdw) 分组 mean ± std
- 关键 4 列对照：`stdw_off` vs `stdw_on` 的 `final_mse_after_drift` / `convergence_step`

最终报告写入 `REPORT_full_matrix_<date>.md`。

---

## 5. 跑前自检

跑 1 cell smoke 验证管线（约 70 s）：

```
python workflows_new_stdw/sweep_full_matrix.py \
  --policy_path .tmp/meta_train_full_20260605_220714/logs/warpauv_parametric/2026-06-05_22-08-02/model_1499.pt \
  --out_root .tmp/full_matrix_smoke_$(date +%Y%m%d_%H%M%S) \
  --total_steps 400 \
  --seeds 0 \
  --waves calm \
  --embodiments base \
  --tunes identity \
  --stdws off
```

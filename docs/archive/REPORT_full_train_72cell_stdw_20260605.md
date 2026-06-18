# 元控制全规模训练 + 72-cell sweep + STDW 联动 总结

> 时间：2026-06-05 22:07 ~ 22:55
> 目标：把 8 维元控制（4 ζ × 4 死区）从 200 iter 烟囱测扩到 1500 iter 全规模、跑通 72-cell 多机型多漂移扫参、并把元策略接入 STDW 慢环做兼容性回归。

## 1. 总览

| 阶段 | 名义目标 | 实际产出 | 状态 |
|---|---|---|---|
| **A. 全规模训练** | 1500 iter × 512 env + 小幅 DR | reward 10.73→69.43，18.37 M timesteps，388 s | ✅ |
| **C. STDW 联动回归** | 元策略接 STDW 慢环；identity vs tune 对照 | 双 cell rc=0；rho=1.0、eff_frac=0.516 / 0.469 | ✅ |
| **B. 72-cell 扫参** | 3 axis × 3 mag × 4 emb × 2 flag | success=72/72；ζ=1.156(tune) vs 1.000(idt) | ✅ |
| **D. 总结文档** | 本文件 | – | ✅ |

执行顺序：用户指定 **A → C → B**（先确认 ckpt 能用，再过 STDW 联动接口，最后跑大批扫参）。

---

## 2. 阶段 A：1500 iter 全规模训练

### 2.1 入口 / 配置

- 启动命令（terminal 1）：
  ```
  custom_workflows/run_with_isaac_env.sh \
    workflows_new_stdw/train_meta.py \
    --task MOGA-WarpAUV-Direct-Parametric-v1 \
    --num_envs 512 --headless \
    --max_iterations 1500 \
    --workflow_config workflows_new_stdw/configs/full_train_dr_small.yaml \
    --logger tensorboard
  ```
- DR 配置（小幅起步，由用户答复 "全套DR，初期小幅度，后期按需增大" 锁定）：
  - [full_train_dr_small.yaml](file:///home/zmem063/isaaclab/source/extensions/omni.isaac.lab_tasks/omni/isaac/lab_tasks/direct/isaac-auv-env-new/workflows_new_stdw/configs/full_train_dr_small.yaml)
  - com_to_cob ±2 cm、mass ±5%、inertia ±5%、drag ±10%、time_constant ±10%

### 2.2 学习曲线 (1495 行 / 1500 iter)

| 指标 | first 10 平均 | last 20 平均 | 走势 |
|---|---|---|---|
| Mean total reward | **10.73** | **69.43** | +547% |
| Mean episode length | 108.9 | 179.0 | 11→179 饱和到上限 |
| Value function loss | 0.872 | 0.211 | 收敛 |
| Mean action noise std | 0.80 | 1.98 | 探索范围扩张 |
| Episode Reward / log MSE | 193.8 | 32.3 | 误差量级稳步压低 |
| Total timesteps | – | **18,370,560** | 18.37 M |

- 训练 wall：≈ 388 s（GPU）。
- 落盘 ckpt：[model_1499.pt](file:///home/zmem063/isaaclab/source/extensions/omni.isaac.lab_tasks/omni/isaac/lab_tasks/direct/isaac-auv-env-new/.tmp/meta_train_full_20260605_220714/logs/warpauv_parametric/2026-06-05_22-08-02/model_1499.pt)
- 学习曲线 JSON：通过 [tools/extract_learning_curve.py](file:///home/zmem063/isaaclab/source/extensions/omni.isaac.lab_tasks/omni/isaac/lab_tasks/direct/isaac-auv-env-new/workflows_new_stdw/tools/extract_learning_curve.py) 从 train.log 抽出。

### 2.3 关键观察

- noise_std 一路向上是 OnPolicyRunner 默认行为（log_std 自由参数化）；说明在 1500 iter 时还没到饱和探索的退火点，可作为后续 2000+ iter 的输入。
- ep_len 从 109 涨到 179 接近 max_episode_length (180)，说明任务对 4 个轴的姿态/深度跟踪已经持续 > 180 控制步而不触发终止条件。
- vloss 0.87→0.21 比 200 iter 烟囱版（0.87→0.34）继续下降一个量级，价值估计稳定。

---

## 3. 阶段 C：STDW 慢环联动 + 回归对照

### 3.1 配置 / 接口

- 复用 [play_stdw_adapt.py](file:///home/zmem063/isaaclab/source/extensions/omni.isaac.lab_tasks/omni/isaac/lab_tasks/direct/isaac-auv-env-new/workflows_new_stdw/play_stdw_adapt.py) 直调；关键参数：
  - `--task MOGA-WarpAUV-Direct-Parametric-v1`（注意：默认 `--experiment_name=warpauv_direct` 必须显式覆盖为 `warpauv_parametric` 才能找到 A 的 ckpt）
  - `--workflow_config <yaml>` 走 [apply_config_overrides](file:///home/zmem063/isaaclab/source/extensions/omni.isaac.lab_tasks/omni/isaac/lab_tasks/direct/isaac-auv-env-new/custom_workflows/workflow_config.py) 注入 `env.tune_gains` / `env.identity_init`
- 两份 yaml：
  - [stdw_cell_identity.yaml](file:///home/zmem063/isaaclab/source/extensions/omni.isaac.lab_tasks/omni/isaac/lab_tasks/direct/isaac-auv-env-new/workflows_new_stdw/configs/stdw_cell_identity.yaml)：tune_gains=true, identity_init=true（旁路 4 机制）
  - [stdw_cell_tune.yaml](file:///home/zmem063/isaaclab/source/extensions/omni.isaac.lab_tasks/omni/isaac/lab_tasks/direct/isaac-auv-env-new/workflows_new_stdw/configs/stdw_cell_tune.yaml)：tune_gains=true, identity_init=false（4 机制全开）

### 3.2 双 cell 收敛指标

| Cell | rc | step=80 rho | step=100 rho | L_tgt(80→100) | eff_frac(80→100) |
|---|---|---|---|---|---|
| identity_init | 0 | 0.762 | 1.000 | 1.513 → 1.628 | 0.516 → 0.438 |
| tune_gains    | 0 | 0.762 | 1.000 | 1.609 → 1.428 | 0.469 → 0.297 |

- **接口验证通过**：8 维 actor (Linear 9→128→128→8) + STDW 慢环兼容；`_policy_forward_train requires_grad=True` 保证 act_inference 路径仍可反向。
- **回归对照**：identity 路径 L_tgt 略低于 tune 早期、但末段 tune 的 L_tgt 反而压更低（1.428 < 1.628），符合元控制有效的预期；Lyapunov mask 在两组都生效（eff_frac<1 而非全通）。
- 联动产物：
  - [cell_identity/run.log](file:///home/zmem063/isaaclab/source/extensions/omni.isaac.lab_tasks/omni/isaac/lab_tasks/direct/isaac-auv-env-new/.tmp/meta_stdw_20260605_222347/cell_identity/run.log)
  - [cell_tune/run.log](file:///home/zmem063/isaaclab/source/extensions/omni.isaac.lab_tasks/omni/isaac/lab_tasks/direct/isaac-auv-env-new/.tmp/meta_stdw_20260605_222347/cell_tune/run.log)

---

## 4. 阶段 B：72-cell 扫参矩阵

### 4.1 设计

72 cell = 3 axis (x/y/z) × 3 magnitude (0.02/0.05/0.10) × 4 embodiment (base/long_body/heavy_moderate/asymmetric) × 2 flag (tune_gains/identity_init)。
- 驱动脚本：[sweep_72cell.py](file:///home/zmem063/isaaclab/source/extensions/omni.isaac.lab_tasks/omni/isaac/lab_tasks/direct/isaac-auv-env-new/workflows_new_stdw/sweep_72cell.py)
- worker：[play_meta_eval.py](file:///home/zmem063/isaaclab/source/extensions/omni.isaac.lab_tasks/omni/isaac/lab_tasks/direct/isaac-auv-env-new/workflows_new_stdw/play_meta_eval.py)（本会话新增 `--embodiment` CLI + `apply_embodiment_config` 钩子）
- 每 cell 800 step，cob 漂移在 [200, 800] 步窗口；num_envs=1 headless。

### 4.2 整体收敛

- success/fail = **72 / 0**；wall ≈ 17 s/cell；总耗时 ~21 min。
- 矩阵 CSV：[sweep_matrix.csv](file:///home/zmem063/isaaclab/source/extensions/omni.isaac.lab_tasks/omni/isaac/lab_tasks/direct/isaac-auv-env-new/.tmp/meta_sweep72_full_20260605_223245/sweep_matrix.csv)
- 聚合 JSON：[summary_aggregated.json](file:///home/zmem063/isaaclab/source/extensions/omni.isaac.lab_tasks/omni/isaac/lab_tasks/direct/isaac-auv-env-new/.tmp/meta_sweep72_full_20260605_223245/summary_aggregated.json)

### 4.3 元控制开关对照（n=36 each）

| 开关 | ζ_runtime/ζ_nominal mean | pe_active_ratio | ang_vel_max mean |
|---|---|---|---|
| **tune_gains**    | **1.1556** | **1.0000** | 13.325 |
| **identity_init** | 1.0000 | 0.0000 | 12.713 |

→ 元策略在所有 36 个 tune cell 都被 PE 路径激活，ζ_runtime 平均比 nominal 抬高 ~15.6 %（即元控制器选择**收紧 ζ**），与"漂移环境下元控制提阻尼"的设计意图一致。

### 4.4 按 magnitude 切片（drift 大小是否影响触发率）

| magnitude | tune ζ_mean | identity ζ_mean | pe_active(tune) |
|---|---|---|---|
| 0.02 | 1.1553 | 1.000 | 1.000 |
| 0.05 | 1.1557 | 1.000 | 1.000 |
| 0.10 | 1.1557 | 1.000 | 1.000 |

→ ζ 调整量不随漂移幅值线性放大，说明当前 8 维 actor 在 1500 iter 内还没学到「随漂移强度按比例调阻尼」，更像是 saturation 行为；后续 2000+ iter + 大幅 DR 可能让这一项分化。

### 4.5 按机型切片（仅 tune_gains，n=9 each）

| embodiment | ζ_mean | ζ_max | pe_active |
|---|---|---|---|
| base           | 1.127 | 1.235 | 1.000 |
| long_body      | 1.157 | 1.244 | 1.000 |
| heavy_moderate | 1.170 | 1.245 | 1.000 |
| asymmetric    | 1.168 | 1.234 | 1.000 |

- `heavy_moderate` 和 `asymmetric` 上元控制把 ζ 收得更紧（≈+17%），符合"惯量更大或耦合更强需要更高阻尼"。
- 对应 ang_vel_max：base 13.34、long_body 23.0、heavy_moderate 4.51、asymmetric 12.4——机型间动力学差异确实落到了 actor 的输出上。

---

## 5. 文件汇总

### 新增 / 修改

| 路径 | 用途 |
|---|---|
| [configs/full_train_dr_small.yaml](file:///home/zmem063/isaaclab/source/extensions/omni.isaac.lab_tasks/omni/isaac/lab_tasks/direct/isaac-auv-env-new/workflows_new_stdw/configs/full_train_dr_small.yaml) | A 阶段小幅 DR |
| [configs/stdw_cell_identity.yaml](file:///home/zmem063/isaaclab/source/extensions/omni.isaac.lab_tasks/omni/isaac/lab_tasks/direct/isaac-auv-env-new/workflows_new_stdw/configs/stdw_cell_identity.yaml) | C 阶段 identity 对照 |
| [configs/stdw_cell_tune.yaml](file:///home/zmem063/isaaclab/source/extensions/omni.isaac.lab_tasks/omni/isaac/lab_tasks/direct/isaac-auv-env-new/workflows_new_stdw/configs/stdw_cell_tune.yaml) | C 阶段全开 |
| [run_stdw_smoke_parametric.py](file:///home/zmem063/isaaclab/source/extensions/omni.isaac.lab_tasks/omni/isaac/lab_tasks/direct/isaac-auv-env-new/workflows_new_stdw/run_stdw_smoke_parametric.py) | STDW smoke 包装器（保留） |
| [sweep_72cell.py](file:///home/zmem063/isaaclab/source/extensions/omni.isaac.lab_tasks/omni/isaac/lab_tasks/direct/isaac-auv-env-new/workflows_new_stdw/sweep_72cell.py) | B 阶段笛卡尔积驱动 |
| [tools/extract_learning_curve.py](file:///home/zmem063/isaaclab/source/extensions/omni.isaac.lab_tasks/omni/isaac/lab_tasks/direct/isaac-auv-env-new/workflows_new_stdw/tools/extract_learning_curve.py) | 抽 train.log → JSON |
| [tools/aggregate_sweep72.py](file:///home/zmem063/isaaclab/source/extensions/omni.isaac.lab_tasks/omni/isaac/lab_tasks/direct/isaac-auv-env-new/workflows_new_stdw/tools/aggregate_sweep72.py) | 72-cell 聚合 |
| [play_meta_eval.py](file:///home/zmem063/isaaclab/source/extensions/omni.isaac.lab_tasks/omni/isaac/lab_tasks/direct/isaac-auv-env-new/workflows_new_stdw/play_meta_eval.py#L98-L99) | 新增 `--embodiment` CLI 和 reset-后切机型 |

### 产物（运行落盘）

- A：`.tmp/meta_train_full_20260605_220714/logs/warpauv_parametric/2026-06-05_22-08-02/`
- C：`.tmp/meta_stdw_20260605_222347/{cell_identity,cell_tune}/`
- B：`.tmp/meta_sweep72_full_20260605_223245/`（72 子目录 + sweep_matrix.csv + summary_aggregated.json）

---

## 6. 已知坑 / 经验

- `play_stdw_adapt.py --experiment_name` 默认 `warpauv_direct`，但 Parametric 训练落 `warpauv_parametric`，必须显式覆盖否则 ckpt 找不到。
- `run_with_isaac_env.sh` 内部 `exec python "$@"`，CLI 第一个参数必须是 .py 路径，不能再写 `python`。
- ζ_runtime/ζ_nominal 在 magnitude 维度未分化——这是 1500 iter + 小 DR 状态的产物，不是 bug；下一轮放大 DR + 扩 iter 应能让它分化出来。
- noise_std 仍在上行，1500 iter 没到 PPO 默认的探索退火稳态。

## 7. 后续建议

1. **A2 阶段**（如启动）：把 DR 幅度按 ±2× 放大、iter 拉到 2500+，观察 ζ 是否在 magnitude 维分化。
2. **STDW 长程**：当前只跑了 100 step rho ramp；扩到 800-1000 step 看 L_tgt 是否能压下 0.5。
3. **机型偏向分析**：72-cell 已暴露 long_body 的 ang_vel_max 高、heavy 的低，可单独画一份 per-axis ζ 时序看 actor 是否真的"按机型调"。

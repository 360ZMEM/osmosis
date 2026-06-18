# Error Cases / 调试坑记录

本仓库迁移与运行过程中真实碰到的 6 个坑。每条都给出**触发条件 / 根因 / 修复 / 验证**。
新成员先扫一遍能省 1-2 小时。

---

## Case 1. JONSWAP yaml 注入失效（wave 三档完全相同）

### 触发
跑 sweep 矩阵之后，`stdw_pairwise.csv` 里 calm/medium/storm 三档的 `final_mse`、
`fluid_vx@step100` 完全一致。

### 根因（三层 bug，逐层揭开）
1. `play_stdw_adapt.py` 在 `apply_config_overrides(yaml)` 之后用 CLI 默认值
   （`wave_mode=sine`、`base_vel=[0.06,0,0.02]`）整段覆盖 `disturbance_cfg`。
2. `disturbance_cfg` 是 `easyuuv_env.py` 内的 inner dataclass，
   `apply_config_overrides` 递归 setattr 后 env 内部仍读到错误值。
3. `_set_initial_disturbance` 在 main loop 入口第三次用 CLI 默认值调
   `apply_runtime_domain_shift`，把 mode/base_vel 又覆盖回 sine。

### 修复
- `_set_initial_disturbance` 接受 `yaml_disturbance` 字典；sentinel 模式（仅当 yaml 没指定才用 CLI 默认值）。
- `apply_runtime_domain_shift` 加 6 个 jonswap kwargs；变更时使 `_wave_manager` 失效以重建。

### 验证
calm/medium/storm 三档 step 100 处 `fluid_vx` 应分别约 `0.058 / 0.093 / 0.187`。
不一致就是退化。

---

## Case 2. Actor MLP 维度不匹配（4 vs 8）

### 触发
```
RuntimeError: Error(s) in loading state_dict for Actor:
  size mismatch for actor.mlp.weight: copying a param with shape [4, 64] but
  the model expects [8, 64].
```

### 根因
ckpt 是 8 维 parametric 训练产出（`easyuuv_parametric`），但 play 命令默认
`--experiment_name easyuuv_direct`（4 维）。

### 修复
`--task EasyUUV-Direct-Parametric-v1 --experiment_name easyuuv_parametric` 必须**同时**给。

### 验证
`policy.actor.mlp.0.weight.shape[0]` 应等于 `cfg.num_actions`（4 或 8）。

---

## Case 3. STDW "off" 实际仍跑 wrapper

### 触发
预期"无 STDW 基线"应该 fmse 与 wrapper 完全无关，但实测 off 与 on 数据走势相似，
都受 drift 影响。

### 根因
`--use_stdw False` 只 skip 慢环触发；wrapper 仍每步推 COB drift（默认 `target_drift=0.05`）。

### 修复
clean baseline 必须 **`--use_stdw False --target_drift 0`** 双关。

### 验证
B (off, clean) 的 `final_mse_after_drift` 应与 `final_mse` 接近相等（drift 不应有效果）。

---

## Case 4. ζ_nominal 快照时机错误

### 触发
跑 8 维 parametric 任务，发现 `ζ_runtime` 在 reset 后第一步就异常大或为 0。

### 根因
domain randomization 在 `_pre_physics_step` 早期会改 `PID_args`；
若 `_zeta_nominal = PID_args[:,:,0].clone()` 发生在 DR 之前，则之后 PE/safeguard 都基于错误基线。

### 修复
快照统一放在 `_refresh_domain_randomization_defaults` 末尾，每次 DR 之后重置；
同时 `_gain_tuner.reset()`。

### 验证
单 cell + identity_init=True 应得到 `ζ_runtime ≡ ζ_nominal`，`abs(diff).max() < 1e-6`。

---

## Case 5. `heavy_duty` embodiment 不可达

### 触发
sweep 想扫 5 个 embodiment 但只跑出 4 行结果。

### 根因
`play_stdw_adapt.py --embodiment` 的 argparse `choices` 漏 `heavy_duty`；
env 字典里有但 CLI gate 拒收。

### 修复（视需要二选一）
1. 改 `play_stdw_adapt.py` 的 argparse choices 列表加上 `heavy_duty`。
2. 改 sweep worker 改用 `play_meta_eval.py`（无 choices 限制）。

### 验证
跑 `--embodiment heavy_duty` 不应报 argparse error。

---

## Case 6. USD 渲染断链（资产命名）

### 触发
迁移仓库时把 `data/warpauv/` 改名为 `data/easyuuv/`，启动报
`UsdShade Material xxxx not found`。

### 根因
USD 内部对纹理 / 子 mesh 路径有**绝对引用** `data/warpauv/...`，
重命名后 stage 解析不到。

### 修复
保留 `data/warpauv/` 与 `WARPAUV_CFG` 两个内部命名，
对外只在 Gym ID / 类名 / experiment_name 层做 EasyUUV 重命名。

### 验证
`grep -r '"data/warpauv' assets/` 应仍有匹配；
启动训练 5 秒内不报 USD error。

---

## 通用排查清单

跑实验前的 60-second 自检：

```bash
# 1. py_compile 所有 .py
python -m py_compile easyuuv_env.py easyuuv_stdw_wrapper.py gain_tuner.py \
    workflows/train_meta.py workflows/play_stdw_adapt.py \
    workflows/sweep_full_matrix.py

# 2. eval 子模块独立测试（不依赖 Isaac）
python -m py_compile eval/__init__.py eval/wrappers.py eval/policy_loader.py

# 3. Gym 注册检查
python -c "import gymnasium as gym; import easyuuv_stdw; \
    print([s for s in gym.registry if 'EasyUUV' in s])"
# 期望: ['EasyUUV-Direct-v1', 'EasyUUV-Direct-Parametric-v1']
```

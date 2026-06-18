# Architecture

EasyUUV-STDW 是一个 Isaac Lab Direct RL 任务包 + 独立的 Isaac-independent 评估子模块。
本文给出模块层次、数据流、文件职责清单。

---

## 1. 模块图

```
EasyUUV-STDW
├── __init__.py            ── Gym 注册：EasyUUV-Direct-v1 / EasyUUV-Direct-Parametric-v1
│
├── easyuuv_env.py         ── 主 env (DirectRLEnv)
│   ├── 4-D 观测构建（pos_err, yaw_err, lin_vel_b, ang_vel_b）
│   ├── 4 / 8-D action: ctrl + a_gain
│   ├── PID 内环 (apply_control_profile)
│   ├── 流体动力学 (rigid_body_hydrodynamics + thruster_dynamics)
│   ├── 扰动注入 (get_current_fluid_velocity → wave_disturbance_manager)
│   └── meta-control 钩子 (gain_tuner: LPF→deadzone→safeguard→PE)
│
├── easyuuv_stdw_wrapper.py ── gym.Wrapper (慢时域)
│   ├── COB drift 推进
│   ├── 5s 滑窗 RMS
│   └── Lyapunov mask
│
├── gain_tuner.py          ── ParametricGainTuner（开环 4 阶段）
├── wave_disturbance_manager.py  ── JONSWAP 谱波生成
├── rigid_body_hydrodynamics.py
├── thruster_dynamics.py
├── asymmetric_noise_cfg.py
├── expanded_domain_randomization.py
│
├── agents/
│   └── rsl_rl_ppo_cfg.py  ── 两个 PPORunnerCfg（baseline 4-D / parametric 8-D）
│
├── stdw_integration/      ── STDW 与 PPO 内核的桥接（slow-loop 触发器、buffer）
│
├── workflows/             ── Isaac-Lab-依赖的训练/评估脚本
│   ├── train_meta.py
│   ├── play_stdw_adapt.py
│   ├── play_meta_eval.py
│   ├── sweep_full_matrix.py / sweep_72cell.py / sweep_stdw.py
│   ├── configs/<wave_*.yaml>
│   └── tools/aggregate_*.py
│
├── eval/                  ── Isaac-independent (numpy + torch / onnxruntime)
│   ├── wrappers.py        ── obs_from_state / reward_from_state
│   ├── policy_loader.py   ── Policy({.pt | .jit | .onnx})
│   ├── train_loop.py      ── 50 行 reference PPO
│   ├── deploy_eval.py     ── CLI replay 评估
│   └── examples/
│
├── assets/                ── 物理 cfg + USD 引用
├── data/warpauv/          ── USD/URDF 资产（保留命名以避免 USD 内部路径断链）
└── docs/                  ── 本目录
```

---

## 2. 数据流

### 2.1 训练（C1）

```
ENV_RESET ──▶ obs (10D) ──▶ Actor MLP ──▶ μ (4D or 8D)
                                           │
                                           ├──▶ ctrl[:4]  ──▶ apply_control_profile (PID)
                                           │                  ├─▶ thruster_dynamics
                                           │                  └─▶ hydrodynamics + wave/fluid
                                           │                          │
                                           │                          ▼
                                           │                       sim step
                                           │                          │
                                           └──▶ a_gain[4:] ──▶ ParametricGainTuner
                                                                ├─ LPF
                                                                ├─ deadzone
                                                                ├─ Bounded Safeguard ζ_eff = ζ_nom·(1+β·a)
                                                                └─ PE injection
                                                                  ▼
                                                              ζ_runtime → PID_args[:,:,col]
```

`compact_log.jsonl` + `mse_curve.jsonl` 由 monkey-patched `runner.log` 写出。

### 2.2 慢环评估（C3）

```
EasyUUVEnv ──gym.Wrapper──▶ EasyUUVStdwWrapper ──▶ Policy.act()
   │            ├ COB drift 推进
   │            ├ 5s RMS
   │            └ Lyapunov mask
   │
   每 60 step ──▶ slow_loop_trigger ──▶ stdw_integration ──▶ policy 微调
                                                              ├─ MSE on masked window
                                                              └─ L2 anchor / behavior_kl
   │
   END ──▶ summary.json (final_mse, drift, slow_loop_triggers, ...)
       └─▶ tracking_mse.csv
```

### 2.3 部署评估（Isaac-independent）

```
real-vehicle telemetry / CSV ──▶ obs_from_state(state)  (10D)
                                       │
                                       ▼
                                Policy(.pt|.jit|.onnx).act(obs)
                                       │
                                       ▼
                              4D or 8D action ──▶ thruster mixer ──▶ vehicle
                                                       │
                                                       ▼
                                       deploy_eval.py 累计 fmse / rmse
```

---

## 3. 关键耦合点

| 文件对 | 耦合点 | 失效后果 |
|---|---|---|
| `easyuuv_env.py` ↔ `gain_tuner.py` | `_zeta_nominal` 的快照时机（必须在 domain randomization 之后） | a_gain 路由错位，整定头无效 |
| `easyuuv_env.py` ↔ `wave_disturbance_manager.py` | `_wave_manager` 缓存键签名 | hs/fp 改了但波形不变 |
| `easyuuv_env.py` ↔ `easyuuv_stdw_wrapper.py` | `set_environment_drift_axes`，`get_buoyancy_offset` | wrapper 推不动 COB drift |
| `play_stdw_adapt.py` ↔ `easyuuv_env.py` | `apply_runtime_domain_shift` 不覆盖 jonswap_* | yaml 注入失效（[`ERROR_CASES.md`](../guide/ERROR_CASES.md) §1） |
| `__init__.py` ↔ `agents/rsl_rl_ppo_cfg.py` | `experiment_name` 与 `num_actions` 必须一致 | actor MLP 维度不匹配 |
| `eval/wrappers.py` ↔ `easyuuv_env.py` | obs 列顺序与训练时一致 | 推理输出乱码 |

---

## 4. 设计原则

1. **快慢分离**：env step (1/120 s) 是快环，wrapper 累积 5 s 后慢环触发，二者通过 buffer 解耦。
2. **机制开关，不是闭环**：整定不调 PID，而是在 ζ_nom 上做 Bounded Safeguard ±β。
   这样可证明每一步都不会让控制器离开稳定边界。
3. **eval 不依赖 Isaac**：所有部署侧数据契约（obs/reward）都用 numpy 实现，
   保证可在板载 PC、CI、ONNX Runtime Web 上跑。
4. **历史命名保留**：`WARPAUV_CFG`、`data/warpauv/` 不重命名，因为 USD 内部对资产路径有硬引用，
   重命名会直接断渲染。对外 API 全部 EasyUUV*，对内资产保留 warpauv。

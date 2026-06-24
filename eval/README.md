# `eval/` - Isaac 独立部署与 replay 评估工具

`eval/` 是 EasyUUV-STDW 面向 **Isaac 外部运行** 的接口层。它不导入
`omni.isaac.*`、`rsl_rl`、Isaac Lab runtime 或 gym 注册，因此可以放到板载 PC、
水池实验台、CI runner 或普通 Linux 笔记本上运行。

当前代码的主线部署对象是：

- 任务：`EasyUUV-Direct-Parametric-v1`
- 策略：A3 stage2 / `model_2398` 系列导出的 `*_deploy.jit`
- 观测：默认 `a3_12d`
- 动作：8D 参数化动作 `[u0, u1, u2, u3, a_gain0, a_gain1, a_gain2, a_gain3]`
- STDW 实物策略：默认关闭，先验证 baseline，再做少样本 micro-probe / 慢环更新

完整 SOP 见：

- [`docs/guide/EVAL_SOP.md`](../docs/guide/EVAL_SOP.md)：离线 replay、指标、CI gate。
- [`docs/guide/DEPLOY_SOP_realworld.md`](../docs/guide/DEPLOY_SOP_realworld.md)：实物上板流程、安全项、TODO 索引。

## 1. 模块清单

| 文件 | 作用 |
|---|---|
| `wrappers.py` | `obs_from_state(state, layout=...)` 和 `reward_from_state(...)`；定义硬件 state dict 到策略观测的转换。当前默认 `a3_12d`，保留 `legacy_10d` 兼容路径。 |
| `policy_loader.py` | `Policy(path).act(obs)`；按扩展名加载 `.pt` / `.jit` / `.onnx`。实物部署推荐 `*_deploy.jit`，不要直接用 RSL-RL 的 `model_*.pt` checkpoint dict。 |
| `deploy_config.py` | 读取 `deploy_config.yaml`，将串口、控制周期、策略路径、STDW 选项集中配置。 |
| `deploy_config.yaml` | 实物部署配置入口；默认 `serial.enabled=false`、`stdw.enable=false`，便于 dry-run。 |
| `deploy_eval.py` | 离线 replay CLI：读取真实或仿真日志 CSV，逐步跑策略，输出动作、reward、位置误差和摘要指标。 |
| `train_loop.py` | Isaac 独立的最小 PPO 参考实现，只用于理解 obs/reward 契约，不替代 `workflows/train_meta.py`。 |
| `examples/replay_csv_demo.py` | 单步加载策略、构造 state、生成 obs/action 的最小自检。 |
| `examples/thruster_io_demo.py` | 展示 4D/8D action 布局，以及一个占位推进器分配矩阵。上板前必须替换为真实 mixer。 |
| `examples/stdw_deploy_manager_demo.py` | 从 STDW `summary.json` 或 metadata 中选取最新 `*_deploy.jit`，做一次部署推理。 |
| `examples/real_world_runtime.py` | 实物运行骨架：dry-run 或串口读取 ESP32 姿态，构造 obs，调用策略并下发姿态命令。当前是 skeleton，不是完整 UUV autopilot。 |

## 2. 依赖与导入方式

最小依赖：

```bash
pip install numpy torch pyyaml
# 仅 ONNX 后端需要：
pip install onnxruntime
# 仅串口实物运行需要：
pip install pyserial
```

从仓库根运行时，推荐用包路径：

```bash
PYTHONPATH=/home/zmem063/isaaclab/source/extensions/omni.isaac.lab_tasks/omni/isaac/lab_tasks/direct \
python -m easyuuv_stdw.eval.deploy_eval --help
```

在当前 Isaac Lab 源树中，也可以先进入 `direct/` 的父级，使
`easyuuv_stdw` 包名可被 Python 找到。

## 3. State Dict 契约

任何硬件桥、ROS 节点、串口 bridge 或 replay loader，最终都应产出如下字典：

```python
{
    "position":           np.ndarray shape (3,),  # 世界系，单位 m
    "orientation_quat":   np.ndarray shape (4,),  # w,x,y,z
    "linear_velocity_b":  np.ndarray shape (3,),  # 机体系，单位 m/s
    "angular_velocity_b": np.ndarray shape (3,),  # 机体系，单位 rad/s
    "goal_position":      np.ndarray shape (3,),  # 世界系，单位 m
    "goal_yaw":           float,                  # rad
}
```

注意：

- 四元数顺序必须是 `w,x,y,z`。如果硬件或中间件输出 `x,y,z,w`，必须在 bridge 层转换。
- 当前 A3 12D 观测不直接使用 `linear_velocity_b` 和完整 `goal_position`，但 reward、legacy 10D、日志评估和后续扩展会用到，建议保持字段完整。
- `position[2]` 被作为深度/垂向状态进入 A3 12D 观测；实物上需明确 ENU/NED、深度正方向和仿真训练约定是否一致。

## 4. 观测布局

### 4.1 当前默认：`a3_12d`

`obs_from_state(state, layout="a3_12d")` 输出：

```text
[goal_quat(4), depth_z(1), root_quat(4), angular_velocity_b(3)]
shape = (12,), dtype = float32
```

如果 state 中没有 `goal_orientation_quat`，代码会用 `goal_yaw` 自动生成 yaw-only 目标四元数。

### 4.2 兼容旧路径：`legacy_10d`

旧的 Isaac-independent demo 使用：

```text
[pos_err_x, pos_err_y, pos_err_z,
 yaw_err,
 lin_vel_bx, lin_vel_by, lin_vel_bz,
 ang_vel_bx, ang_vel_by, ang_vel_bz]
shape = (10,), dtype = float32
```

只有在加载旧 10D demo 策略时才使用 `--obs_layout legacy_10d`。当前 A3 stage2 部署策略应使用默认 `a3_12d`。

## 5. 动作布局

4D baseline 策略输出：

```text
[u_surge, u_sway, u_heave, u_yaw]
```

8D parametric 策略输出：

```text
[u_surge, u_sway, u_heave, u_yaw,
 a_gain_kp, a_gain_ki, a_gain_kd, a_gain_kff]
```

部署约束：

- 前 4 维是控制意图，可进入低层控制器或推进器 mixer。
- 后 4 维是增益调制通道，不能直接映射为推进器命令。
- 所有动作都应先 clip 到 `[-1, 1]`。
- `examples/thruster_io_demo.py` 中的 `THRUSTER_ALLOC` 只是参考矩阵，不代表真实 UUV 推进器布局。

## 6. 快速自检

### 6.1 单步策略推理

```bash
PYTHONPATH=/path/to/direct python -m easyuuv_stdw.eval.examples.replay_csv_demo \
  --config eval/deploy_config.yaml \
  --policy /abs/path/to/stdw_step_001499_deploy.jit
```

应看到 `obs shape=(12,)` 和 `action shape=(8,)`。

### 6.2 离线 replay 评估

```bash
PYTHONPATH=/path/to/direct python -m easyuuv_stdw.eval.deploy_eval \
  --policy /abs/path/to/stdw_step_001499_deploy.jit \
  --replay /abs/path/to/log.csv \
  --output ./.results/eval_out.csv \
  --obs_layout a3_12d
```

replay CSV 必含列：

```text
t, px, py, pz, qw, qx, qy, qz, vx, vy, vz, wx, wy, wz, gx, gy, gz, gyaw
```

输出 CSV 会追加 `a0..a3` 或 `a0..a7`、`reward`、`pos_err_norm`。

摘要字段：

| 字段 | 单位 | 含义 |
|---|---|---|
| `n_steps` | step | replay 长度 |
| `fmse_pos_m2` | m^2 | 平均位置误差平方 |
| `rmse_pos_m` | m | 位置误差 RMSE |
| `mean_reward` | - | replay 上的参考 shaping reward |
| `action_dim` | - | 4 或 8 |
| `obs_layout` | - | `a3_12d` 或 `legacy_10d` |

### 6.3 dry-run 实物主循环

```bash
PYTHONPATH=/path/to/direct python -m easyuuv_stdw.eval.examples.real_world_runtime \
  --config eval/deploy_config.yaml \
  --steps 5
```

默认 `serial.enabled=false`。如果策略路径不存在且串口关闭，脚本会使用 zero-action dry-run policy，用于检查配置、导入和主循环结构。

## 7. 可部署到实际 UUV 的部分

当前 `eval/` 已经具备以下实用基础：

1. Isaac 独立运行：策略推理、replay 评估、配置读取都不需要 Isaac Sim。
2. 部署模型入口明确：支持 `.jit`、`.onnx` 和可调用 `.pt`；RSL-RL checkpoint dict 会被显式拒绝并提示导出 `*_deploy.jit`。
3. A3 12D 观测契约已落到代码：`wrappers.py` 是硬件状态到策略输入的唯一转换点。
4. 8D 动作语义已明确：前 4 维为控制意图，后 4 维为增益调制。
5. 配置集中：串口、控制周期、策略路径、STDW/micro-probe 参数都在 `deploy_config.yaml`。
6. 安全默认值偏保守：串口和 STDW 在线更新默认关闭，适合先做 dry-run 和 baseline。
7. 有实物主循环骨架：`real_world_runtime.py` 能从串口读取 ESP32 风格姿态字符串，估计角速度，生成 obs，并下发姿态命令。
8. 有端侧算力证据：当前报告测得一次 STDW 有效更新约 1-2 ms 量级，触发频率低，端侧开销可控。

## 8. 目前不足和上板前必须补齐的点

这些不是小的文档问题，而是真实 UUV 部署前必须解决的工程缺口：

1. 硬件状态桥仍是 skeleton。当前只演示 ESP32 姿态串口，缺少完整的深度、位置、线速度、DVL/压力计/声学定位/滤波融合输入。
2. 坐标系和符号约定需要实测闭环确认。A3 观测依赖 `depth_z`、`root_quat`、`angular_velocity_b`，实物上 ENU/NED、深度正方向、IMU 安装方向、四元数顺序都不能靠假设。
3. 推进器 mixer 是占位。`THRUSTER_ALLOC` 不是任何真实 UUV 的标定矩阵；必须替换为实物推进器编号、方向、PWM 范围、死区和饱和逻辑。
4. `real_world_runtime.py` 的动作到姿态命令映射只是演示。它把动作片段映射到 R/P/Y 增量，尚未严格对齐仿真中的 `[u_surge,u_sway,u_heave,u_yaw]` 语义，也没有真正接入 S 面低层控制器的 8D `a_gain` 更新路径。
5. 急停、watchdog 和故障降级尚未工程化。代码里有 `halt()` 占位，但真实 `a`/`e` 命令、串口断连、超时、推进器饱和、IMU 异常、策略 NaN 都需要硬件级处理。
6. 在线 STDW 还不能直接宣称实物闭环可用。仿真与算力结果支持可行性，但实物默认应保持 `stdw.enable=false`，先通过 baseline 水池短跑，再开启 micro-probe，最后才允许慢环更新。
7. asymmetric / 偏置机型存在已知风险。实验显示盲目 STDW 在 asymmetric 上会劣化，需要 OPR/router 或 gate；实物部署必须先做短跑判定，不应把 STDW 当成无条件改进器。
8. replay 指标仍偏离真实控制安全指标。`deploy_eval.py` 输出 fmse/rmse/reward，但还缺控制能耗、推进器饱和率、姿态角速度峰值、深度越界、通信丢包、实时 deadline miss 等实物安全指标。
9. 还缺正式导出流水线说明。部署侧需要 `*_deploy.jit`，不能直接用 `model_2398.pt`；导出命令在 `DEPLOY_SOP_realworld.md` 中有，但这里尚未形成自动化产物检查。
10. 没有 ROS2/MAVLink/厂商 SDK 适配层。当前接口足够清楚，但真正接船仍需要平台驱动层。

## 9. 建议的实际上板顺序

1. 离线确认：用实测或仿真 replay CSV 跑 `deploy_eval.py`，确认 `obs_layout=a3_12d`、动作维度为 8。
2. dry-run：`serial.enabled=false` 跑 `real_world_runtime.py`，确认策略加载、obs shape、循环频率。
3. 硬件只读：接入 IMU/深度/定位，但不下发推进器，记录 state dict 和 obs，检查单位、符号、噪声。
4. 单轴低幅闭环：保持 `stdw.enable=false`，只启用 yaw 或深度中的一个通道，动作限幅保守。
5. baseline 水池短跑：记录 100-200 step，检查 overshoot、饱和、角速度和急停。
6. micro-probe：开启极小幅可观测微扰，若证据不足则不启动 STDW。
7. STDW 慢环：满足 `min_real_samples` 后再开启，保留随时回退 baseline 的开关。
8. 报告：同时记录 fmse/rmse、尾部误差、推进器饱和率、deadline miss、急停事件和原始传感器日志。

## 10. 常见错误

| 症状 | 原因 / 处理 |
|---|---|
| `Policy loader` 报 RSL-RL checkpoint dict 不可调用 | 不能直接用 `model_*.pt`；先导出 `*_deploy.jit`。 |
| `shape mismatch` | 当前主线策略需要 `obs.shape == (12,)`；旧 demo 才用 `legacy_10d`。 |
| 导入 `easyuuv_stdw` 失败 | 设置 `PYTHONPATH` 到 `.../omni/isaac/lab_tasks/direct`。 |
| action 是 8D 但推进器只需要 4D | 只把 `action[:4]` 送入 mixer；`action[4:]` 是增益调制。 |
| 实物静止时角速度很大 | 检查 IMU 采样周期、单位、安装方向和滤波；不要先改策略。 |
| STDW 开启后误差变大 | 立即回退 baseline；检查 gate/OPR、坐标系、偏置方向和推进器饱和。 |

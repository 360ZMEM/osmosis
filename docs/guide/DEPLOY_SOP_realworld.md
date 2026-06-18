# STDW 实物机器部署 SOP（Isaac 独立）

日期：2026-06-13  
适用对象：`EasyUUV-Direct-Parametric-v1` A3 12D 观测、8D 参数化策略、STDW 少样本在线自适应部署。

## 0. 硬性边界

本 SOP 面向实物机器。运行侧不需要 Isaac、Isaac Lab、`omni.*`、`rsl_rl` 或 gym 注册。板载端只需要：

- Python 3.10+
- `numpy`
- `torch`（运行 `*_deploy.jit`）或 `onnxruntime`（运行 ONNX）
- `pyyaml`（读取 `eval/deploy_config.yaml`）
- `pyserial`（仅当串口实物运行时需要）

STDW 的实物侧原则是少样本：先用已有仿真训练策略完成稳定闭环，再用几十到一两百步真实响应样本做 observable-only 的 micro-probe / 慢环修正，不要求重新训练，也不要求大批量实物数据。

## 1. 文件清单

- `eval/deploy_config.yaml`：实物部署集中配置。
- `eval/deploy_config.py`：Isaac 独立配置 loader。
- `eval/policy_loader.py`：`.jit/.pt/.onnx` 策略加载器。
- `eval/wrappers.py`：硬件 state dict 到 A3 12D obs 的转换。
- `eval/examples/real_world_runtime.py`：实物主循环骨架，默认 dry-run。
- `eval/examples/stdw_deploy_manager_demo.py`：从 STDW summary/metadata 选择 `*_deploy.jit` 的部署管理示例。
- `eval/examples/thruster_io_demo.py`：8D action 到推进器命令的参考映射。
- `eval/examples/replay_csv_demo.py`：离线 replay/单步 obs-action 自检示例。

## 2. 导出可部署策略

在仿真机器上导出 TorchScript：

```bash
bash custom_workflows/run_with_isaac_env.sh workflows/play_stdw_adapt.py \
  --headless \
  --task EasyUUV-Direct-Parametric-v1 \
  --num_envs 1 \
  --experiment_name easyuuv_parametric \
  --logs_root logs/rsl_rl \
  --load_run 2026-06-08_13-48-14_stage2 \
  --checkpoint model_2398.pt \
  --workflow_config workflows/configs/matrix_wave_medium_full.yaml \
  --embodiment base \
  --use_stdw True \
  --total_steps 80 \
  --save_stdw_ckpt True \
  --stdw_ckpt_interval 40 \
  --export_deploy_jit True
```

将生成的 `stdw_step_*_deploy.jit` 拷贝到板载机器，并在 `eval/deploy_config.yaml` 中设置：

```yaml
policy:
  model_path: /absolute/path/to/stdw_step_001499_deploy.jit
  obs_layout: a3_12d
  device: cpu
```

## 3. 配置文件

`eval/deploy_config.yaml` 是实物部署的唯一参数入口。优先改 YAML，不要在 demo 脚本里散落硬编码。

关键字段：

- `serial.port`：串口，如 `/dev/ttyUSB0` 或 `COM8`。
- `serial.enabled`：默认 `false`，用于 dry-run。实物上板前改为 `true`。
- `control.control_dt`：控制周期，默认 `1/160` 秒。
- `control.steps_per_action`：每个策略动作保持的低层步数。
- `control.control_mode`：roll/pitch/yaw 使能。
- `control.action_limit_rpy`：策略动作映射到角度增量的上限。
- `controller.roll_zeta/pitch_zeta/yaw_zeta`：S 面控制器 `[ζ1, ζ2]` 初值。
- `stdw.min_real_samples`：慢环最小实物样本数，默认 64。
- `stdw.micro_probe`：observable-only 微扰识别参数。

## 4. 观测接口

当前 A3 观测固定为 12 维：

```text
[goal_quat(4), depth_z(1), root_quat(4), angular_velocity_b(3)]
```

硬件 bridge 必须提供 `eval/wrappers.py` 的 state dict：

```python
state = {
    "position": np.ndarray shape (3,),           # world, m
    "orientation_quat": np.ndarray shape (4,),   # w,x,y,z
    "linear_velocity_b": np.ndarray shape (3,),  # body, m/s
    "angular_velocity_b": np.ndarray shape (3,), # body, rad/s
    "goal_position": np.ndarray shape (3,),      # world, m
    "goal_yaw": float,                           # rad
}
```

注意：A3 当前不使用 `linear_velocity_b` 进入 12D obs，但 replay reward 与 legacy 10D 仍可能读取它，因此建议保持字段完整。

## 5. 动作接口

参数化策略输出 8 维：

```text
[u0, u1, u2, u3, a_gain0, a_gain1, a_gain2, a_gain3]
```

前 4 维是控制意图，后 4 维是增益调制通道。推进器映射只使用前 4 维；后 4 维进入低层控制器的增益调制/安全边界，不应直接映射为推进器命令。

`eval/examples/thruster_io_demo.py` 中的 `THRUSTER_ALLOC` 只是参考矩阵。上板前必须确认：

- 推进器编号；
- 每个推进器正反转符号；
- PWM/归一化命令范围；
- 急停命令；
- 饱和裁剪和死区。

## 6. 首次上板流程

1. 保持 `stdw.enable=false`，`serial.enabled=false`，先在开发机运行 dry-run：

```bash
PYTHONPATH=/path/to/direct python -m easyuuv_stdw.eval.examples.real_world_runtime \
  --config eval/deploy_config.yaml \
  --steps 5
```

2. 设置真实 `policy.model_path`，运行单步策略/推进器映射自检：

```bash
PYTHONPATH=/path/to/direct python -m easyuuv_stdw.eval.examples.thruster_io_demo \
  --config eval/deploy_config.yaml
```

3. 设置 `serial.port` 和 `serial.enabled=true`，保持 `stdw.enable=false`，只做短时闭环。

4. 检查日志中 `obs_tail_ang_vel` 是否量级合理。静止时陀螺噪声应接近 0，明显漂移时先做 IMU 标定或低层滤波。

5. 确认基线闭环稳定后，再开启 `stdw.micro_probe.enable=true` 或 `stdw.enable=true`。

## 7. STDW 少样本流程

建议顺序：

1. Baseline：STDW off，记录 100-200 step 的姿态/深度/动作。
2. Micro-probe：只做小幅可观测微扰，使用 `paired_axis` 评分。若证据不足，选择 `baseline`，遵循最小扰动原则。
3. Slow loop：满足 `stdw.min_real_samples` 后才启动慢环，优先 `slow_loop_interval=120`。
4. 回退：若 final window 误差升高或推进器饱和明显，立即关 `stdw.enable` 并恢复 baseline。

## 8. 安全项

- 实物运行前必须确认 `RealWorldEnv.halt()` 对应真实急停。
- `serial.enabled=false` 是默认值，防止误连硬件。
- 上板初期只启用 yaw 或单轴；不要一次性启用全部自由度。
- `action_limit_rpy` 应先保守，例如 yaw 不超过 `0.34 rad`。
- 任何策略输出必须先 clip 到 `[-1, 1]`。

## 9. 代码 TODO 索引

需要根据实物平台修改的关键位置：

- `eval/deploy_config.yaml`: `serial.port`, `policy.model_path`, `control.action_limit_rpy`, `controller.*_zeta`。
- `eval/examples/thruster_io_demo.py`: `THRUSTER_ALLOC` 推进器分配矩阵。
- `eval/examples/real_world_runtime.py`: ESP32 激活命令 `a`、急停 `e`、目标命令格式 `r..p..y..`。
- `eval/wrappers.py`: 若硬件输出四元数顺序不是 `w,x,y,z`，必须在 bridge 层修正。

## 10. 常见问题

`Policy loader` 报 RSL-RL checkpoint dict 不可调用：使用 `*_deploy.jit`，不要直接把 `model_2398.pt` 放到板载端。

导入失败 `No module named easyuuv_stdw`：设置 `PYTHONPATH` 到 `.../direct` 目录。

角速度噪声明显：先检查 IMU 标定和采样周期；若仿真实验确认低通有益，再在低层控制器对 ζ2 的 D 项启用低通。


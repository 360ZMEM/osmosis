# Evaluation SOP (Isaac-independent)

EasyUUV-STDW 的部署评估面向**没有 Isaac Sim 的目标主机**：板载 PC、CI runner、
科研 Linux 笔记本、ROS 节点。本 SOP 描述如何把 ckpt 部署到这些主机上做 replay 评估或在线推理。

> 上游入口：[`eval/README.md`](../../eval/README.md) | [`eval/`](../../eval/) 子模块

---

## 0. 依赖

| 后端 | 必装包 |
|---|---|
| `.pt` / `.jit` | `numpy`、`torch>=2.0` |
| `.onnx` | `numpy`、`onnxruntime>=1.16`（CPU）或 `onnxruntime-gpu` |

不需要 `omni.isaac.*`、`rsl_rl`、`gymnasium`（除非跑 `train_loop.py` demo）。

## 1. 状态字典契约

任何 host 数据桥（hardware bridge / log replay）必须产出：

```python
{
    "position":           np.ndarray (3,)  [m]    world frame
    "orientation_quat":   np.ndarray (4,)  [w,x,y,z]
    "linear_velocity_b":  np.ndarray (3,)  [m/s]  body frame
    "angular_velocity_b": np.ndarray (3,)  [rad/s] body frame
    "goal_position":      np.ndarray (3,)  [m]
    "goal_yaw":           float            [rad]
}
```

通过 `eval.wrappers.obs_from_state(state) -> np.ndarray (10,)` 转成 obs。

## 2. 模型导出

### 2.1 TorchScript（推荐 — 单文件、支持 GPU/CPU）

在训练完成后：

```python
import torch
policy = ...   # rsl_rl ActorCritic
ts = torch.jit.script(policy.actor)   # or trace
ts.save("ckpt.jit")
```

### 2.2 ONNX（推荐 — 跨语言、跨硬件）

```python
import torch
dummy = torch.zeros(1, 10)
torch.onnx.export(
    policy.actor, dummy, "ckpt.onnx",
    input_names=["obs"], output_names=["action"],
    opset_version=17,
)
```

### 2.3 Pickle `.pt`（最方便但最脆 — 仅同 Python 版本可用）

```python
torch.save(policy.actor, "ckpt.pt")
```

## 3. Replay-based offline 评估

```bash
python -m easyuuv_stdw.eval.deploy_eval \
    --policy /abs/path/to/ckpt.jit \
    --replay /abs/path/to/log.csv \
    --output ./.results/eval_out.csv
```

**replay CSV 必含列**：

```
t, px, py, pz, qw, qx, qy, qz, vx, vy, vz, wx, wy, wz, gx, gy, gz, gyaw
```

输出 CSV 在原列后追加 `a0..a3`（或 `a0..a7`）+ `reward` + `pos_err_norm`。

控制台 print 一份摘要：

| 字段 | 单位 | 含义 |
|---|---|---|
| `n_steps` | step | replay 长度 |
| `fmse_pos_m2` | m² | mean squared position error |
| `rmse_pos_m` | m | √fmse |
| `mean_reward` | — | 与训练 shaping 一致的 per-step reward |
| `action_dim` | — | 4 / 8 |

## 4. 在线推理 stub（伪代码）

```python
import numpy as np
from easyuuv_stdw.eval import obs_from_state, Policy

pol = Policy("/path/to/ckpt.jit", device="cpu")

def control_callback(state):
    obs = obs_from_state(state)            # (10,)
    action = pol.act(obs)                  # (4,) or (8,)
    cmd = thruster_mixer(action[:4])       # 用户自实现
    return cmd
```

## 5. CI sanity test（建议加进 GitHub Actions）

```bash
pip install numpy torch
python -m py_compile easyuuv_stdw/eval/*.py easyuuv_stdw/eval/examples/*.py
python easyuuv_stdw/eval/examples/replay_csv_demo.py --policy tests/data/dummy.jit
```

把一份 64KB 的 dummy.jit 提交到 `tests/data/`，确保 CI < 30 s 完成。

## 6. 性能指标单位规范

写部署报告必须严格使用以下单位（与 [`RESEARCH_ANALYSIS.md`](../principles/RESEARCH_ANALYSIS.md) 对齐）：

| 指标 | 单位 | 必须保留小数位 |
|---|---|---|
| `final_mse` | m² | 3 |
| `rmse_pos` | m | 3 |
| `target_drift` | m | 3 |
| `slow_loop_interval` | step | 整数 |
| `Δfmse %` | % | 1 |

## 7. 故障排查

| 症状 | 排查 |
|---|---|
| `RuntimeError: shape mismatch` 在 act() | 确认 `obs.shape == (10,)` 且 `dtype=float32` |
| ONNX session 不能初始化 | `pip install onnxruntime`，检查 opset ≥ 17 |
| `Policy.act` 返回全零 | ckpt 是否是训练前的 init weight；检查 `policy.eval()` 是否生效 |
| 4-D vs 8-D 混淆 | 看 `policy_loader.py` print 出的 `action.shape[0]`，必须匹配训练任务 |
| reward 数值与训练时报的差很多 | `obs_from_state` 的 yaw_err 是否 wrap 到 (-π, π] |

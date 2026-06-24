# 干净 Python 环境下跑通 STDW/eval 最小链路（Fossen 6-DOF 虚拟实物）

本教程回答一个很具体的问题：

> 如果我不在 IsaacLab 里，只拿到 EasyUUV-STDW 这个算法包/源码，我要准备什么，才能把“实物状态 -> 策略推理 -> 动作 -> 外部系统执行 -> replay 评估”这条链路跑通？

本教程不追求控制效果好，只验证链路完整。虚拟实物是一个手搓的简化 Fossen 6-DOF wrapper，放在被 `.gitignore` 忽略的目录：

```text
.results/clean_env_fossen_demo/
```

该目录是本地工作区样例，不作为正式源码发布；正式说明保存在本文档中。

---

## 0. 这条链路验证什么

本次验证的链路：

```text
Fossen6DofWrapper
  -> state dict
  -> easyuuv_stdw.eval.obs_from_state(layout="a3_12d")
  -> easyuuv_stdw.eval.Policy("*_deploy.jit")
  -> action shape=(8,)
  -> action[:4] 映射到虚拟力/力矩
  -> 6-DOF 状态积分
  -> replay CSV
  -> easyuuv_stdw.eval.deploy_eval 离线评估
```

验收标准：

- 不导入 Isaac / IsaacLab / `omni.*` / `rsl_rl`。
- 在一个新建 Python venv 中只安装 `numpy + torch + pyyaml` 即可运行。
- `obs_dim=12`，`action_dim=8`。
- 能输出 `fossen_replay.csv`、`deploy_eval_out.csv` 和 `chain_summary.json`。
- 不要求 dummy 策略效果好。

---

## 1. 文件清单

本地样例目录：

```text
.results/clean_env_fossen_demo/
├── requirements-clean.txt
├── make_dummy_policy.py
├── fossen6dof_wrapper.py
├── run_fossen_chain.py
└── run_clean_smoke.sh
```

各文件作用：

| 文件 | 作用 |
|---|---|
| `requirements-clean.txt` | 干净环境最小依赖：`numpy`、`torch`、`pyyaml`。 |
| `make_dummy_policy.py` | 生成一个 12D->8D 的 dummy TorchScript 策略，用于链路冒烟。 |
| `fossen6dof_wrapper.py` | 手搓简化 Fossen 6-DOF 虚拟实物，产出 `eval/wrappers.py` 要求的 state dict。 |
| `run_fossen_chain.py` | 将虚拟实物、`obs_from_state`、`Policy`、动作执行、replay CSV 和 `deploy_eval` 串起来。 |
| `run_clean_smoke.sh` | 一键创建干净 venv、安装依赖、生成 dummy 策略并跑通链路。 |

---

## 2. 小白版一键跑通

从仓库根目录运行：

```bash
cd /home/zmem063/isaaclab/source/extensions/omni.isaac.lab_tasks/omni/isaac/lab_tasks/direct/easyuuv_stdw
bash .results/clean_env_fossen_demo/run_clean_smoke.sh
```

脚本会自动做这些事：

1. 优先使用 `/usr/bin/python3` 创建 `.results/clean_env_fossen_demo/.venv-clean`；如果系统 Python 缺少 `venv`，会尝试 `/home/zmem063/anaconda3/bin/python`，再尝试当前 `python3`。
   - 这样可以避开当前 shell 里可能存在的 IsaacLab/conda `PYTHONPATH`。
   - 如需指定其它干净 Python：`CLEAN_PYTHON_BIN=/path/to/python bash .results/clean_env_fossen_demo/run_clean_smoke.sh`
   - 如果报 `ensurepip is not available`，Ubuntu/Debian 需要先安装 `python3-venv`，例如 `sudo apt install python3.12-venv`。
2. 安装 `requirements-clean.txt` 中的轻量依赖，并从 PyTorch CPU wheel index 安装 CPU-only `torch`。
3. 清理旧的 `PYTHONPATH/PYTHONHOME`，再只设置 `PYTHONPATH` 到 `.../omni/isaac/lab_tasks/direct`，让 Python 能导入 `easyuuv_stdw`。
4. 生成 `.results/clean_env_fossen_demo/dummy_a3_policy.jit`。
5. 运行 `.results/clean_env_fossen_demo/run_fossen_chain.py`。
6. 写出 `.results/clean_env_fossen_demo/out/` 下的结果文件。

成功时会打印 JSON，关键字段应类似：

```json
{
  "backend": "torchscript",
  "steps": 80,
  "obs_layout": "a3_12d",
  "obs_dim": 12,
  "action_dim": 8,
  "replay_csv": ".../fossen_replay.csv",
  "deploy_eval_csv": ".../deploy_eval_out.csv"
}
```

---

## 3. 手动一步步运行

如果想看清每一步，按下面命令手动执行。

### 3.1 进入仓库根

```bash
cd /home/zmem063/isaaclab/source/extensions/omni.isaac.lab_tasks/omni/isaac/lab_tasks/direct/easyuuv_stdw
```

### 3.2 创建干净 Python 环境

```bash
unset PYTHONPATH
unset PYTHONHOME

# 首选系统 Python；若系统缺 python3-venv，可改用一个干净 conda/base Python。
CLEAN_PYTHON_BIN=/usr/bin/python3
${CLEAN_PYTHON_BIN} -m venv --clear .results/clean_env_fossen_demo/.venv-clean
source .results/clean_env_fossen_demo/.venv-clean/bin/activate
python -m pip install --upgrade pip
```

如果上一步失败并提示 `ensurepip is not available`，先安装系统 venv 支持：

```bash
sudo apt install python3.12-venv
```

没有 sudo 权限时，可以指定一个不含 Isaac 路径的 Python，例如本机的 conda base：

```bash
CLEAN_PYTHON_BIN=/home/zmem063/anaconda3/bin/python \
bash .results/clean_env_fossen_demo/run_clean_smoke.sh
```

### 3.3 安装最小依赖

```bash
python -m pip install -r .results/clean_env_fossen_demo/requirements-clean.txt
python -m pip install --index-url https://download.pytorch.org/whl/cpu "torch>=2.2,<2.4"
```

当前最小依赖只有：

```text
numpy
pyyaml
torch（CPU-only wheel）
```

如果未来要走 ONNX 策略，再额外安装：

```bash
python -m pip install onnxruntime
```

### 3.4 设置包导入路径

当前仓库还不是标准 `pip install easyuuv_stdw` 形式，所以在干净环境中先用 `PYTHONPATH`：

```bash
export PYTHONPATH=/home/zmem063/isaaclab/source/extensions/omni.isaac.lab_tasks/omni/isaac/lab_tasks/direct:${PYTHONPATH:-}
```

检查能否导入：

```bash
python - <<'PY'
from easyuuv_stdw.eval import Policy, obs_from_state
print("IMPORT_OK", Policy, obs_from_state)
PY
```

### 3.5 生成 dummy 策略

```bash
python .results/clean_env_fossen_demo/make_dummy_policy.py \
  --output .results/clean_env_fossen_demo/dummy_a3_policy.jit
```

说明：

- 这是 12D->8D 的 TorchScript 策略。
- 它不是训练好的 STDW 策略。
- 只用于验证部署链路和接口维度。

如果已有真实导出的策略，例如 `stdw_step_001499_deploy.jit`，可以跳过本步骤，后续 `--policy` 改成真实路径。

### 3.6 跑 Fossen 虚拟实物链路

```bash
python .results/clean_env_fossen_demo/run_fossen_chain.py \
  --policy .results/clean_env_fossen_demo/dummy_a3_policy.jit \
  --steps 80 \
  --out-dir .results/clean_env_fossen_demo/out
```

输出：

```text
.results/clean_env_fossen_demo/out/
├── fossen_replay.csv
├── deploy_eval_out.csv
└── chain_summary.json
```

### 3.7 查看结果

```bash
cat .results/clean_env_fossen_demo/out/chain_summary.json
head .results/clean_env_fossen_demo/out/fossen_replay.csv
head .results/clean_env_fossen_demo/out/deploy_eval_out.csv
```

重点看：

- `obs_dim` 是否为 `12`
- `action_dim` 是否为 `8`
- `backend` 是否为 `torchscript`
- `deploy_eval_out.csv` 是否包含 `a0..a7`、`reward`、`pos_err_norm`

---

## 4. 我真正需要准备什么

把这个 dummy demo 换成真实 UUV 部署时，至少需要准备：

1. **可部署策略文件**
   - 推荐：`*_deploy.jit`
   - 不推荐：直接拿 RSL-RL 的 `model_*.pt` checkpoint dict
   - 原因：`eval.Policy` 需要可直接调用的 actor，而不是训练 checkpoint 状态字典。

2. **硬件 state dict bridge**
   - 必须产出 `eval/wrappers.py` 的 state dict：

```python
{
    "position":           np.ndarray shape (3,),  # 世界系，m
    "orientation_quat":   np.ndarray shape (4,),  # w,x,y,z
    "linear_velocity_b":  np.ndarray shape (3,),  # 机体系，m/s
    "angular_velocity_b": np.ndarray shape (3,),  # 机体系，rad/s
    "goal_position":      np.ndarray shape (3,),  # 世界系，m
    "goal_yaw":           float,                  # rad
}
```

3. **坐标系约定**
   - 四元数顺序：必须是 `w,x,y,z`
   - 深度方向：必须确认 `position[2]` 与训练时一致
   - 角速度：必须是机体系 `rad/s`
   - ENU/NED、IMU 安装方向、yaw 正方向都要实测确认。

4. **动作执行层**
   - 策略输出 8D：

```text
[u_surge, u_sway, u_heave, u_yaw, a_gain0, a_gain1, a_gain2, a_gain3]
```

   - `action[:4]` 是控制意图，可接低层控制器或推进器 mixer。
   - `action[4:]` 是增益调制，不应直接接推进器。

5. **安全保护**
   - 急停
   - watchdog
   - 串口/总线断连处理
   - NaN/Inf action 拦截
   - 推进器饱和统计
   - 单轴低幅试运行

6. **replay 日志**
   - 至少记录 `deploy_eval.py` 需要的列：

```text
t, px, py, pz, qw, qx, qy, qz, vx, vy, vz, wx, wy, wz, gx, gy, gz, gyaw
```

---

## 5. 这个 demo 与真实 Fossen 模型的边界

`.results/clean_env_fossen_demo/fossen6dof_wrapper.py` 只实现了一个简化链路模型：

- 有 6-DOF 状态 `eta=[x,y,z,roll,pitch,yaw]`、`nu=[u,v,w,p,q,r]`。
- 有对角质量、线性/二次阻尼、简化深度/横滚/俯仰恢复项。
- 有 `action[:4] -> tau` 的简化分配。
- 没有完整 Fossen 刚体/附加质量矩阵、科氏项、流体扰动、推进器动态和真实饱和。

所以它只能说明：

```text
软件包链路能跑通
```

不能说明：

```text
真实 UUV 控制效果已经可靠
```

---

## 6. 常见错误

### `ModuleNotFoundError: No module named 'easyuuv_stdw'`

没有设置 `PYTHONPATH`。执行：

```bash
export PYTHONPATH=/home/zmem063/isaaclab/source/extensions/omni.isaac.lab_tasks/omni/isaac/lab_tasks/direct:${PYTHONPATH:-}
```

### `Policy loader` 报 RSL-RL checkpoint dict 不可调用

你传的是 `model_*.pt` 训练 checkpoint，不是可部署 actor。需要先导出 `*_deploy.jit`。

### `shape mismatch`

当前主线策略使用 A3 12D 观测。确认：

```text
obs_layout = a3_12d
obs.shape = (12,)
```

### `deploy_eval_out.csv` 里没有 `a4..a7`

说明策略输出不是 8D。检查你是否用了 4D baseline 策略。

---

## 7. 后续可以升级的方向

1. 把仓库整理成标准 Python package，支持 `pip install -e .` 或 wheel 安装。
2. 将 `.results/clean_env_fossen_demo/` 中的样例迁移到正式 `examples/` 或单独 demo 仓库。
3. 把 dummy 策略替换成真实 `*_deploy.jit` 导出流程。
4. 将 Fossen wrapper 升级为更完整的 6-DOF 水动力模型。
5. 接 ROS2 / MAVLink / 串口硬件 bridge。
6. 为链路增加 pytest，固定 `obs_dim=12`、`action_dim=8`、CSV 字段和 `deploy_eval` 输出。

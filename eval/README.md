# `eval/` — Isaac-independent Evaluation Toolkit

This subpackage is the **deployment / replay-evaluation surface** of EasyUUV-STDW.
It deliberately does NOT import `omni.isaac.*`, `rsl_rl`, or the Isaac Lab
runtime, so any host with `numpy + torch` (or `onnxruntime`) can run it —
shipboard PCs, lab benches, or unit-test runners.

For the full evaluation SOP (CI gates, replay log spec, performance units),
see [`docs/guide/EVAL_SOP.md`](../docs/guide/EVAL_SOP.md).

## Modules

| Module | Purpose |
|---|---|
| `wrappers.py` | `obs_from_state(state)` / `reward_from_state(state, action)`; defines the 10-D obs contract and reference shaping reward. |
| `policy_loader.py` | `Policy(path).act(obs)`; auto-dispatches `.pt` / `.jit` / `.onnx`. |
| `train_loop.py` | ~50-line reference PPO loop (no Isaac dep) for sanity-checking the obs/reward contract. |
| `deploy_eval.py` | CLI: replay a state-log CSV through a trained policy, write per-step actions/reward, print fmse/rmse summary. |
| `examples/replay_csv_demo.py` | Minimal end-to-end usage of `Policy` + `obs_from_state`. |
| `examples/thruster_io_demo.py` | Demonstrates 4-D vs 8-D action layout and how to map to thruster commands. |

## State-dict contract

Any caller building obs from real telemetry must produce this dict:

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

Observation layout produced by `obs_from_state`:

```
[ pos_err_x, pos_err_y, pos_err_z,
  yaw_err,
  lin_vel_bx, lin_vel_by, lin_vel_bz,
  ang_vel_bx, ang_vel_by, ang_vel_bz ]   shape=(10,) float32
```

## Quick start

```bash
# 1) Load a TorchScript policy and run a single inference.
python eval/examples/replay_csv_demo.py --policy ckpt.jit

# 2) Full replay-eval over a CSV log (writes eval_out.csv).
python -m easyuuv_stdw.eval.deploy_eval \
    --policy /path/to/policy.pt \
    --replay /path/to/log.csv \
    --output ./.results/eval_out.csv
```

The CLI prints a JSON-style summary; key fields:

| Key | Unit | Meaning |
|---|---|---|
| `fmse_pos_m2` | m² | mean squared position error |
| `rmse_pos_m` | m | √fmse |
| `mean_reward` | — | mean per-step shaping reward |
| `action_dim` | — | 4 (baseline) or 8 (parametric / a_gain) |

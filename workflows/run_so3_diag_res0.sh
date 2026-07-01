#!/usr/bin/env bash
# SO(3) diagnostic: pure analytic geometric controller (geo_residual_scale=0),
# loading model_2846 so the depth channel stays at its known-good euler baseline.
# Isolates the SO(3) S-surface attitude controller quality from the chaotic fine-tune.
set -uo pipefail

REPO="/home/zmem063/isaaclab/source/extensions/omni.isaac.lab_tasks/omni/isaac/lab_tasks/direct/easyuuv_stdw"
RUNNER="$REPO/custom_workflows/run_with_isaac_env.sh"
PLAY="$REPO/workflows/play_stdw_adapt.py"
LOGS_ROOT="$REPO/logs/rsl_rl"
LOAD_RUN="2026-06-29_23-50-50_flip360_curric_b_full_pi_stage0"
CKPT="model_2846.pt"
ROOT="$REPO/.results/so3_diag_res0_2846"
mkdir -p "$ROOT"

run_cell () {
  local name="$1" cfg="$2" emb="$3"
  local cdir="$ROOT/$name"
  mkdir -p "$cdir"
  echo "=== [$name] cfg=$cfg emb=$emb ==="
  bash "$RUNNER" "$PLAY" \
    --headless \
    --task EasyUUV-Direct-Parametric-v1 \
    --num_envs 1 \
    --experiment_name easyuuv_parametric \
    --logs_root "$LOGS_ROOT" \
    --load_run "$LOAD_RUN" \
    --checkpoint "$CKPT" \
    --workflow_config "$cfg" \
    --total_steps 1500 \
    --seed 0 \
    --embodiment "$emb" \
    --auto_drift_router False \
    --drift_router_mode off \
    --enable_micro_probe False \
    --enable_trigger_gate True \
    --trigger_threshold 0.05 \
    --use_stdw False \
    --target_drift 0.0 \
    --results_root "$cdir/results" \
    --artifacts_root "$cdir/artifacts" \
    > "$cdir/run.log" 2>&1
  echo "[$name] returncode=$?"
}

run_cell flip360_base   workflows/configs/pressure_flip360_medium_so3_res0.yaml base
run_cell flip360_asym   workflows/configs/pressure_flip360_medium_so3_res0.yaml asymmetric
run_cell ordinary_base  workflows/configs/matrix_wave_medium_so3_res0.yaml     base
run_cell ordinary_asym  workflows/configs/matrix_wave_medium_so3_res0.yaml     asymmetric

echo "=== DIAG DONE ==="

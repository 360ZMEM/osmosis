#!/usr/bin/env bash
# SO(3) Phase 1 two-objective eval for checkpoint model_2995.
# Mirrors sweep_stdw_safety_pressure.py off_clean flags but uses SO(3)-mode configs
# so the controller path matches the trained policy.
set -uo pipefail

REPO="/home/zmem063/isaaclab/source/extensions/omni.isaac.lab_tasks/omni/isaac/lab_tasks/direct/easyuuv_stdw"
RUNNER="$REPO/custom_workflows/run_with_isaac_env.sh"
PLAY="$REPO/workflows/play_stdw_adapt.py"
LOGS_ROOT="$REPO/logs/rsl_rl"
LOAD_RUN="2026-07-01_20-15-47_flip360_so3_p1_stage0"
CKPT="${CKPT:-model_2995.pt}"
TAG="${CKPT%.pt}"
ROOT="$REPO/.results/so3_p1_eval_${TAG#model_}"
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

run_cell flip360_base     workflows/configs/pressure_flip360_medium_so3.yaml base
run_cell flip360_asym     workflows/configs/pressure_flip360_medium_so3.yaml asymmetric
run_cell ordinary_base    workflows/configs/matrix_wave_medium_so3.yaml     base
run_cell ordinary_asym    workflows/configs/matrix_wave_medium_so3.yaml     asymmetric

echo "=== ALL CELLS DONE ==="
for name in flip360_base flip360_asym ordinary_base ordinary_asym; do
  s=$(ls -t "$ROOT/$name"/results/**/summary.json 2>/dev/null | head -1)
  [ -z "$s" ] && s=$(find "$ROOT/$name/results" -name summary.json -print 2>/dev/null | head -1)
  if [ -n "$s" ]; then
    mse=$(python3 -c "import json,sys; print(json.load(open('$s')).get('final_mse'))" 2>/dev/null)
    echo "$name: final_mse=$mse ($s)"
  else
    echo "$name: NO summary.json"
  fi
done

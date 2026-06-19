#!/usr/bin/env bash
# STDW online-handling side experiment: a deliberately mis-tuned initial
# controller (depth P gain halved to 0.5x, the 1/2x boundary, which is exactly
# the gain STDW's slow loop adapts online) is run under the original stage2
# checkpoint with STDW off vs on, for base + heavy_moderate.
# Shared deterministic reference so off/on overlays are like-for-like.
set -euo pipefail

REPO_ROOT="/home/zmem063/isaaclab/source/extensions/omni.isaac.lab_tasks/omni/isaac/lab_tasks/direct/easyuuv_stdw"
RUNNER="${REPO_ROOT}/custom_workflows/run_with_isaac_env.sh"
WORKFLOW="${REPO_ROOT}/workflows/play_stdw_adapt.py"
LOGS_ROOT="${REPO_ROOT}/logs/rsl_rl"
CFG="${REPO_ROOT}/workflows/configs/matrix_wave_storm_full.yaml"

OUT_ROOT="${REPO_ROOT}/.results/stdw_online_misset_20260620"
MISSET='{"depth_zeta1": 0.5}'

mkdir -p "${OUT_ROOT}"
echo "[INFO] mis-set = ${MISSET}" | tee "${OUT_ROOT}/MISSET.txt"

run_cell () {
  local emb="$1" stdw="$2" use_stdw target
  if [ "${stdw}" = "on" ]; then use_stdw="True"; target="0.05"; else use_stdw="False"; target="0.0"; fi
  local cell="${emb}_${stdw}"
  local cell_dir="${OUT_ROOT}/${cell}"
  mkdir -p "${cell_dir}"
  echo "=== [RUN] ${cell} (use_stdw=${use_stdw} target=${target}) ==="
  "${RUNNER}" "${WORKFLOW}" \
    --headless \
    --task EasyUUV-Direct-Parametric-v1 \
    --num_envs 1 \
    --experiment_name easyuuv_parametric \
    --logs_root "${LOGS_ROOT}" \
    --load_run 2026-06-08_13-48-14_stage2 \
    --checkpoint model_2398.pt \
    --workflow_config "${CFG}" \
    --embodiment "${emb}" \
    --use_stdw "${use_stdw}" \
    --target_drift "${target}" \
    --enable_trigger_gate True \
    --trigger_threshold 0.05 \
    --deterministic_reference True \
    --total_steps 1500 \
    --seed 0 \
    --pid_multipliers "${MISSET}" \
    --results_root "${cell_dir}/results" \
    --artifacts_root "${cell_dir}/artifacts" \
    > "${cell_dir}/run.log" 2>&1
  echo "=== [DONE] ${cell} rc=$? ==="
}

for emb in base heavy_moderate; do
  for stdw in off on; do
    run_cell "${emb}" "${stdw}"
  done
done

echo "[ALL DONE] outputs under ${OUT_ROOT}"

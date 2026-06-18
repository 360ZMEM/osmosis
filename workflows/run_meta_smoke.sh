#!/usr/bin/env bash
# 8 维元控制端到端 smoke launcher（实施计划 §3.7）。
#
# 三段式：
#   1) 5-iter PPO 短训（64 envs），验证维度链路与 RSL-RL 串通。
#   2) 200 步无漂移 eval：检查 ζ 时间轴 CSV 列齐全且 ζ ≈ ζ_nominal。
#   3) 800 步 COB drift eval（y 轴 0.05 m）：可观察到 ζ 在漂移区间被网络调整。
#
# 落盘根目录：<repo>/.tmp/meta_smoke_<timestamp>/

set -uo pipefail

REPO_ROOT="/home/zmem063/isaaclab/source/extensions/omni.isaac.lab_tasks/omni/isaac/lab_tasks/direct/easyuuv_stdw"
TS="$(date +%Y%m%d_%H%M%S)"
BASE="${REPO_ROOT}/.tmp/meta_smoke_${TS}"
mkdir -p "${BASE}"
echo "[meta-smoke] base dir = ${BASE}"

cd "${REPO_ROOT}"

# ---------------------------------------------------------------------------
# Phase 1: 5-iter 短训
# ---------------------------------------------------------------------------
TRAIN_LOG="${BASE}/phase1_train.log"
TRAIN_RESULTS="${BASE}/phase1_results"
TRAIN_LOGS="${BASE}/phase1_logs"
mkdir -p "${TRAIN_RESULTS}" "${TRAIN_LOGS}"

echo "==================================================================="
echo "[meta-smoke] phase 1: 5-iter PPO short training (num_envs=64)"
echo "[meta-smoke] log = ${TRAIN_LOG}"
echo "==================================================================="
bash "${REPO_ROOT}/custom_workflows/run_with_isaac_env.sh" \
  "${REPO_ROOT}/workflows_new_stdw/train_meta.py" \
  --headless --cpu \
  --task EasyUUV-Direct-Parametric-v1 \
  --num_envs 64 \
  --max_iterations 5 \
  --logs_root "${TRAIN_LOGS}" \
  --results_root "${TRAIN_RESULTS}" \
  > "${TRAIN_LOG}" 2>&1
echo "[meta-smoke] phase 1 rc=$?"

# ---------------------------------------------------------------------------
# Phase 2: 200 步无漂移 eval
# ---------------------------------------------------------------------------
EVAL1_DIR="${BASE}/phase2_eval_no_drift"
mkdir -p "${EVAL1_DIR}"
EVAL1_LOG="${EVAL1_DIR}/run.log"
echo "==================================================================="
echo "[meta-smoke] phase 2: 200-step eval, no COB drift"
echo "[meta-smoke] log = ${EVAL1_LOG}"
echo "==================================================================="
bash "${REPO_ROOT}/custom_workflows/run_with_isaac_env.sh" \
  "${REPO_ROOT}/workflows_new_stdw/play_meta_eval.py" \
  --headless --cpu \
  --task EasyUUV-Direct-Parametric-v1 \
  --num_envs 1 \
  --steps 200 \
  --cob_drift_magnitude 0.0 \
  --save_dir "${EVAL1_DIR}" \
  --logs_root "${TRAIN_LOGS}" \
  --results_root "${EVAL1_DIR}/results" \
  > "${EVAL1_LOG}" 2>&1
echo "[meta-smoke] phase 2 rc=$?"

# ---------------------------------------------------------------------------
# Phase 3: 800 步 COB drift eval（y 轴 0.05 m）
# ---------------------------------------------------------------------------
EVAL2_DIR="${BASE}/phase3_eval_cob_drift_y_005"
mkdir -p "${EVAL2_DIR}"
EVAL2_LOG="${EVAL2_DIR}/run.log"
echo "==================================================================="
echo "[meta-smoke] phase 3: 800-step eval, COB drift y=+0.05 m"
echo "[meta-smoke] log = ${EVAL2_LOG}"
echo "==================================================================="
bash "${REPO_ROOT}/custom_workflows/run_with_isaac_env.sh" \
  "${REPO_ROOT}/workflows_new_stdw/play_meta_eval.py" \
  --headless --cpu \
  --task EasyUUV-Direct-Parametric-v1 \
  --num_envs 1 \
  --steps 800 \
  --cob_drift_axis y \
  --cob_drift_magnitude 0.05 \
  --cob_drift_start_step 200 \
  --cob_drift_end_step 800 \
  --save_dir "${EVAL2_DIR}" \
  --logs_root "${TRAIN_LOGS}" \
  --results_root "${EVAL2_DIR}/results" \
  > "${EVAL2_LOG}" 2>&1
echo "[meta-smoke] phase 3 rc=$?"

echo "[meta-smoke] DONE; results under ${BASE}"

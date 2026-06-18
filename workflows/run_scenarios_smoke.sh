#!/usr/bin/env bash
# STDW scenario / embodiment smoke launcher (600 steps each, sequential).
# 对应 plan §5 verification step 3+4：跑通 7 个 non-none scenario × embodiment=base
# 以及 scenario=none × 4 embodiment 变体。
# 落盘根目录：<repo>/.tmp/stdw_scenarios_smoke_<timestamp>/
set -uo pipefail

REPO_ROOT="/home/zmem063/isaaclab/source/extensions/omni.isaac.lab_tasks/omni/isaac/lab_tasks/direct/easyuuv_stdw"
TS="$(date +%Y%m%d_%H%M%S)"
BASE="${REPO_ROOT}/.tmp/stdw_scenarios_smoke_${TS}"
mkdir -p "${BASE}"
echo "[launcher] base dir = ${BASE}"

cd "${REPO_ROOT}"

# 索引 CSV 表头
echo "name,scenario,embodiment,returncode,started_at,ended_at" > "${BASE}/index.csv"

run_one() {
  local name="$1"
  local scenario="$2"
  local embodiment="$3"
  local out_dir="${BASE}/${name}"
  mkdir -p "${out_dir}"
  local log="${out_dir}/run.log"
  local started ended rc
  started="$(date -Iseconds)"

  echo "==================================================================="
  echo "[launcher] >>> ${name}  scenario=${scenario}  embodiment=${embodiment}"
  echo "[launcher] log  = ${log}"
  echo "[launcher] start at ${started}"
  echo "==================================================================="

  bash "${REPO_ROOT}/custom_workflows/run_with_isaac_env.sh" \
    "${REPO_ROOT}/workflows_new_stdw/play_stdw_adapt.py" \
    --headless --cpu \
    --task EasyUUV-Direct-v1 \
    --num_envs 1 \
    --load_run SS4 \
    --checkpoint model_500.pt \
    --total_steps 600 \
    --use_stdw True \
    --enable_filter True \
    --use_quantile_filter True \
    --scenario "${scenario}" \
    --embodiment "${embodiment}" \
    --results_root "${out_dir}/results" \
    --artifacts_root "${out_dir}/artifacts" \
    > "${log}" 2>&1
  rc=$?
  ended="$(date -Iseconds)"
  echo "[launcher] end   at ${ended}  rc=${rc}"
  echo "${name},${scenario},${embodiment},${rc},${started},${ended}" >> "${BASE}/index.csv"
}

# Phase 1: 7 个 non-none scenario × embodiment=base
run_one "sc_sine"               "sine"                 "base"
run_one "sc_current_bias"       "current_bias"         "base"
run_one "sc_jonswap_mild"       "jonswap_mild"         "base"
run_one "sc_jonswap_strong"     "jonswap_strong"       "base"
run_one "sc_current_jonswap"    "current_plus_jonswap" "base"
run_one "sc_wave_plus_noise"    "wave_plus_noise"      "base"
run_one "sc_wave_plus_fault"    "wave_plus_fault"      "base"

# Phase 2: scenario=none × 4 embodiment 变体（不要求结果质量）
run_one "emb_base"              "none"                 "base"
run_one "emb_long_body"         "none"                 "long_body"
run_one "emb_heavy_moderate"    "none"                 "heavy_moderate"
run_one "emb_asymmetric"        "none"                 "asymmetric"

echo "[launcher] all done. index = ${BASE}/index.csv"

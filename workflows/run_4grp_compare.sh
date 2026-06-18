#!/usr/bin/env bash
# STDW v3 four-arm comparison launcher (1400 steps each, sequential).
# 对应 plan §5 4 组对照最小集。
# 落盘根目录在工作区内：<repo>/.tmp/stdw_4grp_<timestamp>/
#
# NOTE (2026-06-04): 本脚本只跑 play_stdw_adapt.py（单组 4 跑），不依赖
# sweep_stdw.py 的矩阵开关，所以不会受 DEFAULT_MATRIX 由"算法网格"切换为
# "场景×机型网格"的影响。如需复现 sweep8 / sweep72 的算法网格，请改用：
#     python workflows_new_stdw/sweep_stdw.py --algo_grid --full_matrix \
#         --total_steps 1400 --csv_out logs/stdw_algo_grid.csv
# 必须显式加 --algo_grid，否则会跑成新场景×机型矩阵。
set -uo pipefail

REPO_ROOT="/home/zmem063/isaaclab/source/extensions/omni.isaac.lab_tasks/omni/isaac/lab_tasks/direct/easyuuv_stdw"
TS="$(date +%Y%m%d_%H%M%S)"
BASE="${REPO_ROOT}/.tmp/stdw_4grp_${TS}"
mkdir -p "${BASE}"
echo "[launcher] base dir = ${BASE}"

cd "${REPO_ROOT}"

# 索引 CSV 表头
echo "name,use_stdw,enable_filter,use_quantile_filter,returncode,started_at,ended_at" \
  > "${BASE}/index.csv"

run_one() {
  local name="$1"
  local use_stdw="$2"
  local enable_filter="$3"
  local use_quantile_filter="$4"
  local out_dir="${BASE}/${name}"
  mkdir -p "${out_dir}"
  local log="${out_dir}/run.log"
  local started ended rc
  started="$(date -Iseconds)"

  echo "==================================================================="
  echo "[launcher] >>> ${name}  use_stdw=${use_stdw}  filter=${enable_filter}  quantile=${use_quantile_filter}"
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
    --total_steps 1400 \
    --use_stdw "${use_stdw}" \
    --enable_filter "${enable_filter}" \
    --use_quantile_filter "${use_quantile_filter}" \
    --results_root "${out_dir}/results" \
    --artifacts_root "${out_dir}/artifacts" \
    >"${log}" 2>&1
  rc=$?
  ended="$(date -Iseconds)"

  echo "[launcher] <<< ${name} exited rc=${rc} at ${ended}"
  echo "${name},${use_stdw},${enable_filter},${use_quantile_filter},${rc},${started},${ended}" \
    >> "${BASE}/index.csv"
  return ${rc}
}

# 4 组对照（plan §5 最小集）
run_one baseline     False False False
run_one stdw_only    True  False False
run_one stdw_filter  True  True  False
run_one stdw_full    True  True  True

echo "[launcher] all four arms complete; index = ${BASE}/index.csv"

#!/usr/bin/env bash
set -uo pipefail

# Stable local launcher for this workstation.  It avoids two common blockers:
# 1. zsh sourcing Isaac setup scripts without BASH_SOURCE;
# 2. the default pytorch conda environment shadowing Isaac Lab dependencies.
source /home/zmem063/anaconda3/etc/profile.d/conda.sh
conda activate isaaclab
set -e
source /home/zmem063/isaacsim/setup_python_env.sh

export PYTHONPATH="/home/zmem063/isaacsim/exts/omni.isaac.core:/home/zmem063/isaaclab/source/extensions/omni.isaac.lab:/home/zmem063/isaaclab/source/extensions/omni.isaac.lab_tasks:/home/zmem063/isaaclab/source/extensions/omni.isaac.lab_assets:${PYTHONPATH:-}"

exec python "$@"

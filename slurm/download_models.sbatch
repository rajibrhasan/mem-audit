#!/usr/bin/env bash
# Download a pinned backbone checkpoint (CPU job). For Gemma, also build its own
# vLLM environment: Gemma 4 needs vLLM 0.24.0 with Transformers 5.7.0.
#
#   sbatch --export=ALL,MEMAUDIT_MODEL=gemma slurm/download_models.sbatch
#
#SBATCH --job-name=memaudit-download
#SBATCH --partition=hpg-default
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=02:00:00
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?Submit from the repository root}"
source slurm/environment.sh
model="${MEMAUDIT_MODEL:?Set MEMAUDIT_MODEL to llama, gpt-oss or gemma}"
"$MEMAUDIT_PYTHON" -u scripts/download_models.py "$model"
if [[ "$model" == gemma ]]; then
    uv="${MEMAUDIT_UV:-uv}"
    venv="$MEMAUDIT_STORAGE/gemma-serving-venv"
    if [[ ! -x "$venv/bin/python" ]]; then
        "$uv" venv --python "${MEMAUDIT_BASE_PYTHON:-python3.11}" "$venv"
    fi
    "$uv" pip install --python "$venv/bin/python" 'vllm==0.24.0' 'transformers==5.7.0'
fi

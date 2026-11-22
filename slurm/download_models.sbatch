#!/usr/bin/env bash
# Prepare one backbone (CPU job): download its checkpoint at the pinned revision
# and build its vLLM serving environment at the version used for the reported runs
# (vLLM 0.19.0 for Llama and GPT-OSS; vLLM 0.24.0 with Transformers 5.7.0 for Gemma).
#
#   sbatch --export=ALL,MEMAUDIT_MODEL=gemma slurm/download_models.sbatch
#
# Llama and Gemma are gated: accept their licenses on Hugging Face and export HF_TOKEN first.
#SBATCH --job-name=memaudit-download
#SBATCH --partition=hpg-default
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=04:00:00
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?Submit from the repository root}"
source slurm/environment.sh
model="${MEMAUDIT_MODEL:?Set MEMAUDIT_MODEL to llama, gpt-oss or gemma}"
"$MEMAUDIT_PYTHON" -u scripts/download_models.py "$model"

case "$model" in
    gemma) packages=('vllm==0.24.0' 'transformers==5.7.0') ;;
    *)     packages=('vllm==0.19.0') ;;
esac
uv="${MEMAUDIT_UV:-uv}"
venv="$MEMAUDIT_STORAGE/serving/$model"
if [[ ! -x "$venv/bin/python" ]]; then
    "$uv" venv --python "${MEMAUDIT_BASE_PYTHON:-python3.11}" "$venv"
fi
"$uv" pip install --python "$venv/bin/python" "${packages[@]}"
echo "Ready: $model checkpoint and $venv/bin/vllm"

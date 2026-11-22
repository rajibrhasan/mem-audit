#!/usr/bin/env bash
# Serve one backbone with vLLM on 127.0.0.1, from the checkpoint prepared by
# scripts/download_models.py. Runs in the foreground.
#   bash slurm/serve_vllm.sh {llama|gpt-oss|gemma}
# Settings: MEMAUDIT_PORT (default 8000), MEMAUDIT_CONTEXT (default 32768),
# MEMAUDIT_WORKERS (server batch size; match the run's --workers), and
# MEMAUDIT_VLLM (vLLM executable; Gemma defaults to the environment that
# slurm/download_models.sbatch builds).
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
source slurm/environment.sh

model="${1:?Usage: bash slurm/serve_vllm.sh {llama|gpt-oss|gemma}}"
vllm="${MEMAUDIT_VLLM:-vllm}"
cuda=cuda/12.8.1
extra=()
case "$model" in
    llama)
        directory=llama31-8b-instruct
        served=meta-llama/Meta-Llama-3.1-8B-Instruct ;;
    gpt-oss)
        directory=gpt-oss-20b
        served=openai/gpt-oss-20b
        extra=(--reasoning-parser openai_gptoss) ;;
    gemma)
        directory=gemma4-31b-it
        served=google/gemma-4-31B-it
        cuda=cuda/13.0.2
        vllm="${MEMAUDIT_VLLM:-$MEMAUDIT_STORAGE/gemma-serving-venv/bin/vllm}"
        extra=(--reasoning-parser gemma4 --language-model-only) ;;
    *)
        echo "Unknown model: $model" >&2
        exit 2 ;;
esac

# Load the CUDA runtime where environment modules are available.
if type module >/dev/null 2>&1 || source /etc/profile.d/modules.sh 2>/dev/null; then
    module load "$cuda"
fi
vllm="$(command -v "$vllm")"
# Use the executable's own virtual environment, if it has one.
if [[ -f "$(dirname "$vllm")/activate" ]]; then
    source "$(dirname "$vllm")/activate"
fi
export HF_HUB_OFFLINE=1
export VLLM_CACHE_ROOT="$XDG_CACHE_HOME/vllm"
export TRITON_CACHE_DIR="$XDG_CACHE_HOME/triton"
export TORCHINDUCTOR_CACHE_DIR="$XDG_CACHE_HOME/inductor"
export FLASHINFER_WORKSPACE_BASE="$XDG_CACHE_HOME/flashinfer/$model"
export TORCH_EXTENSIONS_DIR="$XDG_CACHE_HOME/torch-extensions/$model"

exec "$vllm" serve "$MEMAUDIT_STORAGE/models/$directory" --served-model-name "$served" \
    --host 127.0.0.1 --port "${MEMAUDIT_PORT:-8000}" --max-model-len "${MEMAUDIT_CONTEXT:-32768}" \
    --dtype bfloat16 --gpu-memory-utilization 0.90 --max-num-seqs "${MEMAUDIT_WORKERS:-1}" \
    --enforce-eager "${extra[@]}"

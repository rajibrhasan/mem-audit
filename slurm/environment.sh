#!/usr/bin/env bash
# Source this file. It keeps the Python environment, model weights and every cache
# under one storage directory. Set MEMAUDIT_STORAGE before sourcing to move it
# (use scratch space: the checkpoints alone take tens of gigabytes).
export MEMAUDIT_STORAGE="${MEMAUDIT_STORAGE:-$HOME/memaudit}"
export MEMAUDIT_PYTHON="$MEMAUDIT_STORAGE/venv/bin/python"
export UV_CACHE_DIR="$MEMAUDIT_STORAGE/uv-cache"
export PIP_CACHE_DIR="$MEMAUDIT_STORAGE/pip-cache"
export HF_HOME="$MEMAUDIT_STORAGE/huggingface"
export HF_HUB_CACHE="$HF_HOME/hub"
export HUGGINGFACE_HUB_CACHE="$HF_HUB_CACHE"
export SENTENCE_TRANSFORMERS_HOME="$HF_HUB_CACHE"
unset TRANSFORMERS_CACHE  # an inherited legacy setting would redirect the model cache
export XDG_CACHE_HOME="$MEMAUDIT_STORAGE/cache"
export TORCH_HOME="$MEMAUDIT_STORAGE/torch"
export MEM0_DIR="$MEMAUDIT_STORAGE/mem0-config"
export TMPDIR="$MEMAUDIT_STORAGE/tmp"
export MEM0_TELEMETRY=false
export ANONYMIZED_TELEMETRY=false
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-2}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-2}"
mkdir -p "$MEMAUDIT_STORAGE" "$TMPDIR" "$XDG_CACHE_HOME"

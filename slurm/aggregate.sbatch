#!/usr/bin/env bash
# CPU-only steps after the shard jobs finish. Submit from the repository root:
#
#   sbatch --export=ALL,MEMAUDIT_MODEL=gemma slurm/aggregate.sbatch collect           # merge one model's shards
#   sbatch slurm/aggregate.sbatch compare                                             # compare the three models
#   sbatch slurm/aggregate.sbatch export                                              # metrics table + figures
#   sbatch --export=ALL,MEMAUDIT_MODEL=gemma slurm/aggregate.sbatch collect-ablation  # reasoning ablation
#
#SBATCH --job-name=memaudit-aggregate
#SBATCH --partition=hpg-default
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=1-00:00:00
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?Submit from the repository root}"
source slurm/environment.sh
runs="$MEMAUDIT_STORAGE/runs"
python="$MEMAUDIT_PYTHON"

case "${1:?Usage: sbatch slurm/aggregate.sbatch {collect|compare|export|collect-ablation}}" in
    collect)
        model="${MEMAUDIT_MODEL:?Set MEMAUDIT_MODEL}"
        "$python" -u -m memory_audit collect --config "configs/full-runs/$model.json" --output "$runs/$model" ;;
    compare)
        "$python" -u -m memory_audit compare --run llama "$runs/llama" --run gpt-oss "$runs/gpt-oss" \
            --run gemma "$runs/gemma" --output "$runs/comparison" ;;
    export)
        "$python" -u scripts/export_metrics.py --comparison "$runs/comparison"
        "$python" -u scripts/make_figures.py --metrics "$runs/comparison/comparison-metrics.json" \
            --out "$runs/comparison/figures" ;;
    collect-ablation)
        model="${MEMAUDIT_MODEL:?Set MEMAUDIT_MODEL}"
        "$python" -u -m memory_audit collect-ablation --config "configs/ablation/$model.json" \
            --baseline "$runs/$model" --output "$runs/$model-ablation" ;;
    *)
        echo "Unknown step: $1" >&2
        exit 2 ;;
esac

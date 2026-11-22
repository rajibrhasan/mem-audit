#!/usr/bin/env bash
# Run one shard (one memory system x one seed) on one GPU: start the vLLM server,
# wait until it is ready, then run the cases. Submit from the repository root as
# an array, one task per shard (configs/full-runs/<model>/shard-<task>.json):
#
#   sbatch --array=0-5 --export=ALL,MEMAUDIT_MODEL=llama slurm/run_shard.sbatch
#
# Reasoning ablation: also set MEMAUDIT_BASELINE to the model's finished main run.
# Each task then re-answers the baseline shard with the same index:
#
#   sbatch --array=0,3 --export=ALL,MEMAUDIT_MODEL=gemma,MEMAUDIT_BASELINE=$MEMAUDIT_STORAGE/runs/gemma \
#       slurm/run_shard.sbatch
#
# Optional: MEMAUDIT_WORKERS (cases in parallel; default 1), MEMAUDIT_RUN_ROOT.
# Add --account/--partition/--gres at submission to match your cluster and model size.
#SBATCH --job-name=memaudit-shard
#SBATCH --partition=hpg-turin
#SBATCH --gres=gpu:l4:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=14-00:00:00
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?Submit from the repository root}"
source slurm/environment.sh

model="${MEMAUDIT_MODEL:?Set MEMAUDIT_MODEL to llama, gpt-oss or gemma}"
shard="${SLURM_ARRAY_TASK_ID:?Submit as an array job}"
if [[ -n "${MEMAUDIT_BASELINE:-}" ]]; then
    run_root="${MEMAUDIT_RUN_ROOT:-$MEMAUDIT_STORAGE/runs/$model-ablation}"
else
    run_root="${MEMAUDIT_RUN_ROOT:-$MEMAUDIT_STORAGE/runs/$model}"
fi
mkdir -p "$run_root/logs"

# Serve the model on a job-specific port; stop the server when the job ends.
export MEMAUDIT_PORT=$((20000 + SLURM_JOB_ID % 40000))
export MEMAUDIT_BASE_URL="http://127.0.0.1:$MEMAUDIT_PORT/v1"
bash slurm/serve_vllm.sh "$model" > "$run_root/logs/server-$shard-$SLURM_JOB_ID.log" 2>&1 &
server=$!
trap 'kill "$server" 2>/dev/null || true; wait "$server" 2>/dev/null || true' EXIT
trap 'exit 143' TERM INT

# Wait up to one hour for the model to load.
for _ in $(seq 720); do
    if curl -sf "$MEMAUDIT_BASE_URL/models" >/dev/null; then
        break
    fi
    if ! kill -0 "$server" 2>/dev/null; then
        echo "vLLM server exited; see $run_root/logs" >&2
        exit 1
    fi
    sleep 5
done
curl -sf "$MEMAUDIT_BASE_URL/models" >/dev/null || { echo "vLLM server did not become ready" >&2; exit 1; }

workers="${MEMAUDIT_WORKERS:-1}"
if [[ -n "${MEMAUDIT_BASELINE:-}" ]]; then
    "$MEMAUDIT_PYTHON" -u -m memory_audit ablate --baseline "$MEMAUDIT_BASELINE/shards/$shard" \
        --output "$run_root/shards/$shard" --workers "$workers"
else
    "$MEMAUDIT_PYTHON" -u -m memory_audit run --config "configs/full-runs/$model/shard-$shard.json" \
        --output "$run_root/shards/$shard" --workers "$workers"
fi

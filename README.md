# Auditing Persistent-Memory Deletion in LLM Agents via Membership Inference

When a user asks an LLM agent to forget something, is it really gone? This
repository audits deletion in agent memory. It plants a fact in the agent's
memory, deletes it, and then runs two audits against an agent that was never told
the fact:

- **Behavioral audit**: ask about the fact without revealing its value, and
  measure whether replies disclose it or differ detectably from the never-told
  agent (membership-inference ROC-AUC).
- **Storage audit**: inspect the memory store for the value after deletion.

Each trial also stores an unrelated preference that should survive deletion, so
over-deletion is measured too.

**Memory systems:** **Mem0**, which extracts and consolidates conversational
facts into durable records, and **A-MEM**, which maintains linked, evolving
notes. **Backbones:** Llama 3.1 8B Instruct, GPT-OSS 20B and Gemma 4 31B IT,
served locally with vLLM.

## Repository layout

```
memory_audit/          the audit library and command-line interface
  experiment.py          one case: plant, delete, behavioral and storage audits
  attack.py              interrogation questions and response scoring
  backends.py            Mem0 and A-MEM adapters
  runner.py              runs cases in parallel; resumable; records excluded cases
  aggregate.py           merges shards, compares models, compares reasoning modes
  metrics.py, report.py  ROC-AUC, TPR at 1% FPR, bootstrap intervals, summaries
  ablation.py            reasoning-effort ablation (re-answers saved memory states)
scripts/
  download_models.py     fetch a backbone at its pinned revision
  export_metrics.py      flatten a comparison into one metrics table
  make_figures.py        draw the paper's figures
slurm/                 cluster job scripts (one GPU per shard) and environment setup
configs/
  full-runs/<model>.json            full config for one backbone (3 seeds, both memory systems)
  full-runs/<model>/shard-<n>.json  one memory system x one seed; one GPU job each
  ablation/<model>.json             seed-0 config for the reasoning ablation
```

## Setup

```bash
export MEMAUDIT_STORAGE=/path/to/scratch   # environment, weights and results (tens of GB)
bash slurm/setup.sh                        # Python environment from requirements.txt
source slurm/environment.sh
```

## Command-line interface

```bash
python -m memory_audit plan     --config configs/full-runs/gemma.json   # size of a run
python -m memory_audit run      --config <config> --output <dir> --workers 8
python -m memory_audit ablate   --baseline <finished shard> --output <dir>
python -m memory_audit collect  --config configs/full-runs/gemma.json --output <run>
python -m memory_audit compare  --run llama <run> --run gpt-oss <run> --run gemma <run> --output <dir>
python -m memory_audit collect-ablation --config configs/ablation/gemma.json --baseline <run> --output <dir>
```

`run` and `ablate` resume where they stopped. A case whose memory system never
retained the planted fact, or whose model output was truncated, is recorded as
excluded rather than counted as a successful deletion; every report lists these
exclusions alongside the metrics.

## Reproducing the results

Submit from the repository root. Add `--account`, `--partition` and `--gres`
as your cluster requires; Gemma 4 31B needs a large-memory GPU.

```bash
# 1. Pinned checkpoints (Llama and Gemma need HF_TOKEN)
for m in llama gpt-oss gemma; do sbatch --export=ALL,MEMAUDIT_MODEL=$m slurm/download_models.sbatch; done

# 2. Main grid: six shards per model (2 memory systems x 3 seeds), one GPU each
for m in llama gpt-oss gemma; do sbatch --array=0-5 --export=ALL,MEMAUDIT_MODEL=$m slurm/run_shard.sbatch; done

# 3. Merge each model's shards, compare the models, export the metrics table and figures
for m in llama gpt-oss gemma; do sbatch --export=ALL,MEMAUDIT_MODEL=$m slurm/aggregate.sbatch collect; done
sbatch slurm/aggregate.sbatch compare
sbatch slurm/aggregate.sbatch export

# 4. Reasoning ablation (Gemma thinking on vs. off) on the seed-0 shards
sbatch --array=0,3 --export=ALL,MEMAUDIT_MODEL=gemma,MEMAUDIT_BASELINE=$MEMAUDIT_STORAGE/runs/gemma slurm/run_shard.sbatch
sbatch --export=ALL,MEMAUDIT_MODEL=gemma slurm/aggregate.sbatch collect-ablation
```

Results go to `$MEMAUDIT_STORAGE/runs/`. Each collected run has
`summary.json`/`summary.csv` (metrics per condition and query budget),
`eligibility.json` (valid and excluded cases) and `observations.json` (every
evaluated memory state, with transcripts). The comparison directory adds
`comparison-metrics.csv`, the table behind the paper, and the figures.

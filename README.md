# Auditing Persistent-Memory Deletion in LLM Agents via Membership Inference

LLM agents can remember what users tell them across conversations. When a user
asks the agent to forget something, is it really gone? This repository measures
that. It plants a fact in an agent's memory, deletes it, and then checks two
things:

- **Behavioral audit:** ask the agent about the fact without stating its value,
  and compare its answers with those of an identical agent that was never told
  the fact. If deletion worked, the two behave the same. This comparison is a
  *membership-inference* test, scored with ROC-AUC: 0.5 means the deleted fact is
  indistinguishable from one never told; 1.0 means it is always detectable.
- **Storage audit:** search the memory store for the deleted value.

Each test also stores an unrelated preference that should survive the deletion,
so the audit catches over-deletion as well.

It covers two open-source memory systems, [Mem0](https://github.com/mem0ai/mem0)
(rewrites conversations into short standalone facts) and
[A-MEM](https://github.com/agiresearch/A-mem) (keeps linked notes), each driven
by one of three LLMs served locally with vLLM: Llama 3.1 8B Instruct,
GPT-OSS 20B and Gemma 4 31B IT. All facts are synthetic: random codes such as
"The user's preferred cafe code is MKR-3099e1f8…", plus 12 public facts ("the
capital of France is Paris") as a control. No external dataset is needed.

## Terms

| Term | Meaning |
|---|---|
| backbone | the LLM that writes memories and answers the user |
| case | one fact tested under one backbone, memory system and seed |
| seed | a random seed that generates a fresh set of 600 synthetic facts |
| shard | all cases of one memory system and one seed; one GPU job (612 cases) |
| API / natural-language deletion | delete by record ID, or by a written request ("Forget my …") that the backbone maps to records, in three wordings |
| cascade defense | optional cleanup that rebuilds memory without the deleted statement |
| combined-summary test | adds a summary holding both the fact and the preference before deleting |
| questions per fact | 1, 2, 3 or 5 questions asked in the behavioral audit |
| excluded case | the memory system never stored the fact, or the model's answer was cut off; reported separately, never counted as a successful deletion |

## Repository layout

```
memory_audit/   the audit library and command-line interface
                (experiment.py: one case; runner.py: runs cases; aggregate.py: reports)
scripts/        download_models.py, export_metrics.py, make_figures.py
slurm/          environment and setup scripts, vLLM server launcher, SLURM job scripts
configs/
  full-runs/<model>/shard-<n>.json   one shard; n = 0,1,2 is Mem0 seeds 0-2, n = 3,4,5 is A-MEM seeds 0-2
  full-runs/<model>.json             all six shards of a model, used to merge them
  ablation/gemma.json                seed-0 config for the Gemma reasoning ablation
```

## Requirements

- Linux x86_64 with `uv` and Python 3.11 on `PATH`.
- NVIDIA GPUs. Llama 8B and GPT-OSS 20B fit a 24 GB GPU. Gemma 4 31B needs one
  with about 80 GB or more.
- A Hugging Face account that has accepted the Llama 3.1 and Gemma 4 licenses,
  with its token in `HF_TOKEN`.
- Internet access for setup and model downloads. The first run also downloads the
  `all-MiniLM-L6-v2` embedding model used for scoring.
- At least 150 GB of disk under `MEMAUDIT_STORAGE`, about 90 GB of it model weights.

Configs reject model names from some vendors (a project policy); remove the check
in `memory_audit/config.py` to use other backbones.

## Setup

```bash
export MEMAUDIT_STORAGE=/path/to/scratch   # environments, weights and results; export it in every shell you use
bash slurm/setup.sh                        # CPU-only Python environment from requirements.txt
source slurm/environment.sh                # sets MEMAUDIT_PYTHON and the cache paths
"$MEMAUDIT_PYTHON" -m memory_audit plan --config configs/full-runs/llama/shard-0.json   # size of one shard
```

Each model also needs its checkpoint and its own GPU vLLM environment. Run
`slurm/download_models.sbatch` for each model (below), or run its two steps directly.

## Run one shard on a single GPU (no SLURM)

```bash
source slurm/environment.sh
export MEMAUDIT_WORKERS=8                                 # cases in parallel; also the server's batch size
bash slurm/serve_vllm.sh llama > server.log 2>&1 &        # serves 127.0.0.1:18000, where the configs point
"$MEMAUDIT_PYTHON" -m memory_audit doctor --config configs/full-runs/llama/shard-0.json   # repeat until it reports ready
"$MEMAUDIT_PYTHON" -m memory_audit run --config configs/full-runs/llama/shard-0.json \
    --output "$MEMAUDIT_STORAGE/runs/llama/shards/0" --workers 8
```

`serve_vllm.sh` loads the UF HiPerGator CUDA modules by default; set
`MEMAUDIT_CUDA_MODULE` to your module name, or to `none` if CUDA is already on
`PATH`. A shard runs 612 cases, each scoring 20 memory states, which is about 73,000
questions to the model plus the memory system's own calls. Rerunning the same
command resumes where it stopped.

## Reproduce the full study (SLURM)

Submit from the repository root, and wait for each step to finish before starting
the next. The `#SBATCH` defaults are for UF HiPerGator (partitions `hpg-turin` and
`hpg-default`, an L4 GPU, a 14-day limit); on another cluster add `--account`,
`--partition`, `--gres` and `--time` to each `sbatch` call, and give Gemma a larger GPU.

```bash
# 1. Checkpoints and vLLM environments
for m in llama gpt-oss gemma; do sbatch --export=ALL,MEMAUDIT_MODEL=$m slurm/download_models.sbatch; done

# 2. Six shards per model, one GPU each
for m in llama gpt-oss gemma; do sbatch --array=0-5 --export=ALL,MEMAUDIT_MODEL=$m,MEMAUDIT_WORKERS=8 slurm/run_shard.sbatch; done

# 3. Merge each model's shards, compare the models, export the metrics table and figures
for m in llama gpt-oss gemma; do sbatch --export=ALL,MEMAUDIT_MODEL=$m slurm/aggregate.sbatch collect; done
sbatch slurm/aggregate.sbatch compare
sbatch slurm/aggregate.sbatch export

# 4. Reasoning ablation: Gemma with thinking on, replayed on the seed-0 shards (0 = Mem0, 3 = A-MEM)
sbatch --array=0,3 --export=ALL,MEMAUDIT_MODEL=gemma,MEMAUDIT_BASELINE=$MEMAUDIT_STORAGE/runs/gemma slurm/run_shard.sbatch
sbatch --export=ALL,MEMAUDIT_MODEL=gemma slurm/aggregate.sbatch collect-ablation
```

## Results

```
$MEMAUDIT_STORAGE/runs/
  <model>/shards/<n>/   raw evidence for every case of one shard (step 2)
  <model>/              merged run: summary.csv, eligibility.json, observations.json (step 3)
  comparison/           all models on the facts valid for every model: comparison-metrics.csv, figures/
  gemma-ablation/       thinking off versus on (step 4)
```

- `comparison/comparison-metrics.csv` is the main table. Each row is one model,
  memory system, test, deletion method, wording, cleanup setting and number of
  questions. Key columns: `exact_disclosure_rate` (value revealed in an answer),
  `confirmed_active_residual_rate` (value still in memory), `roc_auc` with its 95%
  interval, and `retained_useful_memory` (unrelated preference kept).
- `eligibility.json` counts valid and excluded cases.
- `observations.json` has every evaluated memory state, with the questions and answers.

## Command reference

```
"$MEMAUDIT_PYTHON" -m memory_audit plan | doctor | run | ablate | collect | compare | collect-ablation | report --help
```

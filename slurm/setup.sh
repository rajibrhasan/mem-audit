#!/usr/bin/env bash
# Create the Python environment and install the pinned dependencies.
#   bash slurm/setup.sh
# Uses `uv` and `python3.11` from PATH; override with MEMAUDIT_UV / MEMAUDIT_BASE_PYTHON.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
source slurm/environment.sh
uv="${MEMAUDIT_UV:-uv}"
if [[ ! -x "$MEMAUDIT_PYTHON" ]]; then
    "$uv" venv --python "${MEMAUDIT_BASE_PYTHON:-python3.11}" "$MEMAUDIT_STORAGE/venv"
fi
"$uv" pip install --python "$MEMAUDIT_PYTHON" --torch-backend cpu -r requirements.txt
echo "Environment ready: $MEMAUDIT_PYTHON"

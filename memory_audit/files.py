"""Small file helpers shared across the package."""
import hashlib
import json
import os
from pathlib import Path
import uuid


def dump(path, value):
    """Write JSON atomically: write a temporary file, then rename it into place."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.{uuid.uuid4().hex}.tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")
    os.replace(temporary, path)


def sha256(path):
    """SHA-256 of a file's bytes, used to check saved results before reuse."""
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

"""Download a backbone at its pinned revision into $MEMAUDIT_STORAGE/models/<name>.

slurm/serve_vllm.sh serves the checkpoint from that directory.

Usage:
  python scripts/download_models.py {llama|gpt-oss|gemma}

Gated models (Llama, Gemma) need a Hugging Face token in HF_TOKEN.
"""
import argparse
import json
import os
from pathlib import Path

from huggingface_hub import HfApi, snapshot_download

# name: (repository, pinned revision, local directory)
MODELS = {
    "llama": ("meta-llama/Llama-3.1-8B-Instruct", "0e9e39f249a16976918f6564b8830bc894c89659", "llama31-8b-instruct"),
    "gpt-oss": ("openai/gpt-oss-20b", "6cee5e81ee83917806bbde320786a8fb61efebee", "gpt-oss-20b"),
    "gemma": ("google/gemma-4-31B-it", "842da3794eaa0b77d5f08bae87a17459d91ff475", "gemma4-31b-it"),
}
PATTERNS = ["*.json", "*.safetensors", "*.model", "*.tiktoken", "*.jinja", "*.txt"]


def main():
    parser = argparse.ArgumentParser(description="Download a pinned backbone checkpoint.")
    parser.add_argument("model", choices=MODELS)
    args = parser.parse_args()
    repository, revision, directory = MODELS[args.model]
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    downloaded = Path(snapshot_download(repository, revision=revision, token=token,
                                        allow_patterns=PATTERNS, max_workers=4))
    weights = [name for name in HfApi(token=token).list_repo_files(repository, revision=revision)
               if name.endswith(".safetensors")]
    if not weights or any(not (downloaded / name).is_file() for name in weights):
        raise RuntimeError("Checkpoint weights are incomplete")
    destination = Path(os.environ["MEMAUDIT_STORAGE"]) / "models" / directory
    destination.mkdir(parents=True, exist_ok=True)
    for item in downloaded.iterdir():
        target = destination / item.name
        if not target.exists():
            target.symlink_to(item.resolve(), target_is_directory=item.is_dir())
    (destination / "checkpoint-source.json").write_text(json.dumps(
        {"repository": repository, "revision": revision, "weights": weights}, indent=2) + "\n")
    print(f"Prepared {repository}@{revision} in {destination}")


if __name__ == "__main__":
    main()

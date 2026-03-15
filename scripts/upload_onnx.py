#!/usr/bin/env python3
"""
Upload ONNX artifacts to HuggingFace Hub.
Called by: just upload-onnx
"""

import argparse
from pathlib import Path
from huggingface_hub import HfApi

REPO_ID = "MiicaLabs/vibevoice-onnx-artifacts"

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifacts", type=Path, default=Path("artifacts"))
    args = parser.parse_args()

    api = HfApi()

    # Create repo if it doesn't exist
    api.create_repo(
        repo_id=REPO_ID,
        repo_type="model",
        exist_ok=True,  # no error if already exists
    )
    print(f"✓ Repo ready: https://huggingface.co/{REPO_ID}")

    files = {
        "vibevoice_acoustic.onnx": "onnx/vibevoice_acoustic.onnx",
        "vibevoice_semantic.onnx": "onnx/vibevoice_semantic.onnx",
    }

    for local_name, repo_path in files.items():
        local_path = args.artifacts / local_name
        if not local_path.exists():
            print(f"❌ Not found: {local_path} — run: just export")
            continue

        size_mb = local_path.stat().st_size / 1_048_576
        print(f"Uploading {local_name} ({size_mb:.1f} MB)...")
        api.upload_file(
            path_or_fileobj=str(local_path),
            path_in_repo=repo_path,
            repo_id=REPO_ID,
            repo_type="model",
        )
        print(f"    ✓ {repo_path}")

    print(f"\nDone — https://huggingface.co/{REPO_ID}")

if __name__ == "__main__":
    main()
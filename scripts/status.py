#!/usr/bin/env python3
"""
Print the last validation verdict and artifact inventory.
Called by: just status
"""

import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifacts", type=Path, default=Path("artifacts"))
    args = parser.parse_args()

    print("=== Artifacts ===")
    onnx_files = list(args.artifacts.glob("*.onnx"))
    if onnx_files:
        for f in onnx_files:
            size_mb = f.stat().st_size / 1_048_576
            print(f"  {f.name:<40} {size_mb:.1f} MB")
    else:
        print("  No .onnx files found — run: just export")

    print("\n=== Last validation report ===")
    report_path = args.artifacts / "validation_report.json"
    if report_path.exists():
        with open(report_path) as f:
            r = json.load(f)
        verdict = "GO" if r["go_nogo"] == "GO" else "NO-GO"
        print(f"  Verdict  : {verdict}")
        print(f"  Acoustic : {r['acoustic_passed']}/{r['n_samples']}")
        print(f"  Semantic : {r['semantic_passed']}/{r['n_samples']}")
        print(f"  p95 err  : acoustic={r['acoustic_p95_err']:.2e}  semantic={r['semantic_p95_err']:.2e}")
    else:
        print("  No report yet — run: just validate")


if __name__ == "__main__":
    main()
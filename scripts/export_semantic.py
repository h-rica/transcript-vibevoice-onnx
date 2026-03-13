#!/usr/bin/env python3
"""
Export the VibeVoice-ASR Semantic Tokenizer to ONNX.

The Semantic Tokenizer is trained with an ASR proxy task and captures
the linguistic representation of speech, complementary to the Acoustic
Tokenizer which captures timbre and prosody.

Usage:
    python scripts/export_semantic.py --output artifacts/
    python scripts/export_semantic.py --output artifacts/ --validate --samples 20
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import torch
from transformers import AutoProcessor, VibeVoiceAsrForConditionalGeneration

# Configuration 

MODEL_ID    = "microsoft/VibeVoice-ASR-HF"
SAMPLE_RATE = 24_000
DUMMY_SECS  = 10
OPSET       = 17
OUTPUT_NAME = "vibevoice_semantic.onnx"


# Helpers

def load_model(device: str) -> tuple:
    """Load the VibeVoice-ASR model from HuggingFace."""
    print(f"[1/4] Loading model {MODEL_ID} on {device}...")
    t0 = time.time()

    processor = AutoProcessor.from_pretrained(MODEL_ID)
    model = VibeVoiceAsrForConditionalGeneration.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.float32,
        device_map=device,
    )
    model.eval()

    print(f"    ✓ Model loaded in {time.time() - t0:.1f}s")
    return processor, model


def extract_semantic_tokenizer(model) -> torch.nn.Module:
    """Extract the Semantic Tokenizer submodule from the full model."""
    if hasattr(model, "semantic_tokenizer"):
        return model.semantic_tokenizer.eval()
    elif hasattr(model, "semantic_encoder"):
        return model.semantic_encoder.eval()
    raise AttributeError(
        "Cannot find semantic tokenizer in model. "
        "Inspect model structure with: print(model)"
    )


def export_to_onnx(
        tokenizer: torch.nn.Module,
        output_path: Path,
        device: str,
        opset: int,
) -> None:
    """Export the Semantic Tokenizer to ONNX."""
    print(f"[2/4] ONNX export (opset {opset})...")
    t0 = time.time()

    dummy = torch.randn(1, SAMPLE_RATE * DUMMY_SECS, dtype=torch.float32).to(device)

    torch.onnx.export(
        tokenizer,
        (dummy,),
        str(output_path),
        opset_version=opset,
        input_names=["audio"],
        output_names=["semantic_latents"],
        dynamic_axes={
            "audio":            {0: "batch", 1: "samples"},
            "semantic_latents": {0: "batch", 1: "frames"},
        },
        do_constant_folding=True,
        export_params=True,
        verbose=False,
    )

    size_mb = output_path.stat().st_size / 1_048_576
    print(f"    ✓ Export done in {time.time() - t0:.1f}s — {size_mb:.1f} MB")


def validate_onnx_model(output_path: Path) -> None:
    """Validate the ONNX model structure."""
    print("[3/4] Validating ONNX structure...")
    model = onnx.load(str(output_path))
    onnx.checker.check_model(model)
    print(f"    ✓ Valid ONNX model — {len(model.graph.node)} nodes")

    for inp in model.graph.input:
        shape = [d.dim_value or d.dim_param for d in inp.type.tensor_type.shape.dim]
        print(f"    ↳ Input  '{inp.name}': {shape}")
    for out in model.graph.output:
        shape = [d.dim_value or d.dim_param for d in out.type.tensor_type.shape.dim]
        print(f"    ↳ Output '{out.name}': {shape}")


def validate_numerical(
        tokenizer: torch.nn.Module,
        output_path: Path,
        device: str,
        n_samples: int = 10,
        threshold: float = 1e-4,
) -> dict:
    """Compare PyTorch vs ONNX outputs on n_samples random inputs."""
    print(f"[4/4] Numerical validation ({n_samples} samples, threshold {threshold})...")

    session = ort.InferenceSession(str(output_path), providers=["CPUExecutionProvider"])
    errors, passed = [], 0

    for i in range(n_samples):
        duration = np.random.randint(5, 30)
        audio_np = np.random.randn(1, SAMPLE_RATE * duration).astype(np.float32)
        audio_pt = torch.from_numpy(audio_np).to(device)

        with torch.no_grad():
            pt_out = tokenizer(audio_pt).cpu().numpy()
        ort_out = session.run(["semantic_latents"], {"audio": audio_np})[0]

        max_err  = float(np.abs(pt_out - ort_out).max())
        mean_err = float(np.abs(pt_out - ort_out).mean())
        errors.append({"max": max_err, "mean": mean_err, "duration_s": duration})

        ok = max_err < threshold
        passed += int(ok)
        print(f"    {'✓' if ok else '✗'} Sample {i+1:02d} ({duration}s) — "
              f"max_err={max_err:.2e}  mean_err={mean_err:.2e}")

    max_errors = [e["max"] for e in errors]
    report = {
        "component": "semantic_tokenizer",
        "passed": passed,
        "total": n_samples,
        "success": passed == n_samples,
        "threshold": threshold,
        "max_absolute_error": {
            "min":  float(np.min(max_errors)),
            "max":  float(np.max(max_errors)),
            "mean": float(np.mean(max_errors)),
            "p95":  float(np.percentile(max_errors, 95)),
        },
        "go_nogo": "GO" if passed == n_samples else "NO-GO",
    }

    verdict = "✅ GO" if report["success"] else "❌ NO-GO"
    print(f"\n    {verdict} — {passed}/{n_samples} samples under threshold {threshold}")
    return report

def main():
    parser = argparse.ArgumentParser(description="Export Semantic Tokenizer → ONNX")
    parser.add_argument("--output",    type=Path,  default=Path("artifacts"))
    parser.add_argument("--opset",     type=int,   default=OPSET)
    parser.add_argument("--validate",  action="store_true")
    parser.add_argument("--samples",   type=int,   default=10)
    parser.add_argument("--threshold", type=float, default=1e-4)
    parser.add_argument("--device",    type=str,   default="cpu",
                        choices=["cpu", "cuda", "mps"])
    args = parser.parse_args()

    args.output.mkdir(parents=True, exist_ok=True)
    output_path = args.output / OUTPUT_NAME

    print("=" * 60)
    print("  VibeVoice-ASR — Export Semantic Tokenizer → ONNX")
    print("=" * 60)

    processor, model = load_model(args.device)
    tokenizer        = extract_semantic_tokenizer(model)

    export_to_onnx(tokenizer, output_path, args.device, args.opset)
    validate_onnx_model(output_path)

    report = {}
    if args.validate:
        report = validate_numerical(tokenizer, output_path, args.device,
                                    n_samples=args.samples, threshold=args.threshold)
        report_path = args.output / "semantic_validation_report.json"
        with open(report_path, "w") as f:
            json.dump(report, f, indent=2)
        print(f"\n    Report saved: {report_path}")

    print("\n" + "=" * 60)
    if args.validate:
        print(f"  Final verdict: {report.get('go_nogo', 'N/A')}")
    else:
        print(f"  Export complete: {output_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()
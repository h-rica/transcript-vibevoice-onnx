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
from transformers import VibeVoiceAsrForConditionalGeneration

# Configuration

MODEL_ID    = "microsoft/VibeVoice-ASR-HF"
SAMPLE_RATE = 24_000
DUMMY_SECS  = 10
OPSET = 18
OUTPUT_NAME = "vibevoice_semantic.onnx"


# Helpers

class SemanticTokenizerExportWrapper(torch.nn.Module):
    """Expose a raw-waveform interface for ONNX export."""

    def __init__(self, encoder: torch.nn.Module) -> None:
        super().__init__()
        self.encoder = encoder

    def forward(self, audio: torch.Tensor) -> torch.Tensor:
        if audio.dim() != 2:
            raise ValueError(f"Expected audio shape [batch, samples], got {tuple(audio.shape)}")
        encoded = self.encoder(audio.unsqueeze(1))
        return encoded.latents


def load_model(device: str) -> torch.nn.Module:
    """Load the VibeVoice-ASR model from HuggingFace."""
    print(f"[1/4] Loading model {MODEL_ID} on {device}...")
    t0 = time.time()

    model = VibeVoiceAsrForConditionalGeneration.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.float32,
        device_map=device,
    )
    model.eval()

    print(f"    Model loaded in {time.time() - t0:.1f}s")
    return model


def extract_semantic_tokenizer(model) -> torch.nn.Module:
    """Extract the Semantic Tokenizer submodule from the full model."""
    if hasattr(model, "semantic_tokenizer_encoder"):
        encoder = model.semantic_tokenizer_encoder.eval()
        return SemanticTokenizerExportWrapper(encoder).eval()
    raise AttributeError(
        f"Cannot find semantic tokenizer in model. "
        f"Available submodules: {[name for name, _ in model.named_children()]}"
    )


def build_dummy_input(device: str) -> dict:
    """Create a dummy input tensor for ONNX export."""
    n_samples = SAMPLE_RATE * DUMMY_SECS
    return {"audio": torch.randn(1, n_samples, dtype=torch.float32).to(device)}


def export_to_onnx(
        tokenizer: torch.nn.Module,
        dummy_input: dict,
        output_path: Path,
        opset: int,
) -> None:
    """Export the Semantic Tokenizer to ONNX with dynamic axes."""
    print(f"[2/4] ONNX export (opset {opset})...")
    t0 = time.time()

    export_kwargs = {
        "opset_version": opset,
        "input_names": ["audio"],
        "output_names": ["semantic_latents"],
        "do_constant_folding": True,
        "export_params": True,
        "verbose": False,
    }
    dynamic_axes = {
        "audio": {0: "batch", 1: "samples"},
        "semantic_latents": {0: "batch", 1: "frames"},
    }

    try:
        torch.onnx.export(
            tokenizer,
            (dummy_input["audio"],),
            str(output_path),
            dynamo=True,
            dynamic_axes=dynamic_axes,
            **export_kwargs,
        )
    except Exception as exc:
        print(f"    Dynamo exporter failed: {exc.__class__.__name__}: {exc}")
        print("    Retrying with legacy exporter (dynamo=False)...")
        torch.onnx.export(
            tokenizer,
            (dummy_input["audio"],),
            str(output_path),
            dynamo=False,
            dynamic_axes=dynamic_axes,
            **export_kwargs,
        )

    size_mb = output_path.stat().st_size / 1_048_576
    print(f"    Export done in {time.time() - t0:.1f}s - {size_mb:.1f} MB")


def validate_onnx_model(output_path: Path) -> None:
    """Validate the ONNX model structure using the official checker."""
    print("[3/4] Validating ONNX structure...")
    model = onnx.load(str(output_path))
    onnx.checker.check_model(model)
    print(f"    Valid ONNX model - {len(model.graph.node)} nodes")

    for inp in model.graph.input:
        shape = [d.dim_value or d.dim_param for d in inp.type.tensor_type.shape.dim]
        print(f"    Input  '{inp.name}': {shape}")
    for out in model.graph.output:
        shape = [d.dim_value or d.dim_param for d in out.type.tensor_type.shape.dim]
        print(f"    Output '{out.name}': {shape}")


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
        print(f"    {'OK' if ok else 'FAIL'} Sample {i + 1:02d} ({duration}s) - "
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

    verdict = "GO" if report["success"] else "NO-GO"
    print(f"\n    {verdict} - {passed}/{n_samples} samples under threshold {threshold}")
    print(f"    p95 max absolute error: {report['max_absolute_error']['p95']:.2e}")
    return report


def main():
    parser = argparse.ArgumentParser(description="Export Semantic Tokenizer -> ONNX")
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
    print("  VibeVoice-ASR - Export Semantic Tokenizer -> ONNX")
    print("=" * 60)

    model = load_model(args.device)
    tokenizer = extract_semantic_tokenizer(model)
    dummy = build_dummy_input(args.device)

    export_to_onnx(tokenizer, dummy, output_path, args.opset)
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
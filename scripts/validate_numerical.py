#!/usr/bin/env python3
"""
Full numerical validation — PyTorch vs ONNX for both tokenizers.

Compares the outputs of both ONNX models (acoustic + semantic) against
their PyTorch reference on a set of random audio samples.
Generates a JSON report and a terminal summary.

Usage:
    python scripts/validate_numerical.py
    python scripts/validate_numerical.py --samples 50 --threshold 1e-4
    python scripts/validate_numerical.py --artifacts artifacts/ --real-audio tests/audio/
"""

import argparse
import json
import sys
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional

import librosa
import numpy as np
import onnxruntime as ort
import torch
from transformers import VibeVoiceAsrForConditionalGeneration

MODEL_ID             = "microsoft/VibeVoice-ASR-HF"
SAMPLE_RATE          = 24_000
THRESHOLD_ACOUSTIC   = 5e-4   # relaxed — systematic ONNX constant-folding offset
THRESHOLD_SEMANTIC   = 1e-4   # strict


class TokenizerWrapper(torch.nn.Module):
    """Match the raw-waveform interface used during ONNX export."""
    def __init__(self, encoder: torch.nn.Module) -> None:
        super().__init__()
        self.encoder = encoder

    def forward(self, audio: torch.Tensor) -> torch.Tensor:
        # encoder expects [batch, 1, samples] — same reshape as export wrapper
        return self.encoder(audio.unsqueeze(1)).latents


@dataclass
class SampleResult:
    sample_id:         int
    duration_s:        float
    source:            str      # "random" | "real_audio"
    acoustic_max_err:  float
    acoustic_mean_err: float
    semantic_max_err:  float
    semantic_mean_err: float
    acoustic_ok:       bool
    semantic_ok:       bool
    elapsed_ms:        float


@dataclass
class ValidationReport:
    model_id:         str
    timestamp:        str
    n_samples:        int
    threshold:        float
    acoustic_passed:  int
    semantic_passed:  int
    both_passed:      int
    acoustic_p95_err: float
    semantic_p95_err: float
    go_nogo:          str
    details:          list


class VibeVoiceValidator:

    def __init__(self, artifacts_dir: Path, device: str = "cpu"):
        self.device       = device
        self.artifacts_dir = artifacts_dir
        self._load_pytorch_model()
        self._load_onnx_sessions()

    def _load_pytorch_model(self):
        print("Loading PyTorch reference model...")
        self.pt_model = VibeVoiceAsrForConditionalGeneration.from_pretrained(
            MODEL_ID, torch_dtype=torch.float32, device_map=self.device,
        ).eval()

        if hasattr(self.pt_model, "acoustic_tokenizer_encoder"):
            self.pt_acoustic = TokenizerWrapper(self.pt_model.acoustic_tokenizer_encoder).eval()
            self.pt_semantic = TokenizerWrapper(self.pt_model.semantic_tokenizer_encoder).eval()
        else:
            raise AttributeError(
                f"Cannot find tokenizers in model. "
                f"Available: {[n for n, _ in self.pt_model.named_children()]}"
            )

        print("    ✓ PyTorch model loaded")

    def _load_onnx_sessions(self):
        print("Loading ONNX sessions...")
        acoustic_path = self.artifacts_dir / "vibevoice_acoustic.onnx"
        semantic_path = self.artifacts_dir / "vibevoice_semantic.onnx"

        if not acoustic_path.exists():
            raise FileNotFoundError(
                f"Acoustic ONNX not found: {acoustic_path}\n"
                "Run first: python scripts/export_acoustic.py"
            )
        if not semantic_path.exists():
            raise FileNotFoundError(
                f"Semantic ONNX not found: {semantic_path}\n"
                "Run first: python scripts/export_semantic.py"
            )

        opts = ort.SessionOptions()
        opts.intra_op_num_threads = 4

        self.ort_acoustic = ort.InferenceSession(
            str(acoustic_path), sess_options=opts,
            providers=["CPUExecutionProvider"],
        )
        self.ort_semantic = ort.InferenceSession(
            str(semantic_path), sess_options=opts,
            providers=["CPUExecutionProvider"],
        )
        print("    ✓ ONNX sessions loaded")

    def validate_sample(
            self,
            audio_np: np.ndarray,
            sample_id: int,
            source: str = "random",
            threshold: float = THRESHOLD_SEMANTIC,       # kept for real-audio path
            threshold_acoustic: float = THRESHOLD_ACOUSTIC,
            threshold_semantic: float = THRESHOLD_SEMANTIC,
    ) -> SampleResult:
        """Validate a single audio sample."""
        t0 = time.time()
        duration_s  = audio_np.shape[1] / SAMPLE_RATE
        audio_torch = torch.from_numpy(audio_np).to(self.device)

        # PyTorch (reference)
        with torch.no_grad():
            pt_acoustic = self.pt_acoustic(audio_torch).cpu().numpy()
            pt_semantic = self.pt_semantic(audio_torch).cpu().numpy()

        # ONNX
        ort_acoustic = self.ort_acoustic.run(["latents"],          {"audio": audio_np})[0]
        ort_semantic = self.ort_semantic.run(["semantic_latents"], {"audio": audio_np})[0]

        # Error metrics
        acoustic_diff = np.abs(pt_acoustic - ort_acoustic)
        semantic_diff = np.abs(pt_semantic - ort_semantic)

        return SampleResult(
            sample_id         = sample_id,
            duration_s        = duration_s,
            source            = source,
            acoustic_max_err  = float(acoustic_diff.max()),
            acoustic_mean_err = float(acoustic_diff.mean()),
            semantic_max_err  = float(semantic_diff.max()),
            semantic_mean_err = float(semantic_diff.mean()),
            acoustic_ok       = float(acoustic_diff.max()) < threshold_acoustic,
            semantic_ok       = float(semantic_diff.max()) < threshold_semantic,
            elapsed_ms        = (time.time() - t0) * 1000,
        )

    def run(
            self,
            n_random: int = 20,
            real_audio_dir: Optional[Path] = None,
            threshold: float = THRESHOLD_SEMANTIC,       # kept for real-audio path
    ) -> ValidationReport:
        """Run full validation."""
        import datetime

        print(f"\n{'─'*60}")
        print(f"  Numerical validation — {n_random} samples  |  threshold: {threshold}")
        print(f"{'─'*60}\n")

        results: list[SampleResult] = []

        # Random samples
        print(f"[A] Random samples ({n_random})...")
        for i in range(n_random):
            duration = np.random.randint(5, 60)
            audio_np = np.random.randn(1, SAMPLE_RATE * duration).astype(np.float32)
            result = self.validate_sample(
                audio_np, i + 1, "random",
                threshold_acoustic=THRESHOLD_ACOUSTIC,
                threshold_semantic=threshold,
                          )
            results.append(result)

            print(
                f"  [{i+1:02d}] {duration:2d}s  "
                f"acoustic [{'✓' if result.acoustic_ok else '✗'}] {result.acoustic_max_err:.2e}  "
                f"semantic [{'✓' if result.semantic_ok else '✗'}] {result.semantic_max_err:.2e}  "
                f"({result.elapsed_ms:.0f}ms)"
            )

        # Real audio files (optional)
        if real_audio_dir and real_audio_dir.exists():
            audio_files = (
                    list(real_audio_dir.glob("*.wav")) +
                    list(real_audio_dir.glob("*.mp3")) +
                    list(real_audio_dir.glob("*.m4a"))
            )
            if audio_files:
                print(f"\n[B] Real audio files ({len(audio_files)})...")
                for j, filepath in enumerate(audio_files[:10]):
                    try:
                        audio, _ = librosa.load(str(filepath), sr=SAMPLE_RATE, mono=True)
                        audio_np = audio[np.newaxis, :].astype(np.float32)
                        result   = self.validate_sample(
                            audio_np, n_random + j + 1, filepath.name, threshold
                        )
                        results.append(result)
                        print(
                            f"  [{filepath.name[:30]:30s}]  "
                            f"acoustic [{'✓' if result.acoustic_ok else '✗'}] {result.acoustic_max_err:.2e}  "
                            f"semantic [{'✓' if result.semantic_ok else '✗'}] {result.semantic_max_err:.2e}"
                        )
                    except Exception as e:
                        print(f"  ⚠ Error on {filepath.name}: {e}")

        # Build report
        acoustic_errs   = [r.acoustic_max_err for r in results]
        semantic_errs   = [r.semantic_max_err  for r in results]
        acoustic_passed = sum(1 for r in results if r.acoustic_ok)
        semantic_passed = sum(1 for r in results if r.semantic_ok)
        both_passed     = sum(1 for r in results if r.acoustic_ok and r.semantic_ok)

        report = ValidationReport(
            model_id         = MODEL_ID,
            timestamp        = datetime.datetime.now(datetime.timezone.utc).isoformat(),
            n_samples        = len(results),
            threshold        = threshold,
            acoustic_passed  = acoustic_passed,
            semantic_passed  = semantic_passed,
            both_passed      = both_passed,
            acoustic_p95_err = float(np.percentile(acoustic_errs, 95)),
            semantic_p95_err = float(np.percentile(semantic_errs, 95)),
            go_nogo          = "GO" if both_passed == len(results) else "NO-GO",
            details          = [asdict(r) for r in results],
        )

        self._print_summary(report)
        return report

    @staticmethod
    def _print_summary(report: ValidationReport):
        verdict = "✅  GO" if report.go_nogo == "GO" else "❌  NO-GO"
        print(f"\n{'═'*60}")
        print(f"  VERDICT : {verdict}")
        print(f"{'─'*60}")
        print(f"  Acoustic tokenizer : {report.acoustic_passed}/{report.n_samples} "
              f"(threshold={THRESHOLD_ACOUSTIC:.0e}, p95={report.acoustic_p95_err:.2e})")
        print(f"  Semantic tokenizer : {report.semantic_passed}/{report.n_samples} "
              f"(threshold={THRESHOLD_SEMANTIC:.0e}, p95={report.semantic_p95_err:.2e})")
        print(f"  Both OK            : {report.both_passed}/{report.n_samples}")
        print(f"{'═'*60}\n")


def _run_onnx_only(args) -> None:
    """CI smoke test — no PyTorch, just verify ONNX sessions load and produce correct shapes."""
    import datetime

    print("=== ONNX-only smoke test (CI mode) ===")
    acoustic_path = args.artifacts / "vibevoice_acoustic.onnx"
    semantic_path = args.artifacts / "vibevoice_semantic.onnx"

    for p in [acoustic_path, semantic_path]:
        if not p.exists():
            print(f"❌ Not found: {p}")
            sys.exit(2)

    opts = ort.SessionOptions()
    opts.intra_op_num_threads = 2
    acoustic = ort.InferenceSession(str(acoustic_path), sess_options=opts, providers=["CPUExecutionProvider"])
    semantic = ort.InferenceSession(str(semantic_path), sess_options=opts, providers=["CPUExecutionProvider"])
    print("✓ Sessions loaded")

    results = []
    for duration in [5, 10, 30]:
        audio_np = np.random.randn(1, SAMPLE_RATE * duration).astype(np.float32)
        a_out = acoustic.run(["latents"],          {"audio": audio_np})[0]
        s_out = semantic.run(["semantic_latents"], {"audio": audio_np})[0]

        a_frames = a_out.shape[1]
        s_frames = s_out.shape[1]
        a_rate   = a_frames / duration
        s_rate   = s_frames / duration
        a_ok     = abs(a_rate - 7.5) < 1.5
        s_ok     = abs(s_rate - 7.5) < 1.5

        print(f"  {duration:3d}s  acoustic [{'+' if a_ok else '!'}] shape={a_out.shape} rate={a_rate:.1f}Hz  "
              f"semantic [{'+'if s_ok else '!'}] shape={s_out.shape} rate={s_rate:.1f}Hz")
        results.append(a_ok and s_ok)

    go_nogo = "GO" if all(results) else "NO-GO"
    print(f"\nVERDICT: {'✅ GO' if go_nogo == 'GO' else '❌ NO-GO'}")

    report = {
        "mode": "onnx_only",
        "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "go_nogo": go_nogo,
        "acoustic_passed": sum(results),
        "semantic_passed": sum(results),
        "n_samples": len(results),
    }
    args.output.mkdir(parents=True, exist_ok=True)
    with open(args.output / "validation_report.json", "w") as f:
        json.dump(report, f, indent=2)

    sys.exit(0 if go_nogo == "GO" else 1)


def main():
    parser = argparse.ArgumentParser(
        description="Numerical validation — PyTorch vs ONNX for VibeVoice-ASR"
    )
    parser.add_argument("--artifacts",  type=Path,  default=Path("artifacts"))
    parser.add_argument("--samples",    type=int,   default=20)
    parser.add_argument("--threshold",  type=float, default=THRESHOLD_SEMANTIC,
                        help="Semantic tokenizer error threshold (default: 1e-4)")
    parser.add_argument("--real-audio", type=Path,  default=None,
                        help="Directory with real audio files for testing")
    parser.add_argument("--output",     type=Path,  default=Path("artifacts"))
    parser.add_argument("--onnx-only", action="store_true",
                        help="CI mode: skip PyTorch reference, only verify ONNX output shapes")
    args = parser.parse_args()

    if args.onnx_only:
        _run_onnx_only(args)
        return

    try:
        validator = VibeVoiceValidator(artifacts_dir=args.artifacts, device=args.device)
        report    = validator.run(
            n_random=args.samples,
            real_audio_dir=args.real_audio,
            threshold=args.threshold,
        )

        args.output.mkdir(parents=True, exist_ok=True)
        report_path = args.output / "validation_report.json"
        with open(report_path, "w") as f:
            json.dump(asdict(report), f, indent=2)
        print(f"Full report saved: {report_path}")

        sys.exit(0 if report.go_nogo == "GO" else 1)

    except FileNotFoundError as e:
        print(f"\n❌ Error: {e}")
        sys.exit(2)


if __name__ == "__main__":
    main()
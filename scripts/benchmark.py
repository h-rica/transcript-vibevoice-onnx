#!/usr/bin/env python3
"""
Performance benchmark for ONNX tokenizers.

Measures RTFx (Real-Time Factor) and encoding latency
for various audio durations, on CPU and GPU if available.

Usage:
    python scripts/benchmark.py
    python scripts/benchmark.py --artifacts artifacts/ --runs 10
    python scripts/benchmark.py --durations 5 30 60 120 300
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import onnxruntime as ort
from tabulate import tabulate


SAMPLE_RATE = 24_000


def run_benchmark(
        session:     ort.InferenceSession,
        input_name:  str,
        output_name: str,
        duration_s:  int,
        n_runs:      int = 5,
) -> dict:
    """Measure ONNX session performance on a signal of given duration."""
    n_samples = SAMPLE_RATE * duration_s
    audio_np  = np.random.randn(1, n_samples).astype(np.float32)

    # Warmup (not counted)
    session.run([output_name], {input_name: audio_np})

    # Timed runs
    times_ms = []
    for _ in range(n_runs):
        t0 = time.perf_counter()
        session.run([output_name], {input_name: audio_np})
        times_ms.append((time.perf_counter() - t0) * 1000)

    mean_ms = float(np.mean(times_ms))
    rtfx    = (duration_s * 1000) / mean_ms  # > 1.0 = faster than real-time

    return {
        "duration_s": duration_s,
        "mean_ms":    round(mean_ms, 1),
        "min_ms":     round(float(np.min(times_ms)), 1),
        "std_ms":     round(float(np.std(times_ms)), 1),
        "rtfx":       round(rtfx, 2),
        "n_runs":     n_runs,
    }


def get_providers() -> list[tuple[str, list]]:
    """Return available execution providers for benchmarking."""
    available = ort.get_available_providers()
    providers = [("CPU", ["CPUExecutionProvider"])]

    if "CUDAExecutionProvider" in available:
        providers.append(("CUDA", ["CUDAExecutionProvider", "CPUExecutionProvider"]))
    if "CoreMLExecutionProvider" in available:
        providers.append(("CoreML (Metal)", ["CoreMLExecutionProvider", "CPUExecutionProvider"]))

    return providers


def main():
    parser = argparse.ArgumentParser(description="Benchmark ONNX tokenizers — VibeVoice-ASR")
    parser.add_argument("--artifacts", type=Path, default=Path("artifacts"))
    parser.add_argument("--runs",      type=int,  default=5,
                        help="Number of runs per configuration (default: 5)")
    parser.add_argument("--durations", type=int,  nargs="+",
                        default=[5, 15, 30, 60, 120, 300],
                        help="Audio durations to test in seconds")
    parser.add_argument("--output",    type=Path, default=Path("artifacts"))
    args = parser.parse_args()

    acoustic_path = args.artifacts / "vibevoice_acoustic.onnx"
    semantic_path = args.artifacts / "vibevoice_semantic.onnx"

    if not acoustic_path.exists() or not semantic_path.exists():
        print("❌ ONNX files not found. Run the export scripts first.")
        return

    providers    = get_providers()
    all_results  = []

    print("=" * 70)
    print("  VibeVoice-ASR — ONNX tokenizer benchmark")
    print("=" * 70)

    for provider_name, provider_list in providers:
        print(f"\n{'─'*70}")
        print(f"  Provider: {provider_name}")
        print(f"{'─'*70}")

        acoustic_session = ort.InferenceSession(str(acoustic_path), providers=provider_list)
        semantic_session = ort.InferenceSession(str(semantic_path), providers=provider_list)

        rows = []
        for duration in args.durations:
            print(f"  {duration:4d}s", end=" ... ", flush=True)

            a = run_benchmark(acoustic_session, "audio", "latents",          duration, args.runs)
            s = run_benchmark(semantic_session, "audio", "semantic_latents", duration, args.runs)

            total_ms   = round(a["mean_ms"] + s["mean_ms"], 1)
            total_rtfx = round((duration * 1000) / total_ms, 2)

            row = {
                "provider":    provider_name,
                "duration_s":  duration,
                "acous_ms":    a["mean_ms"],
                "acous_rtfx":  a["rtfx"],
                "seman_ms":    s["mean_ms"],
                "seman_rtfx":  s["rtfx"],
                "total_ms":    total_ms,
                "total_rtfx":  total_rtfx,
            }
            rows.append(row)
            all_results.append({**row, **{"acoustic": a, "semantic": s}})
            print(f"total RTFx = {total_rtfx:.1f}×")

        print()
        print(tabulate(
            [[r["duration_s"], r["acous_ms"], r["acous_rtfx"],
              r["seman_ms"],   r["seman_rtfx"], r["total_ms"], r["total_rtfx"]]
             for r in rows],
            headers=["Duration (s)", "Acoustic (ms)", "RTFx",
                     "Semantic (ms)", "RTFx", "Total (ms)", "Total RTFx"],
            tablefmt="rounded_outline",
            floatfmt=".1f",
        ))

    # Save report
    args.output.mkdir(parents=True, exist_ok=True)
    report_path = args.output / "benchmark_report.json"
    with open(report_path, "w") as f:
        json.dump(all_results, f, indent=2)

    # Minimum RTFx criterion on CPU, 5-min audio
    print(f"\n{'═'*70}")
    cpu_5min = [r for r in all_results if r["provider"] == "CPU" and r["duration_s"] == 300]
    if cpu_5min:
        rtfx   = cpu_5min[0]["total_rtfx"]
        status = "✅" if rtfx >= 0.5 else "❌"
        print(f"  {status} CPU RTFx on 5-min audio: {rtfx:.2f}× (minimum required: 0.5×)")

    print(f"\n  Report saved: {report_path}")
    print(f"{'═'*70}")


if __name__ == "__main__":
    main()
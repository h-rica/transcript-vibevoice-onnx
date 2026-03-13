# transcript-vibevoice-onnx

> Demo — ONNX export of VibeVoice-ASR audio tokenizers  
> and inference via the Rust `ort` runtime.

**Model** : [`microsoft/VibeVoice-ASR`](https://huggingface.co/microsoft/VibeVoice-ASR-HF) (MIT)  
**Status** : 🚀 Demo

---

## Overview

VibeVoice-ASR (Microsoft, 2026) is a next-generation speech recognition model capable of processing **up to 60 minutes of audio in a single pass**, with native speaker identification and timestamps — no separate post-processing required.

This repo demonstrates how to export the VibeVoice-ASR audio tokenizers to ONNX and run them **entirely in Rust** via the [`ort`](https://github.com/pykeio/ort) crate, with no Python dependency at runtime.

### Model architecture

| Component | Description | Format |
|---|---|---|
| **Acoustic Tokenizer** (σ-VAE) | Encodes 24kHz PCM audio into 7.5 Hz latents | ONNX |
| **Semantic Tokenizer** | Captures the linguistic representation of speech | ONNX |
| **Qwen2.5 Decoder** (7B) | Generates structured Who/When/What transcription | SafeTensors |

---

## Repository structure

```
transcript-vibevoice-onnx/
│
├── scripts/
│   ├── export_acoustic.py        # Export Acoustic Tokenizer → ONNX
│   ├── export_semantic.py        # Export Semantic Tokenizer → ONNX
│   ├── validate_numerical.py     # PyTorch vs ONNX numerical validation (Python)
│   └── benchmark.py              # Latency benchmark for export + inference
│
├── tests/
│   ├── src/
│   │   └── main.rs               # ONNX inference test via ort (Rust)
│   └── Cargo.toml
│
├── notebooks/
│   └── exploration.ipynb         # VibeVoice architecture exploration
│
├── artifacts/                    # Generated .onnx files (gitignored > 100MB)
│   └── .gitkeep
│
├── .github/
│   └── workflows/
│       ├── export.yml            # CI: export + numerical validation
│       └── rust_test.yml         # CI: ort Rust test
│
├── requirements.txt
├── Makefile                      # Convenience commands
└── README.md
```

---

## Installation

### Python requirements

```bash
# Install uv if needed — https://docs.astral.sh/uv/
# Windows
winget install astral-sh.uv

# macOS / Linux
curl -LsSf https://astral.sh/uv/install.sh | sh

# Install project dependencies
uv sync
```

### Rust requirements

```bash
# Rust stable 1.75+
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh

# OnnxRuntime shared library (required by the ort crate)
# macOS
brew install onnxruntime

# Linux
wget https://github.com/microsoft/onnxruntime/releases/download/v1.20.0/onnxruntime-linux-x64-1.20.0.tgz
tar -xzf onnxruntime-linux-x64-1.20.0.tgz
export ORT_LIB_LOCATION=$(pwd)/onnxruntime-linux-x64-1.20.0/lib

# Windows
# Download from https://github.com/microsoft/onnxruntime/releases
```

---

## Usage

### Full workflow

```powershell
# 1. Export tokenizers to ONNX
just export

# 2. Run numerical validation
just validate

# 3. Performance benchmark
just benchmark

# 4. Test Rust inference via ort
just test-rust
```

### Individual commands

```powershell
# Export Acoustic Tokenizer only
uv run python scripts/export_acoustic.py --output artifacts/

# Export Semantic Tokenizer only
uv run python scripts/export_semantic.py --output artifacts/

# Validation with custom threshold
uv run python scripts/validate_numerical.py --threshold 1e-4 --samples 100

# Benchmark on N audio files
uv run python scripts/benchmark.py --audio-dir tests/audio_samples/ --runs 5
```

---

## Generated outputs

After running the workflow, the following files are available in `artifacts/`:

```
artifacts/
├── vibevoice_acoustic.onnx         # ~600–800 MB
├── vibevoice_semantic.onnx         # ~600–800 MB
├── validation_report.json          # PyTorch vs ONNX comparison
├── benchmark_report.json           # Performance metrics by audio duration
└── rust_validation_report.json     # Rust inference results (ort)
```

`.onnx` files are not versioned in Git (too large) but can be published to HuggingFace Hub or distributed via a CDN.

---

## License

Apache 2.0 — see [LICENSE](LICENSE)

> ⚠️ VibeVoice-ASR model weights (Microsoft) are subject to the MIT license.  
> Generated `.onnx` files inherit that license.
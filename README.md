# transcript-vibevoice-onnx

> Demo — ONNX export of VibeVoice-ASR audio tokenizers  
> and inference via the Rust `ort` runtime.

**Model** : [`microsoft/VibeVoice-ASR`](https://huggingface.co/microsoft/VibeVoice-ASR-HF) (MIT)  
**Status** : Demo

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

## License

Apache 2.0 — see [LICENSE](LICENSE)

> ⚠️ VibeVoice-ASR model weights (Microsoft) are subject to the MIT license.  
> Generated `.onnx` files inherit that license.

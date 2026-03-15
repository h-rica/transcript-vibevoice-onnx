# justfile — transcript-vibevoice-onnx
# Requires: https://github.com/casey/just
# Install:  cargo install just  |  winget install Casey.Just

set shell := ["powershell", "-NoLogo", "-Command"]

# Configurable variables
artifacts   := "artifacts"
samples     := "20"
threshold   := "1e-4"
duration    := "10"
device      := "cpu"
ort_version := "1.20.0"

# List available recipes (default)
default:
    @just --list

# Install Python dependencies
setup:
    uv sync

# Download and extract OnnxRuntime (skips if already present)
setup-ort:
    if (Test-Path "onnxruntime-win-x64-{{ort_version}}") { \
        Write-Host "ORT already present: onnxruntime-win-x64-{{ort_version}}" \
    } else { \
        Invoke-WebRequest -Uri "https://github.com/microsoft/onnxruntime/releases/download/v{{ort_version}}/onnxruntime-win-x64-{{ort_version}}.zip" -OutFile ort.zip; \
        Expand-Archive ort.zip -DestinationPath . -Force; \
        Remove-Item ort.zip; \
        Write-Host "ORT extracted: onnxruntime-win-x64-{{ort_version}}" \
    }

# Print the ORT_LIB_LOCATION to set manually if needed
ort-env:
    Write-Host "$env:ORT_LIB_LOCATION = '$(Resolve-Path onnxruntime-win-x64-{{ort_version}}\lib)'"

# Build Rust binary only (no run)
build-rust:
    cd tests; cargo build --release

# Export both tokenizers to ONNX
export:
    python scripts/export_acoustic.py --output {{artifacts}}/ --device {{device}}
    python scripts/export_semantic.py --output {{artifacts}}/ --device {{device}}

# Export with numerical validation included
export-validate:
    python scripts/export_acoustic.py \
        --output {{artifacts}}/ --device {{device}} \
        --validate --samples {{samples}} --threshold {{threshold}}
    python scripts/export_semantic.py \
        --output {{artifacts}}/ --device {{device}} \
        --validate --samples {{samples}} --threshold {{threshold}}

# Numerical validation — PyTorch vs ONNX
validate:
    python scripts/validate_numerical.py \
        --artifacts {{artifacts}}/ \
        --samples {{samples}} \
        --threshold {{threshold}} \
        --output {{artifacts}}/

# Performance benchmark
benchmark:
    python scripts/benchmark.py \
        --artifacts {{artifacts}}/ \
        --runs 5 \
        --durations 5 15 30 60 120 300 \
        --output {{artifacts}}/

# Build and run Rust ort inference test (requires ORT_LIB_LOCATION set at system level)
test-rust:
    cd tests; cargo build --release
    cd tests; cargo run --release -- \
        --artifacts ../{{artifacts}}/ \
        --duration {{duration}} \
        --samples 3 \
        --output ../{{artifacts}}/

# Full workflow: export → validate → benchmark → test-rust
all: export validate benchmark test-rust

# Show generated artifacts and last validation verdict
status:
    uv run python scripts/status.py --artifacts {{artifacts}}

# Remove generated JSON reports
clean-reports:
    -Remove-Item {{artifacts}}/*.json -ErrorAction SilentlyContinue

# Remove all generated artifacts (onnx + reports + rust build)
clean:
    -Remove-Item {{artifacts}}/*.onnx -ErrorAction SilentlyContinue
    -Remove-Item {{artifacts}}/*.json -ErrorAction SilentlyContinue
    cd tests && cargo clean

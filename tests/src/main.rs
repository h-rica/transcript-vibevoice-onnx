//! vibevoice-ort-test
//!
//! Validates the ort Rust runtime for VibeVoice-ASR ONNX tokenizers.
//!
//! Usage:
//!   cargo run -- --artifacts ../../artifacts/
//!   cargo run -- --artifacts ../../artifacts/ --duration 30 --samples 5

use std::path::PathBuf;
use std::time::Instant;

use anyhow::{Context, Result, anyhow};
use clap::Parser;
use ndarray::Array2;
use ort::{
    inputs,
    session::{builder::GraphOptimizationLevel, Session},
    value::TensorRef,
};
use serde::{Deserialize, Serialize};
use tracing::{info, warn, error};

// CLI
#[derive(Parser, Debug)]
#[command(
    name    = "vibevoice-ort-test",
    about   = "Validates the ort runtime for VibeVoice-ASR ONNX tokenizers",
    version = "0.1.0",
)]
struct Cli {
    /// Directory containing the .onnx files
    #[arg(long, default_value = "../../artifacts")]
    artifacts: PathBuf,

    /// Test signal duration in seconds
    #[arg(long, default_value_t = 10)]
    duration: usize,

    /// Number of inference samples
    #[arg(long, default_value_t = 3)]
    samples: usize,

    /// Output directory for the JSON report
    #[arg(long, default_value = "../../artifacts")]
    output: PathBuf,
}

#[derive(Debug, Serialize, Deserialize)]
struct InferenceResult {
    component:     String,
    duration_s:    usize,
    output_shape:  Vec<usize>,
    frame_rate_hz: f64,
    latency_ms:    f64,
    rtfx:          f64,
    success:       bool,
    error:         Option<String>,
}

#[derive(Debug, Serialize, Deserialize)]
struct ValidationReport {
    timestamp:        String,
    acoustic_results: Vec<InferenceResult>,
    semantic_results: Vec<InferenceResult>,
    acoustic_ok:      bool,
    semantic_ok:      bool,
    go_nogo:          String,
    platform:         String,
}

fn ort_error<E: std::fmt::Display>(error: E) -> anyhow::Error {
    anyhow!(error.to_string())
}

// Session builder
fn build_session(model_path: &std::path::Path) -> Result<Session> {
    Session::builder()
        .map_err(ort_error)?
        .with_optimization_level(GraphOptimizationLevel::Disable) // Level1+ too slow on laptop CPU (842 nodes)
        .map_err(ort_error)?
        .with_intra_threads(4)
        .map_err(ort_error)?
        .commit_from_file(model_path)
        .map_err(ort_error)
        .with_context(|| format!("Failed to load model: {}", model_path.display()))
}

// Inference

const SAMPLE_RATE: usize = 24_000;

fn generate_test_audio(duration_s: usize) -> Array2<f32> {
    let n_samples = SAMPLE_RATE * duration_s;
    let data: Vec<f32> = (0..n_samples)
        .map(|i| {
            let t = i as f32 / SAMPLE_RATE as f32;
            0.50 * (2.0 * std::f32::consts::PI * 440.0  * t).sin()
                + 0.25 * (2.0 * std::f32::consts::PI * 880.0  * t).sin()
                + 0.10 * (2.0 * std::f32::consts::PI * 1760.0 * t).sin()
        })
        .collect();

    Array2::from_shape_vec((1, n_samples), data)
        .expect("Failed to build audio tensor")
}

fn run_inference(
    session:    &mut Session,
    audio:      &Array2<f32>,
    output_key: &str,
    component:  &str,
    duration_s: usize,
) -> InferenceResult {
    let t0 = Instant::now();

    let result = (|| -> Result<(Vec<usize>, f64)> {
        let audio_input = TensorRef::from_array_view(audio)
            .map_err(ort_error)
            .context("Failed to create ONNX input tensor")?;
        let outputs = session.run(inputs!["audio" => audio_input])
            .map_err(ort_error)
            .context("ONNX inference failed")?;

        let (shape, _) = outputs[output_key]
            .try_extract_tensor::<f32>()
            .map_err(ort_error)
            .context("Failed to extract output tensor")?;

        let shape: Vec<usize> = shape.iter().map(|dim| *dim as usize).collect();
        info!("{} output shape: {:?}", component, shape);

        let n_frames      = shape.get(1).copied().unwrap_or(0);
        let frame_rate_hz = n_frames as f64 / duration_s as f64;

        Ok((shape, frame_rate_hz))
    })();

    let latency_ms = t0.elapsed().as_secs_f64() * 1000.0;
    let rtfx       = (duration_s as f64 * 1000.0) / latency_ms;

    match result {
        Ok((shape, frame_rate_hz)) => {
            let frame_rate_ok = (frame_rate_hz - 7.5).abs() < 1.5;
            if !frame_rate_ok {
                warn!(
                    "{} unexpected frame rate: {:.2} Hz (expected ~7.5 Hz)",
                    component, frame_rate_hz
                );
            }

            InferenceResult {
                component:     component.to_string(),
                duration_s,
                output_shape:  shape,
                frame_rate_hz: (frame_rate_hz * 100.0).round() / 100.0,
                latency_ms:    (latency_ms * 10.0).round() / 10.0,
                rtfx:          (rtfx * 100.0).round() / 100.0,
                success:       frame_rate_ok,
                error:         None,
            }
        }
        Err(e) => {
            error!("{} inference failed: {}", component, e);
            InferenceResult {
                component:     component.to_string(),
                duration_s,
                output_shape:  vec![],
                frame_rate_hz: 0.0,
                latency_ms:    (latency_ms * 10.0).round() / 10.0,
                rtfx:          0.0,
                success:       false,
                error:         Some(e.to_string()),
            }
        }
    }
}

fn main() -> Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(
            std::env::var("RUST_LOG")
                .unwrap_or_else(|_| "vibevoice_ort_test=info".to_string())
        )
        .init();

    let cli = Cli::parse();

    println!("╔══════════════════════════════════════════════════════════╗");
    println!("║   VibeVoice-ASR — ort runtime validation (Rust)         ║");
    println!("╚══════════════════════════════════════════════════════════╝\n");

    let acoustic_path = cli.artifacts.join("vibevoice_acoustic.onnx");
    let semantic_path = cli.artifacts.join("vibevoice_semantic.onnx");

    for path in &[&acoustic_path, &semantic_path] {
        if !path.exists() {
            anyhow::bail!(
                "ONNX file not found: {}\nRun the Python export scripts first.",
                path.display()
            );
        }
    }

    println!("[1/3] Loading ONNX sessions...");
    let mut acoustic_session = build_session(&acoustic_path)
        .context("Failed to load acoustic session")?;
    let mut semantic_session = build_session(&semantic_path)
        .context("Failed to load semantic session")?;
    println!("    ✓ Sessions loaded\n");

    println!("[2/3] Running inference ({} samples, {}s each)...", cli.samples, cli.duration);

    let mut acoustic_results = Vec::new();
    let mut semantic_results = Vec::new();

    for i in 0..cli.samples {
        let audio = generate_test_audio(cli.duration);
        print!("    Sample {:02}/{:02} ... ", i + 1, cli.samples);

        let a = run_inference(&mut acoustic_session, &audio, "latents",          "acoustic", cli.duration);
        let s = run_inference(&mut semantic_session, &audio, "semantic_latents", "semantic", cli.duration);

        println!(
            "acoustic [{}] {:.0}ms (RTFx {:.1}x)  semantic [{}] {:.0}ms (RTFx {:.1}x)",
            if a.success { "✓" } else { "✗" }, a.latency_ms, a.rtfx,
            if s.success { "✓" } else { "✗" }, s.latency_ms, s.rtfx,
        );

        acoustic_results.push(a);
        semantic_results.push(s);
    }

    let acoustic_ok = acoustic_results.iter().all(|r| r.success);
    let semantic_ok = semantic_results.iter().all(|r| r.success);
    let go_nogo     = if acoustic_ok && semantic_ok { "GO" } else { "NO-GO" };

    let report = ValidationReport {
        timestamp:        unix_timestamp(),
        acoustic_results,
        semantic_results,
        acoustic_ok,
        semantic_ok,
        go_nogo:          go_nogo.to_string(),
        platform:         format!("{}/{}", std::env::consts::OS, std::env::consts::ARCH),
    };

    println!("\n[3/3] Results\n");
    print_summary(&report);

    std::fs::create_dir_all(&cli.output)?;
    let report_path = cli.output.join("rust_validation_report.json");
    std::fs::write(&report_path, serde_json::to_string_pretty(&report)?)?;
    println!("\nReport saved: {}", report_path.display());

    std::process::exit(if go_nogo == "GO" { 0 } else { 1 });
}

fn print_summary(report: &ValidationReport) {
    let verdict = if report.go_nogo == "GO" { "✅  GO" } else { "❌  NO-GO" };
    println!("╔══════════════════════════════════════════════════════════╗");
    println!("║  VERDICT : {:<47}║", verdict);
    println!("╠══════════════════════════════════════════════════════════╣");
    println!("║  Acoustic tokenizer : {:<35}║",
             if report.acoustic_ok { "✓ OK" } else { "✗ FAILED" });
    println!("║  Semantic tokenizer : {:<35}║",
             if report.semantic_ok { "✓ OK" } else { "✗ FAILED" });
    println!("║  Platform           : {:<35}║", report.platform);
    println!("╚══════════════════════════════════════════════════════════╝");
}

fn unix_timestamp() -> String {
    let d = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default();
    format!("unix:{}", d.as_secs())
}
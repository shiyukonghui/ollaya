//! CPU parity check for the NeoHorse engine, against the exported bundle's golden scores.
//!
//! ```text
//! cargo run -p ollaya-runner --example parity_neohorse
//! ```
//!
//! Reads `out/neohorse` (the export of `convert/ollaya_convert/families/neohorse`) and the demo
//! image `out/neohorse/demo.png` (256x240, white with a red square and a blue circle), then checks
//! the same one-image one-question request `parity.py` uses: its four option scores must match the
//! HF/onnxruntime gate to 2e-3, with `red` (option 0) the argmax. It also prints the spliced row's
//! `decide` / `opts` / M-RoPE position, the §4.4 self-check, and checks three text-only questions
//! in one request against `ref.reference_scores` (the HF backbone, no image).

use std::path::Path;
use std::time::Instant;

use ollaya_runner::Engine;
use ollaya_runner::neohorse::NeohorseModel;
use ollaya_runner::{Device, vision};
use serde_json::json;

/// The golden scores of the demo request (`parity.py`, HF `Qwen3_5Model` + pointer head).
const GOLDEN: [f32; 4] = [4.2941, -3.7007, -0.1142, -4.1028];
/// The text-only goldens: `ref.reference_scores` for the three questions below, one request, no
/// image (their rows are 32 / 28 / 29 tokens).
const TEXT_GOLDEN: [&[f32]; 3] = [
    &[-1.369253, -1.30465, -0.775931, -1.613638],
    &[1.366228, -0.744736],
    &[0.115011, 1.641563, -0.730399],
];
const TOL: f32 = 2e-3;
/// The looser tolerance a GPU is allowed: reductions and matmuls reorder on the CUDA provider,
/// so scores move by ~1e-4 (ADR 0004's measured GPU parity) rather than the CPU's ~1e-6.
const GPU_TOL: f32 = 5e-4;
const STATE: &str = "A 256x240 image.";

/// The device this run uses: `PARITY_DEVICE=cpu` (the default) or `PARITY_DEVICE=cuda[:<n>]`.
fn device() -> Device {
    match std::env::var("PARITY_DEVICE").as_deref() {
        Ok(s) if s == "cuda" || s.starts_with("cuda:") => {
            let id = s
                .strip_prefix("cuda:")
                .and_then(|n| n.parse::<i32>().ok())
                .unwrap_or(0);
            Device::Cuda(id)
        }
        _ => Device::Cpu,
    }
}

/// The execution providers the *loaded* ONNX Runtime offers, as `ort` reports them through ONNX
/// Runtime's own `GetAvailableProviders`. This is the load-dynamic build's actual provider set,
/// not this crate's feature flags: with the GPU pack it must contain `CUDAExecutionProvider`.
///
/// `ort` rc.13 has no `session.providers()`, and with `load-dynamic` there is no import library to
/// call `OrtGetApiBase` from, so this asks the providers `ort` itself knows about.
#[cfg(feature = "cuda")]
fn provider_names() -> Vec<String> {
    use ort::ep::ExecutionProvider;
    let mut out = Vec::new();
    let cuda = ort::ep::CUDA::default();
    if cuda.is_available().unwrap_or(false) {
        out.push(cuda.name().to_string());
    }
    let tensorrt = ort::ep::TensorRT::default();
    if tensorrt.is_available().unwrap_or(false) {
        out.push(tensorrt.name().to_string());
    }
    let cpu = ort::ep::CPU::default();
    if cpu.is_available().unwrap_or(false) {
        out.push(cpu.name().to_string());
    }
    out
}

/// The providers the statically linked CPU build offers through `ort`.
#[cfg(not(feature = "cuda"))]
fn provider_names() -> Vec<String> {
    use ort::ep::ExecutionProvider;
    let cpu = ort::ep::CPU::default();
    if cpu.is_available().unwrap_or(false) {
        vec![cpu.name().to_string()]
    } else {
        Vec::new()
    }
}

/// The largest absolute difference between `got` and `want`, and `got`'s argmax.
fn compare(got: &[f32], want: &[f32]) -> (f32, usize) {
    let diff = got
        .iter()
        .zip(want)
        .map(|(g, w)| (g - w).abs())
        .fold(0.0f32, f32::max);
    let argmax = got
        .iter()
        .enumerate()
        .max_by(|a, b| a.1.total_cmp(b.1))
        .map(|(i, _)| i)
        .unwrap_or(usize::MAX);
    (diff, argmax)
}

fn main() -> anyhow::Result<()> {
    let dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../out/neohorse");
    let device = device();
    let providers = provider_names();
    println!("device = {device:?}, available providers = {providers:?}");
    if device != Device::Cpu {
        anyhow::ensure!(
            providers.iter().any(|p| p == "CUDAExecutionProvider"),
            "PARITY_DEVICE={device:?} but the loaded ONNX Runtime has no CUDAExecutionProvider \
             (providers: {providers:?}); the GPU pack is not on ORT_DYLIB_PATH/PATH"
        );
    }
    let load_start = Instant::now();
    let model = NeohorseModel::load(&dir, device, None)?;
    println!("load    {:.3} s", load_start.elapsed().as_secs_f64());
    let png = std::fs::read(dir.join("demo.png"))?;
    let state = json!(STATE);
    let questions = json!({
        "colour": {"type": "choice", "instructions": "What colour is the square?",
                   "criteria": {"red": null, "green": null, "blue": null, "yellow": null}}
    });

    // The spliced row and its positions (no inference).
    let rgb = vision::decode(&png)?;
    let parsed = ollaya_decision::parse_questions(&questions)?;
    let encoding = model.encode(&state, &parsed, Some(&rgb))?;
    let row = &encoding.rows[0];
    let image = encoding.image.as_ref().expect("an image was given");
    let m = model.image.merge_size;
    println!(
        "grid {}x{} -> merged {}x{}, {} visual tokens, row {} tokens",
        image.patches.grid_h,
        image.patches.grid_w,
        image.patches.grid_h / m,
        image.patches.grid_w / m,
        image.tokens,
        row.ids.len()
    );
    println!(
        "decide={} opts={:?} pos[decide]={}",
        row.decide, row.opts, row.positions[0][row.decide]
    );
    anyhow::ensure!(
        row.decide == 97 && row.opts == vec![87, 90, 93, 96],
        "splice/shift is off: decide={} opts={:?}",
        row.decide,
        row.opts
    );
    anyhow::ensure!(
        row.positions[0][row.decide] == 41,
        "M-RoPE is off: pos[decide]={}",
        row.positions[0][row.decide]
    );

    // The real request, through the engine API. The first forward pays one-off GPU costs
    // (CUDA/cuDNN handles, kernel selection, arena growth), so it is timed on its own and then
    // four more runs give a warm per-forward number.
    let run_start = Instant::now();
    let out = model.run_images(&state, &questions, std::slice::from_ref(&png))?;
    let cold_ms = run_start.elapsed().as_secs_f64() * 1e3;
    let mut warm_ms = Vec::new();
    for _ in 0..4 {
        let t = Instant::now();
        model.run_images(&state, &questions, std::slice::from_ref(&png))?;
        warm_ms.push(t.elapsed().as_secs_f64() * 1e3);
    }
    warm_ms.sort_by(f64::total_cmp);
    let got = &out.questions[0].logits;
    anyhow::ensure!(got.len() == GOLDEN.len(), "got {} scores", got.len());
    let (diff, argmax) = compare(got, &GOLDEN);
    println!("got    {got:?}");
    println!("golden {GOLDEN:?}");
    println!(
        "max|diff| = {diff:.3e} (tol {TOL:.1e}, gpu tol {GPU_TOL:.1e}), argmax = {argmax}"
    );
    println!(
        "forward: first {cold_ms:.1} ms, warm min/median {:.1}/{:.1} ms over {} runs",
        warm_ms[0],
        warm_ms[warm_ms.len() / 2],
        warm_ms.len() + 1
    );
    anyhow::ensure!(diff <= TOL, "max|diff| = {diff:.3e} > {TOL:.1e}");
    anyhow::ensure!(argmax == 0, "argmax = {argmax}, expected red (0)");
    println!(
        "PASS image (tokens: input={} state={} truncated={})",
        out.input_tokens, out.state_tokens, out.state_truncated
    );

    // Text-only: three questions in one request (multi-row batching and the zero image slot).
    let text = json!({
        "colour": {"type": "choice", "instructions": "What colour is the square?",
                   "criteria": {"red": null, "green": null, "blue": null, "yellow": null}},
        "circle": {"type": "noul", "instructions": "Is there a circle in the image?"},
        "size": {"type": "score", "instructions": "How big is the square?",
                 "criteria": ["tiny", "small", "big"]}
    });
    let parsed = ollaya_decision::parse_questions(&text)?;
    let out = model.run(&state, &parsed, None)?;
    anyhow::ensure!(
        out.questions.len() == TEXT_GOLDEN.len(),
        "wrong question count"
    );
    for (qid, (q, want)) in out.questions.iter().zip(TEXT_GOLDEN).enumerate() {
        let (diff, argmax) = compare(&q.logits, want);
        let qid = parsed
            .get_index(qid)
            .map(|(k, _)| k.as_str())
            .unwrap_or("?");
        println!(
            "text {qid}: got {:?} want {want:?} max|diff| = {diff:.3e} argmax = {argmax}",
            q.logits
        );
        anyhow::ensure!(diff <= TOL, "{qid}: max|diff| = {diff:.3e} > {TOL:.1e}");
    }
    println!("PASS text (tokens: input={})", out.input_tokens);

    // More than one question with an image is what upstream's adapter cannot serve.
    let two = json!({
        "colour": {"type": "choice", "instructions": "What colour is the square?",
                   "criteria": {"red": null, "green": null, "blue": null, "yellow": null}},
        "circle": {"type": "noul", "instructions": "Is there a circle in the image?"}
    });
    anyhow::ensure!(
        model
            .run_images(&state, &two, std::slice::from_ref(&png))
            .is_err(),
        "two questions with an image must be rejected"
    );
    println!("PASS one-question-per-image guard");
    Ok(())
}

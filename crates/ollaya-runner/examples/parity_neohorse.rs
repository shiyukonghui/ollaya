//! CPU parity check for the NeoHorse engine, against the exported bundle's golden scores.
//!
//! ```text
//! cargo run -p ollaya-runner --example parity_neohorse
//! cargo run -p ollaya-runner --example parity_neohorse -- out/neohorse-q8f16 out/goldens-neohorse.jsonl --quantized
//! ```
//!
//! Reads the model bundle (the default is `out/neohorse`) and the demo image
//! `out/neohorse/demo.png` (256x240, white with a red square and a blue circle), then checks the
//! same one-image one-question request `parity.py` uses: its four option scores must match the
//! HF/onnxruntime gate to 2e-3, with `red` (option 0) the argmax. It also prints the spliced row's
//! `decide` / `opts` / M-RoPE position, the §4.4 self-check, and checks three text-only questions
//! in one request against `ref.reference_scores` (the HF backbone, no image).
//!
//! A second argument names a goldens JSONL (`families/neohorse/goldens.py`); every request in it is
//! gated too, and the token rows the engine builds are compared with the reference's.
//!
//! `--quantized` switches the gate to the int8 export's: the option scores are allowed to drift
//! (8-bit weights move them by up to 6e-2) and the **decisions** are what must match
//! (`docs/decisions/0005-quantized-decoder-weights.md`). Without it the logits are gated tightly.

use std::path::{Path, PathBuf};
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
/// The quantized bundle's logit bound. Its weights are 8-bit, so option scores move by up to 1.2e-1
/// on the fixture set and the logits cannot be gated tightly at all; the decisions are what the gate
/// protects (`docs/decisions/0005-quantized-decoder-weights.md`). This bound only catches a broken
/// artifact, and carries headroom for a GPU, where the provider propagates fp16 past the GEMM.
const INT8_TOL: f32 = 3e-1;
const STATE: &str = "A 256x240 image.";

/// Which export is being checked, and therefore what the gate is.
#[derive(Clone, Copy, PartialEq)]
enum Gate {
    /// The fp32 export: option logits within `TOL`, decisions implied by that.
    Fp32,
    /// The int8 export: every decision must match the reference and the logits stay inside
    /// `INT8_TOL`. Measuring against the fp32 reference, not a fresh HF run.
    Int8,
}

impl Gate {
    fn name(self) -> &'static str {
        match self {
            Gate::Fp32 => "fp32",
            Gate::Int8 => "int8",
        }
    }
}

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

fn argmax_of(v: &[f32]) -> usize {
    v.iter()
        .enumerate()
        .max_by(|a, b| a.1.total_cmp(b.1))
        .map(|(i, _)| i)
        .unwrap_or(usize::MAX)
}

/// Gate one question's scores, and return the drift. `what` names the question in a failure.
fn check(gate: Gate, tol: f32, got: &[f32], want: &[f32], what: &str) -> anyhow::Result<f32> {
    anyhow::ensure!(got.len() == want.len(), "{what}: got {} scores, want {}", got.len(), want.len());
    let (diff, argmax) = compare(got, want);
    match gate {
        Gate::Fp32 => {
            anyhow::ensure!(diff <= tol, "{what}: max|diff| = {diff:.3e} > {tol:.1e}");
            anyhow::ensure!(argmax == argmax_of(want), "{what}: argmax {argmax} moved");
        }
        Gate::Int8 => {
            anyhow::ensure!(
                diff <= INT8_TOL,
                "{what}: max|diff| = {diff:.3e} > {INT8_TOL:.1e}; the artifact is broken, not just \
                 quantized"
            );
            let want_argmax = argmax_of(want);
            anyhow::ensure!(
                argmax == want_argmax,
                "{what}: the quantized bundle decides {argmax}, the reference {want_argmax} \
                 (drift {diff:.3e})"
            );
        }
    }
    Ok(diff)
}

/// Run every request in a goldens JSONL and gate it. Also checks that the engine builds the same
/// token rows the reference did — a different split would score different positions.
fn run_goldens(model: &NeohorseModel, path: &Path, gate: Gate, tol: f32) -> anyhow::Result<()> {
    let text = std::fs::read_to_string(path)
        .map_err(|e| anyhow::anyhow!("{}: {e}", path.display()))?;
    let (mut requests, mut questions, mut worst) = (0usize, 0usize, 0.0f32);
    for line in text.lines() {
        if line.trim().is_empty() {
            continue;
        }
        let rec: serde_json::Value = serde_json::from_str(line)?;
        let id = rec["id"].as_str().unwrap_or("?");
        let state = &rec["state"];
        let parsed = ollaya_decision::parse_questions(&rec["questions"])?;
        let rows = rec["rows"].as_array().expect("goldens rows");
        let plan = rec["plan"].as_array().expect("goldens plan");
        anyhow::ensure!(rows.len() == plan.len(), "{id}: {} rows, {} plan", rows.len(), plan.len());

        let encoding = model.encode(state, &parsed, None)?;
        anyhow::ensure!(
            encoding.rows.len() == rows.len(),
            "{id}: the engine builds {} rows, the reference {}",
            encoding.rows.len(),
            rows.len()
        );
        for (i, (built, want)) in encoding.rows.iter().zip(rows).enumerate() {
            let ids: Vec<u32> = serde_json::from_value(want["ids"].clone())?;
            let opts: Vec<usize> = serde_json::from_value(want["opts"].clone())?;
            anyhow::ensure!(
                built.ids == ids && built.decide == want["decide"].as_u64().unwrap_or(0) as usize
                    && built.opts == opts,
                "{id}: row {i} differs from the reference (ids/decide/opts)"
            );
        }

        let out = model.run(state, &parsed, None)?;
        anyhow::ensure!(out.questions.len() == plan.len(), "{id}: wrong question count");
        for (q, want) in out.questions.iter().zip(plan) {
            let qid = want["qid"].as_str().unwrap_or("?");
            let scores: Vec<f32> = serde_json::from_value(want["option_logits"].clone())?;
            let d = check(gate, tol, &q.logits, &scores, &format!("{id}/{qid}"))?;
            worst = worst.max(d);
            questions += 1;
        }
        requests += 1;
    }
    let bound = match gate {
        Gate::Fp32 => tol,
        Gate::Int8 => INT8_TOL,
    };
    println!(
        "PASS goldens ({}) {} requests, {} questions, worst |diff| {worst:.3e} (bound {bound:.1e})",
        gate.name(),
        requests,
        questions,
    );
    Ok(())
}

fn main() -> anyhow::Result<()> {
    // parity_neohorse [<model-dir> [<goldens.jsonl>]] [--quantized]
    let mut positional = Vec::new();
    let mut gate = Gate::Fp32;
    for a in std::env::args().skip(1) {
        match a.as_str() {
            "--quantized" => gate = Gate::Int8,
            s if s.starts_with('-') => anyhow::bail!(
                "unknown flag {s}; usage: parity_neohorse [<model-dir> [<goldens.jsonl>]] [--quantized]"
            ),
            s => positional.push(s.to_string()),
        }
    }
    let default_dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../out/neohorse");
    let dir: PathBuf = positional.first().map(PathBuf::from).unwrap_or(default_dir);
    let goldens: Option<PathBuf> = positional.get(1).map(PathBuf::from);
    let device = device();
    let providers = provider_names();
    println!("gate = {}, model-dir = {}", gate.name(), dir.display());
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
    match gate {
        // The fp32 gate is `TOL` everywhere; `GPU_TOL` is what the CUDA provider actually holds.
        Gate::Fp32 => println!(
            "max|diff| = {diff:.3e} (tol {TOL:.1e}, gpu tol {GPU_TOL:.1e}), argmax = {argmax}"
        ),
        Gate::Int8 => println!("max|diff| = {diff:.3e} (bound {INT8_TOL:.1e}), argmax = {argmax}"),
    }
    println!(
        "forward: first {cold_ms:.1} ms, warm min/median {:.1}/{:.1} ms over {} runs",
        warm_ms[0],
        warm_ms[warm_ms.len() / 2],
        warm_ms.len() + 1
    );
    check(gate, TOL, got, &GOLDEN, "image")?;
    println!(
        "PASS image ({}) (tokens: input={} state={} truncated={})",
        gate.name(),
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
        check(gate, TOL, &q.logits, want, qid)?;
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

    // The fixture set (`families/neohorse/goldens.py`): every request in it, gated per artifact.
    match &goldens {
        Some(path) => run_goldens(&model, path, gate, TOL)?,
        None => println!("(no goldens file given; pass one as the second argument to check the set)"),
    }
    Ok(())
}

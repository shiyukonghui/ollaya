//! ONNX Runtime engine for NeoHorse-Jev-4B (layout `neohorse-pointer-vision-v1`).
//!
//! NeoHorse is a merged Qwen3.5-4B **multimodal** backbone plus an independent pointer head, served
//! non-generatively: a state, typed questions (choice / noul / score) and at most one image go in,
//! one raw logit per option comes out. Its two halves are already two families of this runner, so
//! this engine is their composition:
//!
//! * the row layout and pointer readout are [`ollaya_decision::kev`]'s (`kev-pointer-v1`) — the same
//!   Qwen delimiters, the same `render`, the same `k(h[opt_pos[j]]) · q(h[decide_pos])` scores;
//! * the vision tower, the image preprocessing and the M-RoPE multimodal backbone are
//!   [`crate::decider_vision`]'s (`decider-vision-v1`).
//!
//! A row is a kev row, with the image spliced in after the leading `fim_prefix` when there is one:
//!
//! ```text
//! row = [fim_prefix] (+ [vision_start] [image]*n [vision_end]) user(render(state))[:max_state - 1]
//!     + [fim_middle] user(render(instructions))
//!     + ([box_start] user(option_j) [box_end])*      opt_pos[j] = index of option j's box_end
//!     + [fim_suffix]                                 decide_pos = the last index
//! ```
//!
//! Two graphs, both reading the bundle's own shards (BF16, widened by `Cast`) and pointer head:
//! * `vision.onnx`: `patches` [N, 1536], `pos_idx` / `pos_w` [N, 4], `rot_ids` [N, 2] ->
//!   `image_embeds` [N / 4, 2560] (see [`crate::vision`] for the inputs);
//! * `model.onnx`: `input_ids` [R, T] (T a multiple of `chunk`), `position_ids` [3, R, T],
//!   `image_embeds` [M, 2560] placed at the flat indices `image_pos` [M] (r * T + t), and the two
//!   readout indices `decide_idx` / `opt_idx` [N] (flat too: entry n is row n / K's decide token and
//!   its option n % K) -> `scores` [N], row-major over (row, option).
//!
//! # Splicing and M-RoPE, which differ from `decider-vision-v1`
//!
//! The prefix goes **after the row's first token**, not in front of the row, and every index that
//! follows shifts by the prefix length `n + 2`:
//!
//! ```text
//! ids     = [row.ids[0], vision_start, image * n, vision_end, row.ids[1..]]
//! decide' = decide + n + 2;  opts' = opts + n + 2
//! ```
//!
//! M-RoPE for a row with an image (`L = row.ids.len()`, `gh` / `gw` the merged patch grid, `n = gh *
//! gw` — measured against HF's `get_rope_index`):
//!
//! ```text
//! t = 0             fim_prefix   -> (0, 0, 0)
//! t = 1             vision_start -> (1, 1, 1)
//! t = 2 ..= n + 1   visual token k = i * gw + j -> (2, 2 + i, 2 + j)     start = 2
//! t = n + 2 ..      the L remaining tokens (vision_end first): all three axes identical,
//!                   counting up from next = 2 + max(gh, gw)
//! ```
//!
//! The `fim_prefix` token NeoHorse keeps in front of the image is why `start` is **2** where
//! [`crate::decider_vision`] uses 1, and why the image's flat position is `r * T + 2 + j` where
//! that engine writes `r * T + 1 + j`. A row without an image counts up on all three axes from 0.
//!
//! Without an image the decoder still takes one image slot: a zero embedding at a padding position
//! after every row's tokens, which no slot can attend to (the model is causal). Upstream's visual
//! adapter serves one question per request and repeats the image over rows; this engine keeps that
//! rule and rejects more than one question when an image is attached.

use std::path::Path;
use std::sync::Mutex;

use ndarray::{Array1, Array2, Array3, Ix1, Ix2};
use ollaya_decision::kev::{KevLayout, KevRow, KevState};
use ollaya_decision::{Calibration, CalibrationFile, Questions, TokenEncoder};
use ort::session::Session;
use ort::session::builder::SessionBuilder;
use serde::Deserialize;
use serde_json::Value;

use crate::decider::WeightsInMemory;
use crate::engine::Engine;
use crate::onnx::{CudaArena, Device, ModelFiles, load_tokenizer, session_for};
use crate::vision::{self, ImageConfig, Patches, Rgb};
use crate::{Error, Output, QuestionOutput};

/// Padded tokens per decoder run, as for the text decoders.
const TOKEN_BUDGET: usize = 8192;
/// Rows per decoder run.
const MAX_ROWS: usize = 1024;
/// Visual tokens per image, and patches (4 each, one `merge_size^2` block).
const MAX_IMAGE_TOKENS: usize = 1024;
const MAX_PATCHES: usize = 4 * MAX_IMAGE_TOKENS;
/// The backbone's hidden size (Qwen3.5-4B; not `decider-vision-v1`'s 2,048).
const HIDDEN: usize = 2560;
const VISION_INPUTS: [&str; 4] = ["patches", "pos_idx", "pos_w", "rot_ids"];
const VISION_OUTPUT: &str = "image_embeds";
const INPUTS: [&str; 6] = [
    "input_ids",
    "position_ids",
    "image_embeds",
    "image_pos",
    "decide_idx",
    "opt_idx",
];
const OUTPUT: &str = "scores";

/// The image delimiters of `decision.json`.
#[derive(Debug, Clone, Deserialize)]
struct VisionTokens {
    vision_start: u32,
    vision_end: u32,
    image: u32,
}

/// The fields of the `decision` layer this engine reads.
#[derive(Debug, Clone, Deserialize)]
struct DecisionConfig {
    engine: String,
    layout: String,
    /// Rows are padded to a multiple of this (one `Scan` step of the DeltaNet layers).
    chunk: usize,
    /// The vision delimiters, as the export repeats them in a top-level `tokens` object. Bundles
    /// that only keep the copy inside `special_tokens` fall back to that one at load.
    #[serde(default)]
    tokens: Option<VisionTokens>,
    image: ImageConfig,
    #[serde(default)]
    weights_in_memory: WeightsInMemory,
    #[serde(flatten)]
    kev: KevLayout,
}

/// An image after preprocessing.
#[derive(Debug, Clone)]
pub struct EncodedImage {
    pub resized: Rgb,
    pub patches: Patches,
    /// Visual tokens the image contributes (after the 2x2 merge).
    pub tokens: usize,
}

/// One question's row, with the image prefix already spliced in.
#[derive(Debug, Clone)]
pub struct NeohorseRow {
    pub ids: Vec<u32>,
    /// Index of the decide token (the row's last one).
    pub decide: usize,
    /// Index of each option's closing token, in option order.
    pub opts: Vec<usize>,
    /// M-RoPE positions of every id, one vector per (t, h, w) axis.
    pub positions: [Vec<i64>; 3],
}

/// The rows of one request and the image they share.
#[derive(Debug, Clone)]
pub struct NeohorseEncoding {
    pub rows: Vec<NeohorseRow>,
    pub state: KevState,
    pub image: Option<EncodedImage>,
}

pub struct NeohorseModel {
    vision: Mutex<Session>,
    decoder: Mutex<Session>,
    tokenizer: Tokenizer,
    chunk: usize,
    tokens: VisionTokens,
    pub image: ImageConfig,
    pub layout: KevLayout,
    pub calibration: Calibration,
    pub device: Device,
}

struct Tokenizer(tokenizers::Tokenizer);

impl TokenEncoder for Tokenizer {
    fn encode(&self, text: &str) -> Result<Vec<u32>, ollaya_decision::Error> {
        self.0
            .encode_fast(text, false)
            .map(|e| e.get_ids().to_vec())
            .map_err(|e| ollaya_decision::Error::Tokenizer(e.to_string()))
    }
}

fn read_json<T: serde::de::DeserializeOwned>(path: &Path) -> Result<T, Error> {
    let text = std::fs::read_to_string(path)
        .map_err(|e| Error::Model(format!("{}: {e}", path.display())))?;
    serde_json::from_str(&text).map_err(|e| Error::Model(format!("{}: {e}", path.display())))
}

fn check_io(session: &Session, inputs: &[&str], output: &str, what: &str) -> Result<(), Error> {
    let got: Vec<&str> = session.inputs().iter().map(|i| i.name()).collect();
    if got.len() != inputs.len()
        || !inputs.iter().all(|n| got.contains(n))
        || !session.outputs().iter().any(|o| o.name() == output)
    {
        return Err(Error::Model(format!(
            "{what} graph inputs {got:?} do not match the neohorse contract ({inputs:?} -> {output:?})"
        )));
    }
    Ok(())
}

impl Engine for NeohorseModel {
    fn run(&self, state: &Value, questions: &Questions) -> Result<Output, Error> {
        NeohorseModel::run(self, state, questions, None)
    }

    fn reads_images(&self) -> bool {
        true
    }

    fn run_images(
        &self,
        state: &Value,
        questions: &Value,
        images: &[Vec<u8>],
    ) -> Result<Output, Error> {
        let questions = ollaya_decision::parse_questions(questions)?;
        let image = match images {
            [] => None,
            [one] => Some(vision::decode(one)?),
            _ => {
                return Err(Error::Image(vision::ImageError::Count(images.len())));
            }
        };
        NeohorseModel::run(self, state, &questions, image.as_ref())
    }
}

impl NeohorseModel {
    /// Load a model exported to one directory (development and parity tooling).
    pub fn load(dir: &Path, device: Device, intra_threads: Option<usize>) -> Result<Self, Error> {
        Self::load_files(&ModelFiles::dir(dir), device, intra_threads)
    }

    pub fn load_files(
        files: &ModelFiles,
        device: Device,
        intra_threads: Option<usize>,
    ) -> Result<Self, Error> {
        let layer: Value = read_json(&files.decision)?;
        let bad = |e: String| Error::Model(format!("{}: {e}", files.decision.display()));
        let config: DecisionConfig =
            serde_json::from_value(layer.clone()).map_err(|e| bad(e.to_string()))?;
        if config.engine != "onnx" || config.layout != "neohorse-pointer-vision-v1" {
            return Err(Error::Model(format!(
                "unsupported engine/layout {}/{}; this engine serves onnx/neohorse-pointer-vision-v1",
                config.engine, config.layout
            )));
        }
        config.kev.validate().map_err(|e| bad(e.to_string()))?;
        if config.chunk == 0 {
            return Err(bad("chunk must be positive".into()));
        }
        let im = &config.image;
        if im.patch_size == 0 || im.merge_size == 0 || im.temporal_patch_size == 0 {
            return Err(bad(
                "image: patch, merge and temporal sizes must be positive".into(),
            ));
        }
        // `special_tokens` carries every delimiter, vision ones included.
        let tokens = match &config.tokens {
            Some(t) => t.clone(),
            None => serde_json::from_value(layer["special_tokens"].clone())
                .map_err(|e| bad(format!("vision tokens: {e}")))?,
        };
        let vision_graph = files
            .vision
            .as_ref()
            .ok_or_else(|| Error::Model("a neohorse model needs its vision graph".into()))?;
        let calibration = match &files.calibration {
            Some(path) => Calibration::from_file(&read_json::<CalibrationFile>(path)?),
            None => Calibration::default(),
        };
        let tokenizer = load_tokenizer(&files.tokenizer)?;

        let weights = config.weights_in_memory;
        let decoder = session_for(
            &files.graph,
            device,
            intra_threads,
            CudaArena::SameAsRequested,
            |b| weights.configure(configure(b, device)?),
        )?;
        check_io(&decoder, &INPUTS, OUTPUT, "decoder")?;
        let vision = session_for(
            vision_graph,
            device,
            intra_threads,
            CudaArena::SameAsRequested,
            |b| weights.configure(configure(b, device)?),
        )?;
        check_io(&vision, &VISION_INPUTS, VISION_OUTPUT, "vision")?;

        Ok(NeohorseModel {
            vision: Mutex::new(vision),
            decoder: Mutex::new(decoder),
            tokenizer: Tokenizer(tokenizer),
            chunk: config.chunk,
            tokens,
            image: config.image,
            layout: config.kev,
            calibration,
            device,
        })
    }

    /// Resize and cut an image into the vision graph's inputs.
    pub fn prepare_image(&self, image: &Rgb) -> Result<EncodedImage, Error> {
        let (resized, patches) = vision::preprocess(image, &self.image)?;
        if patches.count() > MAX_PATCHES {
            return Err(Error::Image(vision::ImageError::TooLarge {
                width: image.width,
                height: image.height,
                resized: (resized.width, resized.height),
                patches: patches.count(),
                max: MAX_PATCHES,
            }));
        }
        let tokens = patches.tokens(self.image.merge_size);
        Ok(EncodedImage {
            resized,
            patches,
            tokens,
        })
    }

    /// Every question's row, after the shared state and with the image spliced in.
    pub fn encode(
        &self,
        state: &Value,
        questions: &Questions,
        image: Option<&Rgb>,
    ) -> Result<NeohorseEncoding, Error> {
        if image.is_some() && questions.len() > 1 {
            return Err(ollaya_decision::Error::invalid(
                "NeoHorse's visual adapter serves one question per request; \
                 send one question per image",
            )
            .into());
        }
        let image = image.map(|i| self.prepare_image(i)).transpose()?;
        let state = self.layout.encode_state(&self.tokenizer, state)?;
        let rows = questions
            .iter()
            .map(|(qid, q)| {
                let row = self.layout.encode(&self.tokenizer, &state.ids, qid, q)?;
                Ok(self.splice(&row, image.as_ref()))
            })
            .collect::<Result<Vec<_>, Error>>()?;
        Ok(NeohorseEncoding { rows, state, image })
    }

    /// `row` with the image prefix spliced after its leading `fim_prefix`, and its M-RoPE
    /// positions. Shifted indices are `+ n + 2`; a row without an image counts 0.. on all axes.
    pub fn splice(&self, row: &KevRow, image: Option<&EncodedImage>) -> NeohorseRow {
        let Some(img) = image else {
            let p: Vec<i64> = (0..row.ids.len() as i64).collect();
            return NeohorseRow {
                ids: row.ids.clone(),
                decide: row.decide,
                opts: row.opts.clone(),
                positions: [p.clone(), p.clone(), p],
            };
        };
        let m = self.image.merge_size;
        let (gh, gw) = (img.patches.grid_h / m, img.patches.grid_w / m);
        let shift = img.tokens + 2;
        let len = row.ids.len() + shift;
        let mut ids = Vec::with_capacity(len);
        ids.push(row.ids[0]);
        ids.push(self.tokens.vision_start);
        ids.extend(std::iter::repeat_n(self.tokens.image, img.tokens));
        ids.push(self.tokens.vision_end);
        ids.extend_from_slice(&row.ids[1..]);

        let mut pos: [Vec<i64>; 3] = std::array::from_fn(|_| Vec::with_capacity(len));
        for p in &mut pos {
            p.push(0);
            p.push(1);
        }
        let start = 2i64;
        for i in 0..gh as i64 {
            for j in 0..gw as i64 {
                pos[0].push(start);
                pos[1].push(start + i);
                pos[2].push(start + j);
            }
        }
        let next = start + gh.max(gw) as i64;
        for k in 0..row.ids.len() as i64 {
            for p in &mut pos {
                p.push(next + k);
            }
        }
        NeohorseRow {
            ids,
            decide: row.decide + shift,
            opts: row.opts.iter().map(|o| o + shift).collect(),
            positions: pos,
        }
    }

    /// Answer every question, about the image when there is one.
    pub fn run(
        &self,
        state: &Value,
        questions: &Questions,
        image: Option<&Rgb>,
    ) -> Result<Output, Error> {
        let encoding = self.encode(state, questions, image)?;
        let scores = self.scores(&encoding)?;
        Ok(self.output(&encoding, scores))
    }

    /// The vision tower's output for an image: [tokens, 2560], row-major.
    pub fn image_embeds(&self, image: &EncodedImage) -> Result<Vec<f32>, Error> {
        let p = &image.patches;
        let n = p.count();
        let patches = Array2::from_shape_vec((n, p.dim), p.values.clone())
            .map_err(|e| Error::Model(e.to_string()))?;
        let pos_idx = Array2::from_shape_vec((n, 4), p.pos_idx.clone())
            .map_err(|e| Error::Model(e.to_string()))?;
        let pos_w = Array2::from_shape_vec((n, 4), p.pos_w.clone())
            .map_err(|e| Error::Model(e.to_string()))?;
        let rot_ids = Array2::from_shape_vec((n, 2), p.rot_ids.clone())
            .map_err(|e| Error::Model(e.to_string()))?;
        let mut session = self.vision.lock().expect("session mutex poisoned");
        let outputs = session.run(ort::inputs![
            "patches" => ort::value::Tensor::from_array(patches)?,
            "pos_idx" => ort::value::Tensor::from_array(pos_idx)?,
            "pos_w" => ort::value::Tensor::from_array(pos_w)?,
            "rot_ids" => ort::value::Tensor::from_array(rot_ids)?,
        ])?;
        let out = outputs[VISION_OUTPUT]
            .try_extract_array::<f32>()?
            .into_dimensionality::<Ix2>()
            .map_err(|e| Error::Model(format!("{VISION_OUTPUT}: {e}")))?;
        if out.shape() != [image.tokens, HIDDEN] {
            return Err(Error::Model(format!(
                "{VISION_OUTPUT} has shape {:?} for {} visual tokens",
                out.shape(),
                image.tokens
            )));
        }
        Ok(out.iter().copied().collect())
    }

    /// Each row's option scores (the first `k_row` entries of the graph's flat output), rows
    /// shortest first.
    ///
    /// The readout indices are flat and row-major, exactly like `image_pos`: entry `r * k + j` is
    /// row `r`'s decide token and its option `j`. A 2-D `opt_pos [R, K]` would be more direct, but
    /// `torch.export` cannot keep its shape symbolic — it derives the trunk's chunk reshape from
    /// the index's shape and folds a constant `floordiv` that is 0 for `R > 1`, so ONNX Runtime
    /// fails with "Integer division by zero". See `convert/.../neohorse/graphs.py`.
    pub fn scores(&self, encoding: &NeohorseEncoding) -> Result<Vec<Vec<f32>>, Error> {
        let rows = &encoding.rows;
        let embeds = encoding
            .image
            .as_ref()
            .map(|i| self.image_embeds(i))
            .transpose()?;
        let image_tokens = encoding.image.as_ref().map_or(0, |i| i.tokens);
        let mut order: Vec<usize> = (0..rows.len()).collect();
        order.sort_by_key(|&i| rows[i].ids.len());
        let padded: Vec<usize> = order
            .iter()
            .map(|&i| (rows[i].ids.len() + 1).div_ceil(self.chunk) * self.chunk)
            .collect();
        let pad = i64::from(self.layout.special_tokens.pad);

        let mut scores = vec![Vec::new(); rows.len()];
        let mut session = self.decoder.lock().expect("session mutex poisoned");
        for range in crate::engine::batches(&padded, TOKEN_BUDGET, MAX_ROWS) {
            let batch = &order[range.clone()];
            let seq = padded[range].iter().copied().max().unwrap_or(0);
            let n = batch.len();
            let k = batch.iter().map(|&i| rows[i].opts.len()).max().unwrap_or(0);
            let mut input_ids = Array2::<i64>::from_elem((n, seq), pad);
            let mut position_ids = Array3::<i64>::zeros((3, n, seq));
            let mut decide_idx = Array1::<i64>::zeros(n * k);
            let mut opt_idx = Array1::<i64>::zeros(n * k);
            for (r, &i) in batch.iter().enumerate() {
                let row = &rows[i];
                for (c, &id) in row.ids.iter().enumerate() {
                    input_ids[[r, c]] = i64::from(id);
                }
                for (a, axis) in row.positions.iter().enumerate() {
                    for (c, &p) in axis.iter().enumerate() {
                        position_ids[[a, r, c]] = p;
                    }
                }
                // Flat, one entry per (row, option): the row's decide token and that option's close.
                let decide = r * seq + row.decide;
                for (c, &p) in row.opts.iter().enumerate() {
                    decide_idx[r * k + c] = decide as i64;
                    opt_idx[r * k + c] = (r * seq + p) as i64;
                }
            }
            let (image_embeds, image_pos) = match &embeds {
                // Every row carries the same image, whose tokens sit at columns 2..n + 1.
                Some(e) => {
                    let mut all = Vec::with_capacity(n * e.len());
                    let mut pos = Vec::with_capacity(n * image_tokens);
                    for r in 0..n {
                        all.extend_from_slice(e);
                        pos.extend((0..image_tokens).map(|j| (r * seq + 2 + j) as i64));
                    }
                    (
                        Array2::from_shape_vec((n * image_tokens, HIDDEN), all)
                            .map_err(|e| Error::Model(e.to_string()))?,
                        Array1::from_vec(pos),
                    )
                }
                // Every row is shorter than `seq`, so row 0's last column is padding.
                None => (
                    Array2::<f32>::zeros((1, HIDDEN)),
                    Array1::from_vec(vec![seq as i64 - 1]),
                ),
            };
            let outputs = session.run(ort::inputs![
                "input_ids" => ort::value::Tensor::from_array(input_ids)?,
                "position_ids" => ort::value::Tensor::from_array(position_ids)?,
                "image_embeds" => ort::value::Tensor::from_array(image_embeds)?,
                "image_pos" => ort::value::Tensor::from_array(image_pos)?,
                "decide_idx" => ort::value::Tensor::from_array(decide_idx)?,
                "opt_idx" => ort::value::Tensor::from_array(opt_idx)?,
            ])?;
            let out = outputs[OUTPUT]
                .try_extract_array::<f32>()?
                .into_dimensionality::<Ix1>()
                .map_err(|e| Error::Model(format!("{OUTPUT}: {e}")))?;
            if out.len() != n * k {
                return Err(Error::Model(format!(
                    "{OUTPUT} has shape {:?} for {n} rows of up to {k} options",
                    out.shape()
                )));
            }
            let out = out.as_slice().expect("the output is contiguous");
            for (r, &i) in batch.iter().enumerate() {
                scores[i] = out[r * k..r * k + rows[i].opts.len()].to_vec();
            }
        }
        Ok(scores)
    }

    /// The request's output from each row's option scores.
    pub fn output(&self, encoding: &NeohorseEncoding, scores: Vec<Vec<f32>>) -> Output {
        Output {
            questions: scores
                .into_iter()
                .map(|logits| QuestionOutput {
                    logits,
                    act_logits: None,
                })
                .collect(),
            input_tokens: encoding.rows.iter().map(|r| r.ids.len()).sum(),
            state_tokens: encoding.state.tokens,
            state_truncated: encoding.state.truncated,
        }
    }
}

/// Session options for both neohorse graphs: the decoder options (see
/// [`crate::decider::configure`]), and no `GemmTransposeFusion`.
///
/// This is the workaround [`crate::kev`] documents, for the same reason: the graphs' pointer head
/// is a `Gemm`, and its input comes from the export's row gather through an identity `Transpose`.
/// ONNX Runtime 1.28 (which this build links) folds such a `Transpose` into the `Gemm` by flipping
/// `transA` without looking at `perm`, so a batch of exactly the wrong size would pass the shape
/// check and return wrong scores. With the pass off, ONNX Runtime runs the graph as exported, which
/// matches the goldens. The setting can go once `ort` links ONNX Runtime 1.30 or newer.
fn configure(builder: SessionBuilder, device: Device) -> Result<SessionBuilder, Error> {
    Ok(crate::decider::configure(builder, device)?
        .with_disabled_optimizers("GemmTransposeFusion")?)
}

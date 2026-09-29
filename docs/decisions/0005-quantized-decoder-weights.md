# 0005: int8 projection weights and an fp16 GEMM for NeoHorse-Jev-4B

- Status: accepted, 2026-09-29 (issue #18)
- Applies to: a new, separate artifact for `neohorse-pointer-vision-v1`
  (`out/neohorse-q8`, `out/neohorse-q8f16`). The shipped fp32 bundle is unchanged, and so is every
  other decoder family. The fp32 rule of
  [0002](0002-decoder-weights-in-memory.md) still holds for `decider-slots-v1` and `kev-pointer-v1`.

## Context

NeoHorse-Jev-4B's decoder holds 8.46 GiB of BF16 weights. 6.66 GiB of that is the 248 projection
matrices (attention q/k/v/o, the linear-attention in/out, the MLP gate/up/down); the rest is a
1.18 GiB embedding and a 0.62 GiB vision tower. On a 22 GB card those weights are what keeps the
model from sharing the GPU, and the decoder's 576-token prefill takes about 0.5 to 1 s.

[0002](0002-decoder-weights-in-memory.md) keeps the checkpoint in BF16 and rejects fp16/bf16
**compute** outright, because the parity gate (logits within 1e-3, identical decisions) was only ever
shown to hold for fp32 compute. This asks whether the weights can be compressed, and whether the
GEMM can reach the card's tensor cores, without giving up the decisions.

Two facts about this card (RTX 2080 Ti, sm_75) and ONNX Runtime 1.28 set the options. Measured by
scanning `onnxruntime_providers_cuda.dll` and reading ORT's v1.28.0 sources:

- The CUDA provider registers only `MatMulNBits`, `GatherBlockQuantized`, `MatMulInteger`,
  `DequantizeLinear` and `QuantizeLinear`. It has **no** `DynamicQuantizeLinear`, `QLinearMatMul` or
  `MatMulIntegerToFloat`, so the usual `quantize_dynamic` route has no CUDA kernel at all and would
  run on the CPU.
- `MatMulNBits` reaches the fused tensor-core GEMM only through its `T1` type. `T1=float16` is the
  fp16 path; `T1=float32` falls back to dequantize-then-fp32. `T1=bfloat16` is refused below compute
  capability 8.0.
- `GatherBlockQuantized` on CUDA is instantiated for 4-bit weights only, so the embedding cannot be
  quantized to 8 bits.

## Options

One-node benchmark of the real `up_proj` shape (M=576, K=2560, N=9216, 27.2 GFLOP), CUDA EP,
median of 20 runs, in one process (`out/bench_nbits.py`):

| node | ms | TFLOPS |
|---|---|---|
| `MatMulNBits` bits=8, T1=float32 | 10.82 | 2.5 |
| `MatMulNBits` bits=8, T1=float16 | 2.70 | **10.1** |
| `MatMul` fp32 | 5.69 | 4.8 |
| `MatMul` fp16 | 2.44 | 11.2 |

Whole decoder, 576 tokens, 477 visual tokens, same session (`out/profile_ort.py`):

| variant | device weights | decoder | drift vs fp32 | decisions |
|---|---|---|---|---|
| shipped fp32 | 8.46 GiB | 654 ms (an earlier session; not re-measurable today, see below) | — | — |
| int8 weights, fp32 GEMM | 5.26 GiB | 1033 ms | max 5.4e-2, p50 1.1e-2 | 16/16 |
| int8 weights, fp16 GEMM | 5.26 GiB | **469 ms** | max 6.0e-2, p50 1.1e-2 | 16/16 |

A whole-graph fp16 export (fp16 *weights*, no quantization) was not built: it would be larger than
the int8 weights and is strictly dominated by the fp16-GEMM variant, which already gets the fp16
kernels.

## Decision

**Ship the quantized decoder as its own artifact: int8 blockwise projection weights, with the GEMM
boundary in fp16.**

1. **A build-time pass**, `convert/ollaya_convert/families/neohorse/quantize.py`, rewrites the
   exported graph: every `MatMul` whose weight reaches it through the weightless pass's
   `Cast`/`Identity`/`Transpose` chain, and every `Gemm` with `transA=0, transB=1, alpha=beta=1`,
   becomes `com.microsoft.MatMulNBits` (bits=8, `block_size=128`, asymmetric, `accuracy_level=1`).
   The stored `[N, K]` layout is already `MatMulNBits`'s `B`, so nothing is re-laid out. Weights are
   quantized by ORT's own CUDA block quantizer (`quantize_matmul_8bits`), one at a time, straight
   out of the author's safetensors; the pass writes a new bundle and never touches the export.
2. **`--fp16-gemm`** puts `Cast(to=float16)` before and `Cast(to=float32)` after each quantized
   node, and stores the scales as fp16. This is what reaches the fused kernel.
3. **The pointer head stays F32.** It is the bundle's only F32 weight (2.5 MiB) and it is where a
   decision is read out; quantizing it would spend error on the score itself and save nothing. It is
   skipped by dtype, and the skip is reported.
4. **The embedding stays BF16**, because CUDA's `GatherBlockQuantized` is 4-bit only and 4 bits is a
   much larger step than 8. `weights_in_memory: bf16` still governs it.
5. **The quantized bundle is a separate artifact with its own gate.** It does not replace the fp32
   bundle, and it does not relax the parity rule for any other family.

## Consequences

- **Memory.** The decoder's weights fall from 8.46 GiB to 5.26 GiB (-3.2 GiB, -38%). The bundle
  adds a 3.65 GiB `model.q8.bin` to disk and still hard-links the author's shards for the embedding,
  the norms and the head.
- **Speed.** 469 ms against 1033 ms for the same weights with fp32 GEMMs (2.2x), on a card where the
  decoder is fp32-compute-bound.
- **The gate moves, and the decisions do not.** The option logits no longer match fp32 within the
  2e-3 of `parity.py`; the drift is max 6e-2, p50 1.1e-2. The decisions were 16/16 in the sample
  measured, and every one of the 19 realistic rows measured across both sessions kept its argmax,
  but the tightest fp32 top-2 gap seen (0.054) is the size of the drift, so flips are expected on
  inputs the sample did not reach. **The gate for this artifact has to be re-baselined against the
  decisions, not the logits** — a 2e-3 logit gate cannot hold at 8 bits, whose weight error is
  ~0.6% before any activation rounding.
- **The fp16 boundary is not surgical.** The intent was to keep every op but the GEMM in fp32, but
  the CUDA EP's optimizer folds the cast chain and propagates fp16 outwards: `Scan` fell from 3122
  to 848 ms and plain `MatMul` from 1316 to 315 ms, far beyond what the quantized GEMMs explain.
  The graph that runs is closer to fp16 compute than the flag name suggests, which is why the
  accuracy was measured on the GPU rather than the CPU — ORT's CPU EP emulates fp16 by upcasting and
  reports a much smaller difference (p50 +8.5e-4 against +2.2e-3 on the GPU).
- **The fp16 rounding is not what costs accuracy.** On the GPU, fp16 adds p50 2.2e-3 and max 7.6e-3
  over the int8-only variant: the 8-bit weights dominate the error by an order of magnitude.
- **The gate is implemented, and it is the decisions.** `parity_neohorse` takes a goldens JSONL and a
  `--quantized` flag that switches the gate. Against `families/neohorse/goldens.py` (12 requests, 26
  questions, measured from the fp32 export, which is itself HF-gated at 7.6e-6): the fp32 bundle
  reproduces the option logits to **5.2e-6**; the int8 bundle drifts up to **1.24e-1** and keeps
  **every decision**. The engine's token rows (`ids` / `decide` / `opts`) are compared with the
  reference's as well, so a different split cannot pass by scoring the wrong positions. The int8
  number is a CPU measurement; on the GPU the provider propagates fp16 past the GEMM (see above), so
  the logit bound carries headroom (`INT8_TOL = 3e-1`) and is a broken-artifact check, not the gate.
- **It is in the registry**: `library/neohorse:4b` and `library/neohorse:4b-int8` (plus the `latest`
  alias). NeoHorse is published on **ModelScope**, not Hugging Face, so `package.py` learned a second
  upstream host (a `modelscope:` prefix; the file list API supplies the sha256 and size the manifest
  pins). The int8 tag carries a **3.65 GB weights blob hosted by the registry** — every other model's
  weights stay with its author and are only referenced — because the quantized weights exist nowhere
  upstream.
- **The runner needs no change.** `MatMulNBits` is an ordinary contrib op and the CUDA pack's own
  `onnxruntime_providers_cuda.dll` carries the kernel; the graphs keep the fp32 bundle's inputs and
  outputs, so `check_io` and the layout are untouched.

## Evidence

- `out/bench_nbits.py` — the one-node fp32/fp16 comparison, both `MatMulNBits` and `MatMul`.
- `out/profile_ort.py` (pre-existing) — the whole-decoder profiles, `out/c_*.log` and
  `out/prof_*.json`.
- `out/q8_decisions.py`, `out/fp16_effect.py`, `out/weights_breakdown.py` — the decision agreement,
  the fp16-on-GPU drift, and the weight inventory that fixes the 6.66 GiB figure.
- `convert/ollaya_convert/families/neohorse/quantize.py` — the pass and its per-graph report.
- `convert/ollaya_convert/families/neohorse/goldens.py` — the fixture set, and
  `crates/ollaya-runner/examples/parity_neohorse.rs` the gate:
  `cargo run -p ollaya-runner --example parity_neohorse -- out/neohorse-q8f16 out/goldens-neohorse.jsonl --quantized`.
- `library/neohorse:4b-int8` in `registry/v2/` — the manifest, its layers and their annotations.
- Measured 2026-09-29 on choso (RTX 2080 Ti 22 GB, sm_75). The card was also serving other work
  (about 14.6 GiB held by other processes, 7.7 GiB free), so the fp32 baseline could not be
  re-measured in the same session — the fp32 bundle needs about 14 GiB and spilled (19.3 s at 576
  tokens, 41.5 s at 192). The two quantized variants need ~5 GiB and were measured back to back.
  The 654 ms fp32 figure is from an earlier, clean session. The two quantized numbers and the
  one-node benchmark were all taken under the same conditions as each other, and the one-node
  benchmark uses no significant VRAM at all.

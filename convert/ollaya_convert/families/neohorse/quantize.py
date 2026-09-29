"""Weight-only int8 quantization of a NeoHorse bundle's projection weights.

    uv run python -m ollaya_convert.families.neohorse.quantize SRC --out OUT [--bits 8] [--block-size 128]

The exported graphs keep every checkpoint weight as an external BF16 reference widened by a `Cast`
(`weightless_sharded`), so a projection weight arrives at its `MatMul` as
`Transpose(Identity(Cast(w)))` with `w` of shape `[N, K]` — which is `MatMulNBits`'s own `B` layout,
so nothing is re-laid out. This pass replaces each such `MatMul` with `com.microsoft.MatMulNBits`,
quantizing along `K` in blocks of `block_size` with one scale and one zero point per block, streams
one weight at a time out of the safetensors, and writes the packed int8 plus scales into a new
external data file next to the graph.

**Activations and accumulation stay fp32.** The `A` operand and every other node are untouched, and
`accuracy_level` is 1 (fp32 accumulate); only the weights lose precision. That is a much smaller
numerical change than the fp16/bf16 *compute* that `docs/decisions/0002-decoder-weights-in-memory.md`
rejects, and it is the only weight-only path ONNX Runtime's CUDA provider offers on this card (see
the module's `--help` note on `bits`).

The exported graphs and their shards are never modified: the pass writes a new bundle directory.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
from dataclasses import dataclass, field

import numpy as np
import onnx
from onnx import TensorProto, helper
from onnxruntime.capi._pybind_state import quantize_matmul_8bits

CONTRIB = "com.microsoft"

# Nodes that may sit between a `MatMul`'s weight input and the checkpoint tensor it comes from.
PASSTHROUGH = ("Cast", "Identity", "Transpose", "Reshape", "Squeeze", "Unsqueeze")

SIDECARS = ("decision.json", "calibration.json", "files.json", "tokenizer.json", "demo.png",
            "model_manifest.json")

#: Byte size of each safetensors dtype this pass reads.
DTYPE_BYTES = {TensorProto.BFLOAT16: 2, TensorProto.FLOAT: 4}


def external_ref(init: TensorProto):
    """`(location, offset, length)` of an external initializer, or None."""
    info = {kv.key: kv.value for kv in init.external_data}
    if "location" not in info:
        return None
    count = int(np.prod(init.dims, dtype=np.int64)) if init.dims else 1
    return (info["location"], int(info.get("offset", 0)),
            int(info.get("length", count * DTYPE_BYTES[init.data_type])))


def read_float(base_dir: str, init: TensorProto) -> np.ndarray:
    """A checkpoint weight as fp32, exactly as the graph's widening `Cast` would produce it.

    BF16 is upcast bit-exactly (`(u16 << 16).view(f32)`, the same trick as
    `weightless_sharded.Source.value`); numpy has no bfloat16, and this never rounds.
    """
    loc, offset, length = external_ref(init)
    with open(os.path.join(base_dir, loc), "rb") as f:
        f.seek(offset)
        buf = f.read(length)
    if init.data_type == TensorProto.BFLOAT16:
        a = np.frombuffer(buf, dtype=np.uint16)
        return (a.astype(np.uint32) << 16).view(np.float32).reshape(tuple(init.dims))
    return np.frombuffer(buf, dtype=np.float32).reshape(tuple(init.dims))


def weight_source(graph, producer, name):
    """`(initializer, transposed)` when `name` carries a checkpoint weight, reached from a `MatMul`
    input through pass-through nodes only; `None` when it is an activation."""
    transposed = False
    seen = set()
    while name not in seen:
        seen.add(name)
        node = producer.get(name)
        if node is None:
            init = next((t for t in graph.initializer if t.name == name), None)
            if (init is None or init.data_location != TensorProto.EXTERNAL
                    or init.data_type not in DTYPE_BYTES):
                return None
            return init, transposed
        if node.op_type not in PASSTHROUGH:
            return None
        if node.op_type == "Transpose":
            perm = next((list(a.ints) for a in node.attribute if a.name == "perm"), None)
            if perm is None or perm != list(reversed(range(len(perm)))):
                return None   # a real re-layout, not the weightless pass's transpose-to-MatMul
            transposed = not transposed
        name = node.input[0]
    return None


@dataclass
class Stats:
    matmuls: int = 0
    quantized: int = 0
    skipped: list = field(default_factory=list)
    weight_bytes: int = 0
    quantized_bytes: int = 0

    def as_dict(self):
        return {"matmuls": self.matmuls, "quantized": self.quantized, "skipped": self.skipped,
                "weight_bytes": self.weight_bytes, "quantized_bytes": self.quantized_bytes,
                "ratio": round(self.quantized_bytes / self.weight_bytes, 4) if self.weight_bytes else 0}


def quantize_graph(src_dir: str, graph_file: str, out_dir: str, bin_name: str,
                   bits: int, block_size: int, fp16_gemm: bool = False) -> Stats:
    """Rewrite one graph of the bundle; returns what it did."""
    if bits != 8:
        raise SystemExit("only bits=8 is wired to a CUDA kernel (docs/families/neohorse.md)")
    model = onnx.load(os.path.join(src_dir, graph_file), load_external_data=False)
    graph = model.graph
    producer = {o: n for n in graph.node for o in n.output}
    stats = Stats()

    # Packed blobs and their side tensors, written to `bin_name` after the graph walk.
    blobs: list[tuple[TensorProto, bytes]] = []
    offset = 0
    nodes: list = []

    def add_external(name, dims, data_type, raw: bytes) -> str:
        nonlocal offset
        t = TensorProto(name=name, data_type=data_type, dims=list(dims))
        t.data_location = TensorProto.EXTERNAL
        for k, v in (("location", bin_name), ("offset", str(offset)), ("length", str(len(raw)))):
            t.external_data.add(key=k, value=v)
        blobs.append((t, raw))
        offset += len(raw)
        return name

    for node in graph.node:
        if node.op_type not in ("MatMul", "Gemm"):
            nodes.append(node)
            continue
        stats.matmuls += 1
        # Both forms the export emits compute `A @ stored^T` with the weight stored `[N, K]`, which is
        # MatMulNBits's own `B` layout: `MatMul(A, T(w))` (the weightless transpose), and
        # `Gemm(A, w, bias, transB=1)`. Anything else is an activation product and is left alone.
        bias = None
        if node.op_type == "MatMul":
            src = weight_source(graph, producer, node.input[1]) if len(node.input) == 2 else None
            if src is None or not src[1]:
                nodes.append(node)   # activation x activation, or an untransposed weight
                continue
        else:
            at = {a.name: a for a in node.attribute}
            if (at["transA"].i if "transA" in at else 0) != 0 or \
                    (at["transB"].i if "transB" in at else 0) != 1 or \
                    (at["alpha"].f if "alpha" in at else 1.0) != 1.0 or \
                    (at["beta"].f if "beta" in at else 1.0) != 1.0:
                nodes.append(node)
                continue
            src = weight_source(graph, producer, node.input[1]) if len(node.input) >= 2 else None
            if src is None or src[1]:
                nodes.append(node)
                continue
            bias = node.input[2] if len(node.input) > 2 and node.input[2] else None
        init = src[0]
        # Only the backbone/vision checkpoint's BF16 weights are compressed. The pointer head is the
        # bundle's one F32 weight (2.5 MiB), and it is where a decision is read out: quantizing it
        # would spend error on the score itself and save nothing.
        if init.data_type != TensorProto.BFLOAT16:
            stats.skipped.append((node.name, list(init.dims), "F32 source (the pointer head)"))
            nodes.append(node)
            continue
        if len(init.dims) != 2:
            stats.skipped.append((node.name, list(init.dims), "weight is not 2-D"))
            nodes.append(node)
            continue
        n, k = init.dims
        if k % block_size:
            stats.skipped.append((node.name, list(init.dims), f"K={k} not a multiple of {block_size}"))
            nodes.append(node)
            continue
        w = np.ascontiguousarray(read_float(src_dir, init), dtype=np.float32)   # [N, K]
        stats.weight_bytes += w.nbytes
        # The kernel takes the MatMul view of B, [K, N]; it writes the packed [N, n_blocks, blob].
        b = np.ascontiguousarray(w.T)

        n_blocks = -(-k // block_size)
        packed = np.zeros((n, n_blocks, block_size), dtype=np.uint8)
        scales = np.zeros((n, n_blocks), dtype=np.float32)
        zero_points = np.zeros((n, n_blocks), dtype=np.uint8)
        # ORT's own CUDA block quantizer, so the layout is the one its kernels expect.
        quantize_matmul_8bits(packed, b, scales, zero_points, block_size, n, k, False)

        tag = f"{node.name or 'matmul'}#q8"
        b_name = add_external(tag + ".weight", packed.shape, TensorProto.UINT8, packed.tobytes())
        # `MatMulNBits`' T1 covers A, the scales and the output together. Keeping T1 fp32 puts it on
        # a dequant-then-fp32-GEMM path measured at 2.5 TFLOPS; T1 fp16 reaches the fused
        # tensor-core kernel at 10.1 TFLOPS on this card (out/bench_nbits.py). `fp16_gemm` takes the
        # fp16 kernel but keeps *only* the GEMM's own boundary in fp16 — every other op stays fp32 —
        # by casting A down and Y back up around the node.
        s_dtype, s_bytes = (TensorProto.FLOAT16, scales.astype(np.float16).tobytes()) if fp16_gemm \
            else (TensorProto.FLOAT, scales.tobytes())
        s_name = add_external(tag + ".scales", scales.shape, s_dtype, s_bytes)
        z_name = add_external(tag + ".zero_points", zero_points.shape, TensorProto.UINT8,
                              zero_points.tobytes())
        stats.quantized += 1
        stats.quantized_bytes += packed.nbytes + len(s_bytes) + zero_points.nbytes
        # `MatMulNBits` has an optional bias input, but its type constraint is narrower than the
        # vision tower's fp32 bias, so the bias is added by its own node instead (the graph's Gemms
        # are all alpha = beta = 1, so this is the same arithmetic).
        out = node.output[0]
        if bias:
            out = tag + ".prebias"
        if fp16_gemm:
            a16, y16 = tag + ".a16", tag + ".y16"
            nodes.append(helper.make_node("Cast", [node.input[0]], [a16], to=TensorProto.FLOAT16,
                                          name=tag + ":a16"))
            nodes.append(helper.make_node(
                "MatMulNBits", [a16, b_name, s_name, z_name], [y16],
                name=tag, domain=CONTRIB, K=k, N=n, bits=bits, block_size=block_size,
                accuracy_level=1))
            nodes.append(helper.make_node("Cast", [y16], [out], to=TensorProto.FLOAT,
                                          name=tag + ":y32"))
        else:
            nodes.append(helper.make_node(
                "MatMulNBits", [node.input[0], b_name, s_name, z_name], [out],
                name=tag, domain=CONTRIB, K=k, N=n, bits=bits, block_size=block_size,
                accuracy_level=1,   # fp32 accumulate; the activations are fp32 already
            ))
        if bias:
            nodes.append(helper.make_node("Add", [out, bias], [node.output[0]], name=tag + ":bias"))

    # Drop the Cast/Identity/Transpose chains and their checkpoint initializers, now unread.
    outputs = {o.name for o in graph.output}
    while True:
        used = {i for nd in nodes for i in nd.input} | outputs
        keep = []
        for nd in nodes:
            dead = (nd.op_type in ("Cast", "Identity", "Transpose")
                    and not any(o in used for o in nd.output)
                    and nd.domain in ("", CONTRIB))
            keep.append(not dead)
        if all(keep):
            break
        nodes = [nd for nd, k2 in zip(nodes, keep) if k2]
    used = {i for nd in nodes for i in nd.input} | outputs
    kept_init = [t for t in graph.initializer if t.name in used] + [t for t, _ in blobs]

    del graph.node[:]
    graph.node.extend(nodes)
    del graph.initializer[:]
    graph.initializer.extend(kept_init)
    if not any(i.domain == CONTRIB for i in model.opset_import):
        model.opset_import.append(helper.make_opsetid(CONTRIB, 1))

    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, bin_name), "wb") as f:
        for _, raw in blobs:
            f.write(raw)
    onnx.save(model, os.path.join(out_dir, graph_file))
    return stats


def quantize_bundle(src_dir: str, out_dir: str, graphs: list[str], bits: int, block_size: int,
                    link: bool = True, fp16_gemm: bool = False) -> dict:
    os.makedirs(out_dir, exist_ok=True)
    report = {}
    for g in graphs:
        if not os.path.exists(os.path.join(src_dir, g)):
            continue
        bin_name = g.replace(".onnx", "") + ".q8.bin"
        report[g] = quantize_graph(src_dir, g, out_dir, bin_name, bits, block_size,
                                  fp16_gemm).as_dict()
    produced = {g.replace(".onnx", "") + ".q8.bin" for g in report}
    for f in os.listdir(out_dir):
        if f.endswith(".q8.bin") and f not in produced:
            os.remove(os.path.join(out_dir, f))   # a graph quantized by an earlier run
    for f in os.listdir(src_dir):
        # A graph this run left alone (`--graphs model.onnx`) ships as the export made it.
        if f.endswith(".onnx") and f not in report:
            shutil.copy(os.path.join(src_dir, f), os.path.join(out_dir, f))

    decision = os.path.join(src_dir, "decision.json")
    if os.path.exists(decision):
        d = json.load(open(decision, encoding="utf-8"))
        d["precision"] = (f"fp32 compute; projection weights int8 blockwise (MatMulNBits, "
                          f"bits={bits}, block_size={block_size}, asymmetric)"
                          + (" widened per block and fed to the GEMM in fp16"
                             if fp16_gemm else " widened per block")
                          + "; embedding and norms BF16; pointer head F32")
        d["quantization"] = {"bits": bits, "block_size": block_size, "symmetric": False,
                             "fp16_gemm": fp16_gemm, "graphs": [g for g in report],
                             "tool": "families/neohorse/quantize.py"}
        report["decision.json"] = {"updated": True}
        json.dump(d, open(os.path.join(out_dir, "decision.json"), "w", encoding="utf-8"),
                  indent=1, ensure_ascii=False)
    for f in SIDECARS:
        if f == "decision.json":
            continue
        p = os.path.join(src_dir, f)
        if os.path.exists(p):
            shutil.copy(p, os.path.join(out_dir, f))
    if link:
        # The graph still reads the embedding, the norms and the head from the author's files.
        for f in sorted(os.listdir(src_dir)):
            if f.endswith((".safetensors", ".pt", ".pth")):
                dst = os.path.join(out_dir, f)
                if os.path.lexists(dst):
                    os.remove(dst)
                try:
                    os.link(os.path.realpath(os.path.join(src_dir, f)), dst)
                except OSError:
                    shutil.copy(os.path.join(src_dir, f), dst)
    return report


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("src", help="the exported bundle (weightless graphs + the author's shards)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--graphs", default="model.onnx",
                    help="model.onnx (the decoder) by default; add vision.onnx to compress the ViT "
                         "too. Measured: the decoder alone drifts ~6x the 2e-3 parity gate, and the "
                         "ViT adds a second error source of its own (max 1.5e-1 on image_embeds) for "
                         "only ~0.3 GiB, so the decoder is the default")
    ap.add_argument("--bits", type=int, default=8, help="8 only: it is the width with a CUDA kernel")
    ap.add_argument("--block-size", type=int, default=128)
    ap.add_argument("--no-link", action="store_true", help="copy the shards instead of hard-linking")
    ap.add_argument("--fp16-gemm", action="store_true",
                    help="run each quantized GEMM with fp16 activations (the fused tensor-core "
                         "kernel; 4x the fp32-activation path) while every other op stays fp32")
    a = ap.parse_args()
    report = quantize_bundle(a.src, a.out, [g.strip() for g in a.graphs.split(",") if g.strip()],
                             a.bits, a.block_size, link=not a.no_link, fp16_gemm=a.fp16_gemm)
    print(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()

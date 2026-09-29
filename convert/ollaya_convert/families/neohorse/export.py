"""Export NeoHorse-Jev-4B to two weightless ONNX graphs + tokenizer + decision/calibration config.

    NEOHORSE_ROOT=/path/to/NeoHorse-Jev-4B OLLAYA_FOLD_LIMIT=64 \
        python -m ollaya_convert.families.neohorse.export --out out/neohorse

Graphs (layout `neohorse-pointer-vision-v1`, see graphs.py, ref.py and layout.py):
    vision.onnx   patches [N, 1536], pos_idx [N, 4], pos_w [N, 4], rot_ids [N, 2] -> image_embeds [N/4, 2560]
    model.onnx    input_ids [R, T], position_ids [3, R, T], image_embeds [M, 2560], image_pos [M],
                  decide_idx [N], opt_idx [N] -> scores [N] (row-major over (row, option);
                  entry n belongs to row n // K, so the caller reshapes N = R * K)
Both reference the bundle's own files by byte offset: the three backbone shards (BF16), the pointer
head (`pointer_head.safetensors`, F32). Nothing is copied.

NeoHorse is merged (no adapter), so unlike `kev` there is no LoRA to keep unmerged.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil

import torch

from ..llm_common import onnx_export as ox
from ..llm_common.qwen35 import CHUNK
from ...weightless_sharded import safetensors_source
from . import ref
from .graphs import PointerDecoderGraph, VisionGraph
from .layout import NeohorseLayout

VISION_INPUTS = ["patches", "pos_idx", "pos_w", "rot_ids"]
VISION_OUTPUTS = ["image_embeds"]
INPUTS = ["input_ids", "position_ids", "image_embeds", "image_pos", "decide_idx", "opt_idx"]
OUTPUTS = ["scores"]

MAX_IMAGE_TOKENS = ref.MAX_IMAGE_TOKENS
MAX_OPTIONS = 255
SPECIAL = ["<|fim_prefix|>", "<|fim_middle|>", "<|box_start|>", "<|box_end|>", "<|fim_suffix|>"]


def rename(name):
    if name.startswith("v."):
        return ["visual." + name[2:]]
    if name.startswith("trunk.m."):
        return ["language_model." + name[len("trunk.m."):]]
    if name.startswith("head."):
        return [name[len("head."):]]  # q.weight / q.bias / k.weight / k.bias
    return [name]


def _backbone_files(root: Path):
    idx = json.loads((root / "backbone" / "model.safetensors.index.json").read_text())
    return sorted({Path(v).name for v in idx["weight_map"].values()})


def export(out_dir, root):
    root = Path(root)
    tok, backbone, head = ref.load(root, device="cpu", dtype=torch.float32)
    proc = ref.image_processor(backbone, root)
    vis = backbone.visual
    cfg = backbone.config

    decision_tokens = {k: tok.convert_tokens_to_ids(t) for k, t in
                       zip(("state", "question", "option_open", "option_close", "decide"), SPECIAL)}
    decision_tokens |= {"vision_start": cfg.vision_start_token_id, "vision_end": cfg.vision_end_token_id,
                        "image": cfg.image_token_id, "pad": tok.pad_token_id or 0}
    layout = NeohorseLayout(tok, {"special_tokens": decision_tokens, "tokens": decision_tokens,
                                  "max_state_tokens": 384, "max_row_tokens": 1024})

    # --- one image + one question (the visual adapter's shape) as the export example -------------
    image = ref.image_from_bytes(ref.demo_png())
    qs = {"colour": {"type": "choice", "instructions": "What colour is the square?",
                     "criteria": {"red": None, "green": None, "blue": None, "yellow": None}}}
    batch = proc(images=image, return_tensors="pt")
    grid = batch["image_grid_thw"][:1]
    n_img = int(grid.prod()) // (proc.merge_size ** 2)
    pv, pos_idx, pos_w, rot_ids = ref.patch_inputs(vis, grid, batch["pixel_values"][: int(grid.prod())])

    vg = VisionGraph(vis).eval()
    with torch.no_grad():
        emb = vg(pv, pos_idx, pos_w, rot_ids)
    print("image tokens", n_img, "embeds", tuple(emb.shape))

    rows, _ = layout.encode("A 256x240 image.", qs, image_tokens=n_img)
    row = rows[0]
    k = len(row["opts"])
    # Two identical rows, not one: with a single-row example `torch.export` resolves the trunk's
    # head-count reshapes against the (dynamic) row axis and emits `16 // rows` / `32 // rows`
    # instead of the constants 16 / 32, so `model.onnx` runs at R = 1 and fails at R > 1. A
    # two-row example keeps them static (the same recipe as `decider_vision/export.py`).
    R = 2
    ids = torch.tensor([row["ids"]] * R, dtype=torch.long)
    T = -(-ids.shape[1] // CHUNK) * CHUNK
    # one slack column per row: text-only requests park a zero image slot in the padding
    if T == ids.shape[1]:
        T += CHUNK
    ids2 = torch.zeros((R, T), dtype=torch.long)
    ids2[:, : ids.shape[1]] = ids
    mm = (ids2 == cfg.image_token_id).long()
    grid2 = grid.repeat(R, 1)  # `get_rope_index` consumes one grid per image in the batch
    pos, _ = backbone.get_rope_index(ids2, mm, image_grid_thw=grid2,
                                     attention_mask=torch.ones_like(ids2))
    where = (ids2 == cfg.image_token_id).nonzero()
    image_pos = where[:, 0] * T + where[:, 1]
    emb2 = torch.cat([emb] * R)  # the same image on every row

    dg = PointerDecoderGraph(backbone.language_model, head, cfg.text_config.rope_parameters["mrope_section"]).eval()
    # N = R * K flat readout indices, row-major: the decide token of row n // K, that row's option
    # n % K.
    decide_idx = torch.tensor([r * T + row["decide"] for r in range(R) for _ in range(k)], dtype=torch.long)
    opt_idx = torch.tensor([r * T + p for r in range(R) for p in row["opts"]], dtype=torch.long)
    args = (ids2, pos, emb2.to(torch.float32), image_pos, decide_idx, opt_idx)
    with torch.no_grad():
        got = dg(*args)
        want, _, _, _ = ref.reference_scores(tok, backbone, head, layout, "A 256x240 image.", qs, image, proc)
    diff = float((got.reshape(R, k) - torch.tensor(want[0])).abs().max())
    print("eager graph vs HF backbone: %.2e" % diff)

    tmp = ox.scratch_dir("neohorse-export-")
    Nt = torch.export.Dim("tokens", min=1, max=MAX_IMAGE_TOKENS)
    secs = ox.export_graph(vg, (pv, pos_idx, pos_w, rot_ids), VISION_INPUTS, VISION_OUTPUTS,
                           {"patches": {0: 4 * Nt}, "pos_idx": {0: 4 * Nt}, "pos_w": {0: 4 * Nt},
                            "rot_ids": {0: 4 * Nt}}, os.path.join(tmp, "vision", "model.onnx"))
    print("vision exported in %.0fs" % secs)
    Rd = torch.export.Dim("rows", min=1, max=1024)
    Nc = torch.export.Dim("chunks", min=1, max=256)
    M = torch.export.Dim("image_tokens", min=1, max=65536)
    # `decide_idx` and `opt_idx` share this: both are one entry per (row, option), row-major.
    N = torch.export.Dim("readout", min=1, max=1024 * MAX_OPTIONS)
    secs = ox.export_graph(dg, args, INPUTS, OUTPUTS,
                           {"input_ids": {0: Rd, 1: CHUNK * Nc}, "position_ids": {1: Rd, 2: CHUNK * Nc},
                            "image_embeds": {0: M}, "image_pos": {0: M}, "decide_idx": {0: N},
                            "opt_idx": {0: N}}, os.path.join(tmp, "decoder", "model.onnx"))
    print("decoder exported in %.0fs" % secs)

    shards = _backbone_files(root)
    src = [safetensors_source(f, str(root / "backbone" / f), repo="local", revision=None, filename=f)
           for f in shards] + [
        safetensors_source("pointer_head.safetensors", str(root / "pointer_head.safetensors"),
                           repo="local", revision=None, filename="pointer_head.safetensors")]
    os.makedirs(out_dir, exist_ok=True)
    vdir = os.path.join(out_dir, "vision")
    rep_v = ox.weightless(os.path.join(tmp, "vision"), vdir, src, rename)
    rep_d = ox.weightless(os.path.join(tmp, "decoder"), out_dir, src, rename)
    shutil.move(os.path.join(vdir, "model.onnx"), os.path.join(out_dir, "vision.onnx"))
    shutil.rmtree(vdir)
    ox.cleanup(tmp)
    shutil.copy(str(root / "tokenizer" / "tokenizer.json"), os.path.join(out_dir, "tokenizer.json"))

    ip = proc
    decision = {
        "engine": "onnx",
        "family": "neohorse",
        "layout": "neohorse-pointer-vision-v1",
        "upstream": {"repo": "NeoHorse-Jev-4B", "bundle": str(root),
                     "note": "merged Qwen3.5-4B multimodal backbone + independent pointer head"},
        "graphs": {"vision": "vision.onnx", "decoder": "model.onnx"},
        "max_state_tokens": 384,
        "max_row_tokens": 1024,
        "max_ctx_tokens": ref.MAX_ROW_TOKENS,
        "min_options": 1,
        "max_options": MAX_OPTIONS,
        "chunk": CHUNK,
        "special_tokens": {**decision_tokens, "texts": SPECIAL, "add_special_tokens": False},
        "user_text_escape": {"pattern": "<\\|([A-Za-z0-9_]+)\\|>", "replace": "<¦$1¦>"},
        "templates": {
            "row": "[fim_prefix] (vision prefix) user(render(state))[:max_state-1] "
                   "[fim_middle] user(render(instructions)) "
                   "([box_start] user(option) [box_end])* [fim_suffix]",
            "choice_option": "{name} | {name}: {render(description)}",
            "noul_options": ["no | no: {render(false)}", "yes | yes: {render(true)}"],
            "score_option": "{render(level)}",
            "render": "neohorse_decision render: None->'', scalars->Python str(), list->'- item' lines, "
                      "dict->'key: value' lines, 2-space nesting",
        },
        "image": {"patch_size": ip.patch_size, "temporal_patch_size": ip.temporal_patch_size,
                  "merge_size": ip.merge_size, "min_pixels": ip.size["shortest_edge"],
                  "max_pixels": ip.size["longest_edge"], "resample": "PIL bicubic",
                  "rescale_factor": ip.rescale_factor, "image_mean": list(ip.image_mean),
                  "image_std": list(ip.image_std), "position_table_side": vis.num_grid_per_side,
                  "position_interpolation": "bilinear, align_corners", "max_image_tokens": MAX_IMAGE_TOKENS},
        "mrope_section": cfg.text_config.rope_parameters["mrope_section"],
        "option_logits": {"shape": "scores is flat, one entry per (row, option) row-major, so row r "
                                   "owns scores[r * K : r * K + k_r]",
                          "choice": "row r's first k", "noul": "row r's first 2 (0 = no = false, 1 = yes = true)",
                          "score": "row r's first `levels`"},
        "opset": ox.OPSET,
        "precision": "fp32 compute; base weights BF16, widened by Cast; head F32",
        "weights_in_memory": ox.weights_in_memory(rep_d),
    }
    calibration = {"temperature": [1.0, 1.0, 1.0], "temperature_by_options": {},
                   "source": "model_manifest.json temperature (1.0)"}
    files = {
        "model": "neohorse",
        "weightless": {"vision": rep_v["stats"], "decoder": rep_d["stats"]},
        "unused_checkpoint_tensors": {"vision": {k: len(v) for k, v in rep_v["unused"].items()},
                                      "decoder": {k: len(v) for k, v in rep_d["unused"].items()}},
    }
    ox.write_json(os.path.join(out_dir, "decision.json"), decision)
    ox.write_json(os.path.join(out_dir, "calibration.json"), calibration)
    ox.write_json(os.path.join(out_dir, "files.json"), files)
    print(json.dumps(files["weightless"]))
    print("decoder graph MB %.1f" % (os.path.getsize(os.path.join(out_dir, "model.onnx")) / 2**20))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True)
    ap.add_argument("--root", default=None)
    a = ap.parse_args()
    export(a.out, a.root or ref.snapshot())


if __name__ == "__main__":
    main()

"""Parity of the weightless NeoHorse ONNX export against the HF backbone, on the CPU.

    python -m ollaya_convert.families.neohorse.parity --model-dir out/neohorse

Runs `vision.onnx` and `model.onnx` through onnxruntime (materialising the byte-referenced
backbone/head tensors) and compares the pointer scores with `ref.reference_scores` — the HF
`Qwen3_5Model` run the way NeoHorse's vision adapter runs it. This is the gate that matters: the
export is only trusted once the ONNX graphs reproduce the backbone, not just the eager
reimplementation (which export.py already checks to ~1e-6).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from ..llm_common.ort_rows import session
from ..llm_common.qwen35 import CHUNK
from . import ref
from .layout import NeohorseLayout


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--root", default=None)
    ap.add_argument("--tol", type=float, default=2e-3)
    a = ap.parse_args()

    md = Path(a.model_dir)
    decision = json.loads((md / "decision.json").read_text())
    root = Path(a.root or ref.snapshot())
    tok, backbone, head = ref.load(root, device="cpu", dtype=torch.float32)
    proc = ref.image_processor(backbone, root)
    cfg = backbone.config
    layout = NeohorseLayout(tok, {"special_tokens": decision["special_tokens"],
                                  "tokens": decision["special_tokens"],
                                  "max_state_tokens": decision["max_state_tokens"],
                                  "max_row_tokens": decision["max_row_tokens"]})

    image = ref.image_from_bytes(ref.demo_png())
    qs = {"colour": {"type": "choice", "instructions": "What colour is the square?",
                     "criteria": {"red": None, "green": None, "blue": None, "yellow": None}}}
    state = "A 256x240 image."

    # --- HF reference ---------------------------------------------------------------------------
    want, _, _, _ = ref.reference_scores(tok, backbone, head, layout, state, qs, image, proc)

    # --- graph inputs (same construction as export.py) ------------------------------------------
    batch = proc(images=image, return_tensors="pt")
    grid = batch["image_grid_thw"][:1]
    n_img = int(grid.prod()) // (proc.merge_size ** 2)
    pv, pos_idx, pos_w, rot_ids = ref.patch_inputs(backbone.visual, grid,
                                                   batch["pixel_values"][: int(grid.prod())])
    rows, _ = layout.encode(state, qs, image_tokens=n_img)
    row = rows[0]
    ids = torch.tensor([row["ids"]], dtype=torch.long)
    T = -(-ids.shape[1] // CHUNK) * CHUNK
    if T == ids.shape[1]:
        T += CHUNK
    ids2 = torch.zeros((1, T), dtype=torch.long)
    ids2[:, : ids.shape[1]] = ids
    mm = (ids2 == cfg.image_token_id).long()
    pos, _ = backbone.get_rope_index(ids2, mm, image_grid_thw=grid,
                                     attention_mask=torch.ones_like(ids2))
    where = (ids2 == cfg.image_token_id).nonzero()
    image_pos = (where[:, 0] * T + where[:, 1]).numpy().astype(np.int64)

    vs = session(str(md / "vision.onnx"))
    emb = vs.run(None, {"patches": pv.numpy().astype(np.float32),
                        "pos_idx": pos_idx.numpy().astype(np.int64),
                        "pos_w": pos_w.numpy().astype(np.float32),
                        "rot_ids": rot_ids.numpy().astype(np.int64)})[0]
    print("vision.onnx -> image_embeds", emb.shape)
    ds = session(str(md / "model.onnx"))
    # N = R * K flat readout indices, row-major: row n // K's decide token, that row's option n % K.
    got = ds.run(None, {
        "input_ids": ids2.numpy().astype(np.int64),
        "position_ids": pos.numpy().astype(np.int64),
        "image_embeds": emb.astype(np.float32),
        "image_pos": image_pos,
        "decide_idx": np.full(len(row["opts"]), row["decide"], dtype=np.int64),
        "opt_idx": np.array(row["opts"], dtype=np.int64),
    })[0]

    k = len(row["opts"])
    w = np.asarray(want[0], dtype=np.float32)[:k]
    g = np.asarray(got, dtype=np.float32)[:k]
    diff = float(np.abs(w - g).max())
    print("HF   ", np.round(w, 4).tolist())
    print("ONNX ", np.round(g, 4).tolist())
    print("max|HF - ONNX| = %.3e  (tol %.1e)  %s" % (diff, a.tol, "PASS" if diff <= a.tol else "FAIL"))
    raise SystemExit(0 if diff <= a.tol else 1)


if __name__ == "__main__":
    main()

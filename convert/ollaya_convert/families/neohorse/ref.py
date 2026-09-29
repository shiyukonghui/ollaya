"""The PyTorch reference for NeoHorse-Jev-4B, and the pieces the export needs.

NeoHorse ships as a merged bundle: `backbone/` is a `Qwen3_5Model` (multimodal, LoRA already merged
into the tensors) and `pointer_head.safetensors` is the head (`q`/`k` projections + scale). There is
no upstream Python package to call, so the reference *is* the HF backbone: run `backbone(...)` the
way `neohorse_decision.vision` does (image prefix, mm_token_type_ids, no cache) and apply the head at
the shifted decide/option positions. The exported graph reimplements that forward (see graphs.py).

    NEOHORSE_ROOT=/path/to/NeoHorse-Jev-4B python -m ollaya_convert.families.neohorse.ref
"""
from __future__ import annotations

import io
import os
from pathlib import Path

import torch
import torch.nn as nn
from safetensors.torch import load_file

DEFAULT_ROOT = os.environ.get("NEOHORSE_ROOT", r"D:/jev-like/models/NeoHorse-Jev-4B")
# NeoHorse's vision adapter caps the processed image at 1024 visual tokens and the row at 12288.
MAX_IMAGE_TOKENS = 1024
MAX_ROW_TOKENS = 12288


def snapshot():
    root = os.environ.get("NEOHORSE_ROOT", DEFAULT_ROOT)
    if not Path(root).exists():
        raise SystemExit("set NEOHORSE_ROOT to the NeoHorse-Jev-4B bundle directory")
    return root


class PointerHead(nn.Module):
    """`neohorse_decision._vendor.model.PointerHead`, reproduced so the converter needs no package."""

    def __init__(self, d: int, dp: int = 256):
        super().__init__()
        self.q, self.k = nn.Linear(d, dp), nn.Linear(d, dp)
        self.scale = 1 / (dp ** 0.5)

    def forward(self, h_decide, h_opts):
        return (self.k(h_opts) @ self.q(h_decide).unsqueeze(-1)).squeeze(-1) * self.scale


def load(root=None, device="cpu", dtype=torch.float32):
    """-> (tok, backbone, head). `backbone` is `Qwen3_5Model` with `.visual` and `.language_model`."""
    from transformers import AutoModel, AutoTokenizer

    root = Path(root or snapshot())
    tok = AutoTokenizer.from_pretrained(root / "tokenizer", local_files_only=True)
    backbone = AutoModel.from_pretrained(
        root / "backbone", dtype=dtype, attn_implementation="sdpa", local_files_only=True
    ).to(device).eval()
    head = PointerHead(backbone.config.text_config.hidden_size)
    head.load_state_dict(load_file(str(root / "pointer_head.safetensors")))
    return tok, backbone.to(device), head.to(device).eval()


def image_processor(backbone, root=None):
    """NeoHorse's processor: the PIL backend, NeoHorse's pixel budgets."""
    from transformers.models.qwen2_vl.image_processing_pil_qwen2_vl import Qwen2VLImageProcessorPil

    root = Path(root or snapshot())
    return Qwen2VLImageProcessorPil.from_pretrained(
        root / "backbone", local_files_only=True,
        size={"shortest_edge": 65536, "longest_edge": 1048576},
    )


def patch_inputs(visual, grid, pv):
    """The three grid-dependent tables the VisionGraph takes, exactly as the processor computes them."""
    from transformers.vision_utils import (
        get_vision_interpolation_indices_and_weights,
        get_vision_position_ids,
    )

    idx, wt = get_vision_interpolation_indices_and_weights(
        grid, visual.num_grid_per_side, mode="bilinear", align_corners=True,
        spatial_merge_size=visual.spatial_merge_size,
    )
    rot = get_vision_position_ids(grid, visual.spatial_merge_size)
    return pv, idx, wt, rot


def image_from_bytes(data):
    from PIL import Image

    return Image.open(io.BytesIO(data)).convert("RGB")


def demo_png():
    from PIL import Image, ImageDraw

    img = Image.new("RGB", (256, 240), "white")
    d = ImageDraw.Draw(img)
    d.rectangle([40, 40, 140, 140], fill="red")
    d.ellipse([160, 120, 230, 200], fill="blue")
    buf = io.BytesIO()
    img.save(buf, "PNG")
    return buf.getvalue()


@torch.no_grad()
def reference_scores(tok, backbone, head, layout, state, questions, image, proc=None):
    """HF backbone run the way NeoHorse's vision adapter does; raw pointer scores per question.

    -> (scores [Q][k], ids [1, T], position_ids [3, 1, T], image_embeds, image_pos, decide, opts)."""
    proc = proc or image_processor(backbone)
    batch = proc(images=image.convert("RGB"), return_tensors="pt") if image is not None else None
    n_img = 0
    if batch is not None:
        grid = batch["image_grid_thw"][:1]
        n_img = int(grid.prod()) // (proc.merge_size ** 2)
    rows, _ = layout.encode(state, questions, image_tokens=n_img if batch is not None else None)
    row = rows[0]
    ids = torch.tensor([row["ids"]], dtype=torch.long)
    mm = torch.zeros_like(ids)
    toks = {}
    if batch is not None:
        toks = {k: batch[k] for k in ("pixel_values", "image_grid_thw")}
        mm = (ids == backbone.config.image_token_id).long()
    out = backbone(input_ids=ids, attention_mask=torch.ones_like(ids), mm_token_type_ids=mm,
                   use_cache=False, **toks).last_hidden_state[0]
    hd = out[row["decide"]]
    ho = out[torch.tensor(row["opts"])]
    return (head(hd, ho).float().cpu().numpy(),), ids, row, batch


def demo_questions():
    return {"colour": {"type": "choice", "instructions": "What colour is the square?",
                       "criteria": {"red": None, "green": None, "blue": None, "yellow": None}},
            "circle": {"type": "noul", "instructions": "Is there a circle in the image?"}}


if __name__ == "__main__":
    from .layout import NeohorseLayout

    root = snapshot()
    tok, backbone, head = load(root)
    proc = image_processor(backbone, root)
    decision = {"special_tokens": {k: tok.convert_tokens_to_ids(t) for k, t in
                                   zip(("state", "question", "option_open", "option_close", "decide"),
                                       ["<|fim_prefix|>", "<|fim_middle|>", "<|box_start|>", "<|box_end|>", "<|fim_suffix|>"])},
                "tokens": {"vision_start": backbone.config.vision_start_token_id,
                           "vision_end": backbone.config.vision_end_token_id,
                           "image": backbone.config.image_token_id},
                "max_state_tokens": 384, "max_row_tokens": 1024}
    layout = NeohorseLayout(tok, decision)
    # one question: the visual adapter serves one per request
    qs = {"colour": {"type": "choice", "instructions": "What colour is the square?",
                     "criteria": {"red": None, "green": None, "blue": None, "yellow": None}}}
    scores, ids, row, batch = reference_scores(tok, backbone, head, layout,
                                               "A 256x240 image.", qs, image_from_bytes(demo_png()), proc)
    print("row", {k: (v if k != "ids" else len(v)) for k, v in row.items()})
    print("scores", scores[0])

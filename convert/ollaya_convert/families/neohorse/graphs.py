"""Export-friendly forwards for NeoHorse-Jev-4B (Qwen3.5-4B multimodal backbone + pointer head).

The readout is a pointer head over two positions of the backbone's final hidden state: the decide
token and each option's closing token — the `kev-pointer-v1` primitive. The backbone is
multimodal, so the decoder graph also takes the image's visual tokens and interleaved M-RoPE
positions — the `decider-vision-v1` machinery.

Both building blocks are reused verbatim from the sibling families:
  * `VisionGraph`     — the Qwen3.5 ViT (same family; NeoHorse's vision tower is Qwen3.5-4B's).
  * `MropeTrunk`      — the hybrid Gated-DeltaNet + gated-attention trunk with multimodal RoPE.
Only the readout differs: a pointer head instead of a letter head.
"""
from __future__ import annotations

import torch
import torch.nn as nn

# Reused verbatim: the vision tower and the multimodal trunk are the same Qwen3.5 pieces.
from ..decider_vision.graphs import MropeTrunk, VisionGraph  # noqa: F401

__all__ = ["VisionGraph", "PointerDecoderGraph"]


class PointerDecoderGraph(nn.Module):
    """input_ids [R, T] (T a multiple of 64, right-padded), position_ids [3, R, T],
    image_embeds [M, H] written at flat positions image_pos [M] (row * T + column),
    decide_idx / opt_idx [N] -> raw pointer scores [N].

    The readout takes **flat** indices, exactly like `image_pos`: entry n is row `n // K`'s decide
    token (`decide_idx`) and its option `n % K` (`opt_idx`), so `scores[n]` is that option's logit
    and the caller reshapes the `N = R * K` values back to rows. A 2-D advanced index
    (`h[rows[:, None], opt_pos]`) instead makes `torch.export` fold shape arithmetic it cannot keep
    symbolic — it derives the trunk's chunk reshape from the index's shape and emits a constant
    `floordiv` that is 0 for `R > 1` (see `export.py`).

    `scores[r * K : r * K + k_r]` are question r's option logits (the head's own temperature is
    applied by the runtime, exactly as for `kev-pointer-v1`)."""

    def __init__(self, text_model, head, mrope_section):
        super().__init__()
        self.trunk = MropeTrunk(text_model, mrope_section)
        self.head = head
        self.scale = float(head.scale)

    def forward(self, input_ids, position_ids, image_embeds, image_pos, decide_idx, opt_idx):
        R, T = input_ids.shape
        x = self.trunk.m.embed_tokens(input_ids)
        flat = x.reshape(R * T, x.shape[-1]).index_copy(0, image_pos, image_embeds.to(x.dtype))
        h = self.trunk.embeds(flat.reshape(R, T, -1), position_ids)
        hf = h.reshape(R * T, -1)
        q = self.head.q(hf.index_select(0, decide_idx))   # [N, P]
        k = self.head.k(hf.index_select(0, opt_idx))      # [N, P]
        return (k * q).sum(-1) * self.scale               # [N]

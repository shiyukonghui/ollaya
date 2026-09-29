"""`neohorse-pointer-vision-v1`: the request -> token rows layout of NeoHorse-Jev-4B.

NeoHorse's row form is byte-for-byte the kev row form (`SPECIAL` and `to_record` are the same five
Qwen delimiters and the same rendering), so the text half of this layout is `kev.layout`'s. The only
addition is vision: when an image is present, its visual tokens are spliced in right after the
leading `<|fim_prefix|>`, i.e. `[fim_prefix] + [vision_start] + [image]*n + [vision_end] + state...`.

      row = [fim_prefix] (+ vision prefix) + user(render(state))[:max_state-1]
          + [fim_middle] + user(render(instructions))
          + for each option: [box_start] + user(option_text) + [box_end]
          + [fim_suffix]
      decide_pos = len(row) - 1;  opt_pos[j] = index of option j's <|box_end|>

Upstream's visual adapter serves **one question per request** (it repeats the same image over rows);
this layout keeps that rule: rows > 1 with an image is an error.
"""
from __future__ import annotations

from typing import Any, Dict, List

from ..kev.layout import (  # noqa: F401  (re-exported: identical rendering)
    MAX_OPTIONS,
    SPECIAL,
    LayoutError,
    option_text,
    render,
    to_record,
)

# NeoHorse's own serving limits (`neohorse_decision._vendor.model`).
MAX_STATE = 384
MAX_BRANCH = 1024


class NeohorseLayout:
    """The text half is `kev.layout.KevLayout`; vision adds a prefix of image tokens."""

    def __init__(self, tokenizer, decision: Dict[str, Any]):
        self.tok = tokenizer
        self.max_state = decision.get("max_state_tokens", MAX_STATE)
        self.max_row = decision.get("max_row_tokens", MAX_BRANCH)
        sp = decision["special_tokens"]
        self.state_id, self.q_id, self.o_id, self.c_id, self.d_id = (
            sp[k] for k in ("state", "question", "option_open", "option_close", "decide")
        )
        tok = decision.get("tokens", {})
        self.vision_start = tok.get("vision_start")
        self.vision_end = tok.get("vision_end")
        self.image_id = tok.get("image")

    def encode(self, state, questions, image_tokens: int | None = None):
        """-> (rows, meta). rows[q] = {"ids", "decide", "opts"}; `image_tokens` splices the prefix."""
        state_text, qs, meta = to_record(state, questions)
        prefix: List[int] = []
        if image_tokens is not None:
            if len(qs) != 1:
                raise LayoutError("NeoHorse's visual adapter supports one question per request")
            prefix = [self.vision_start] + [self.image_id] * image_tokens + [self.vision_end]
        S = [self.state_id] + self.user(state_text)[: self.max_state - 1]
        rows = []
        for instr, opts in qs:
            head = S[:1] + prefix + S[1:]
            br = [self.q_id] + self.user(instr)
            ends = []
            for o in opts:
                br += [self.o_id] + self.user(o) + [self.c_id]
                ends.append(len(head) + len(br) - 1)
            br.append(self.d_id)
            if len(br) > self.max_row - len(head):
                raise LayoutError("branch too long: %d tokens with a %d-token state (row limit %d)"
                                  % (len(br), len(head), self.max_row))
            ids = head + br
            rows.append({"ids": ids, "decide": len(ids) - 1, "opts": ends})
        return rows, meta

    def user(self, text: str) -> List[int]:
        from ..kev.layout import _SPECIAL_RE
        enc = self.tok.encode(_SPECIAL_RE.sub(r"<¦\1¦>", text), add_special_tokens=False)
        return list(enc.ids) if hasattr(enc, "ids") else list(enc)

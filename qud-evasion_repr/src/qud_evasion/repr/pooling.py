"""Token pooling for decoder hidden states.

Three fixed poolings are computed at extraction time and cached, so all
downstream probing is a cheap read:

  last        -- hidden state of the final non-padding token (the classic
                 decoder probe target; reproduces probe_experiment_v2).
  mean_all    -- mean over all non-padding tokens.
  mean_answer -- mean over tokens belonging to the answer span only.
                 This is the pooling that should benefit most from the
                 allocation structure: the judgement is about the answer,
                 the question is context.

A learned attention pooling lives in heads.py (it needs token-level
states for a chosen layer band, cached separately via
``--token-layers``).
"""

from __future__ import annotations

import torch

POOLINGS = ("last", "mean_all", "mean_answer")


def answer_token_mask(
    offsets: torch.Tensor,        # [B, T, 2] char offsets from the tokenizer
    attention_mask: torch.Tensor, # [B, T]
    answer_char_spans: list[tuple[int, int]],
) -> torch.Tensor:
    """Mark tokens whose character span overlaps the answer span. [B, T] bool."""
    starts, ends = offsets[..., 0], offsets[..., 1]
    mask = torch.zeros_like(attention_mask, dtype=torch.bool)
    for b, (a_start, a_end) in enumerate(answer_char_spans):
        overlap = (ends[b] > a_start) & (starts[b] < a_end)
        mask[b] = overlap & attention_mask[b].bool() & (ends[b] > starts[b])
        if not mask[b].any():  # answer truncated away: fall back to last token
            last = int(attention_mask[b].sum().item()) - 1
            mask[b, last] = True
    return mask


def pool_layer(
    hidden: torch.Tensor,          # [B, T, D] one layer
    attention_mask: torch.Tensor,  # [B, T]
    ans_mask: torch.Tensor,        # [B, T] bool
) -> dict[str, torch.Tensor]:
    """Return {pooling_name: [B, D]} for one layer, in float32 on CPU."""
    am = attention_mask.unsqueeze(-1).to(hidden.dtype)

    lengths = attention_mask.sum(dim=1)                       # [B]
    last_idx = (lengths - 1).clamp(min=0).long()              # [B]
    last = hidden[torch.arange(hidden.size(0)), last_idx]     # [B, D]

    mean_all = (hidden * am).sum(1) / am.sum(1).clamp(min=1)

    ans = ans_mask.unsqueeze(-1).to(hidden.dtype)
    mean_answer = (hidden * ans).sum(1) / ans.sum(1).clamp(min=1)

    return {
        "last": last.float().cpu(),
        "mean_all": mean_all.float().cpu(),
        "mean_answer": mean_answer.float().cpu(),
    }

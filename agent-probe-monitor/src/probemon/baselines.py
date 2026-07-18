"""Black-box baselines the probes must beat.

perplexity     mean NLL of the observation text under the monitored model
               (injections are often distributionally odd in context).
action_logprob mean log-prob the model assigns to its own generated action
               (low confidence can correlate with being steered off-task).
llamaguard     output-monitor baseline: score obs/action text with a safety
               classifier LM (default meta-llama/Llama-Guard-3-8B); score is
               p("unsafe") read from the first generated token.

All baselines return real-valued scores aligned with a metadata frame, so
evaluate.py treats them identically to probe scores.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


@torch.no_grad()
def text_nll(model, tokenizer, texts: list[str], device, max_len: int = 2048
             ) -> np.ndarray:
    scores = []
    for text in texts:
        ids = tokenizer(text, return_tensors="pt", truncation=True,
                        max_length=max_len).input_ids.to(device)
        if ids.shape[1] < 2:
            scores.append(0.0)
            continue
        out = model(ids, labels=ids)
        scores.append(out.loss.item())
    return np.array(scores)


@torch.no_grad()
def action_logprob(model, tokenizer, contexts: list[str], actions: list[str],
                   device, max_len: int = 4096) -> np.ndarray:
    """Mean log-prob of action tokens given context; returns NEGATIVE mean
    logprob so that higher = more suspicious, matching other scores."""
    scores = []
    for ctx, act in zip(contexts, actions):
        ctx_ids = tokenizer(ctx, return_tensors="pt", truncation=True,
                            max_length=max_len - 512).input_ids
        act_ids = tokenizer(act, add_special_tokens=False,
                            return_tensors="pt").input_ids
        ids = torch.cat([ctx_ids, act_ids], dim=1).to(device)
        labels = ids.clone()
        labels[0, : ctx_ids.shape[1]] = -100
        out = model(ids, labels=labels)
        scores.append(out.loss.item())
    return np.array(scores)


class LlamaGuardMonitor:
    def __init__(self, model_name: str = "meta-llama/Llama-Guard-3-8B",
                 dtype: str = "bfloat16", device_map: str = "auto"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name, torch_dtype=getattr(torch, dtype), device_map=device_map
        )
        self.model.eval()
        ids_unsafe = self.tokenizer("unsafe", add_special_tokens=False).input_ids
        self.unsafe_id = ids_unsafe[0]

    @torch.no_grad()
    def score_texts(self, texts: list[str]) -> np.ndarray:
        scores = []
        for text in texts:
            chat = [{"role": "user", "content": text}]
            ids = self.tokenizer.apply_chat_template(
                chat, return_tensors="pt"
            ).to(self.model.device)
            logits = self.model(ids).logits[0, -1]
            probs = torch.softmax(logits.float(), dim=-1)
            scores.append(probs[self.unsafe_id].item())
        return np.array(scores)


def run_baselines(meta: pd.DataFrame, model, tokenizer,
                  which: list[str], device,
                  llamaguard: LlamaGuardMonitor | None = None
                  ) -> dict[str, np.ndarray]:
    out: dict[str, np.ndarray] = {}
    if "perplexity" in which:
        out["perplexity"] = text_nll(model, tokenizer,
                                     meta["obs_text"].tolist(), device)
    if "action_logprob" in which:
        out["action_logprob"] = action_logprob(
            model, tokenizer, meta["obs_text"].tolist(),
            meta["action_text"].tolist(), device)
    if "llamaguard" in which:
        lg = llamaguard or LlamaGuardMonitor()
        joined = (meta["obs_text"] + "\n\n" + meta["action_text"]).tolist()
        out["llamaguard"] = lg.score_texts(joined)
    return out

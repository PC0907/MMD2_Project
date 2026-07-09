"""Stage 2: run the model over the dataset and cache residual-stream activations.

Saves one .npz per model containing:
    X       float16 array [n_examples, n_layers, hidden_dim]  (post-block residual)
    meta    parallel list of record dicts (json string)

Activations are taken at the LAST PROMPT TOKEN after applying the chat template
(i.e. the position from which the model would begin generating). This is the
standard choice for belief/context probes; token_position="mean" averages over
the prompt instead.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from .utils import ensure_dir, read_jsonl


def load_model(model_name: str, dtype: str = "bfloat16", device_map: str = "auto"):
    tok = AutoTokenizer.from_pretrained(model_name)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=getattr(torch, dtype),
        device_map=device_map,
    )
    model.eval()
    return model, tok


def _decoder_layers(model):
    """Return the list of transformer blocks for Llama/Qwen/Mistral-style models."""
    for attr in ("model", "transformer"):
        base = getattr(model, attr, None)
        if base is not None and hasattr(base, "layers"):
            return base.layers
        if base is not None and hasattr(base, "h"):
            return base.h
    raise ValueError(f"Cannot locate decoder layers on {type(model)}")


@torch.no_grad()
def extract(
    model,
    tok,
    records: list[dict],
    layers: list[int] | str = "all",
    token_position: str = "last",
    batch_size: int = 8,
    max_length: int = 2048,
) -> tuple[np.ndarray, list[dict]]:
    blocks = _decoder_layers(model)
    layer_idxs = list(range(len(blocks))) if layers == "all" else list(layers)

    captured: dict[int, torch.Tensor] = {}

    def make_hook(idx):
        def hook(_module, _inp, out):
            hidden = out[0] if isinstance(out, tuple) else out
            captured[idx] = hidden.detach()
        return hook

    handles = [blocks[i].register_forward_hook(make_hook(i)) for i in layer_idxs]

    all_X: list[np.ndarray] = []
    kept_meta: list[dict] = []
    try:
        for start in tqdm(range(0, len(records), batch_size), desc="extract"):
            batch = records[start : start + batch_size]
            texts = [
                tok.apply_chat_template(
                    [{"role": "user", "content": r["prompt"]}],
                    tokenize=False,
                    add_generation_prompt=True,
                )
                for r in batch
            ]
            enc = tok(
                texts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=max_length,
            ).to(model.device)

            captured.clear()
            model(**enc)

            attn = enc["attention_mask"]  # [B, T]
            last_idx = attn.sum(dim=1) - 1  # index of last real token
            per_layer = []
            for i in layer_idxs:
                h = captured[i]  # [B, T, D]
                if token_position == "last":
                    vec = h[torch.arange(h.size(0)), last_idx]  # [B, D]
                elif token_position == "mean":
                    m = attn.unsqueeze(-1).to(h.dtype)
                    vec = (h * m).sum(1) / m.sum(1)
                else:
                    raise ValueError(token_position)
                per_layer.append(vec.float().cpu())
            # [B, L, D]
            all_X.append(torch.stack(per_layer, dim=1).to(torch.float16).numpy())
            kept_meta.extend(batch)
    finally:
        for h in handles:
            h.remove()

    return np.concatenate(all_X, axis=0), kept_meta


def run(cfg: dict) -> Path:
    records = read_jsonl(cfg["generated_dataset"])
    model, tok = load_model(cfg["model_name"], cfg["dtype"], cfg["device_map"])
    X, meta = extract(
        model,
        tok,
        records,
        layers=cfg["layers"],
        token_position=cfg["token_position"],
        batch_size=cfg["batch_size"],
        max_length=cfg["max_length"],
    )
    out_dir = ensure_dir(cfg["activations_dir"])
    safe_name = cfg["model_name"].replace("/", "__")
    out = out_dir / f"{safe_name}.npz"
    np.savez_compressed(out, X=X, meta=json.dumps(meta))
    print(f"saved {X.shape} -> {out}")
    return out

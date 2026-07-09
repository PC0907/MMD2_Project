"""Hidden-state extraction with disk caching.

Design: extraction is the only expensive step (one forward pass per
example per model). Everything downstream -- layer sweeps, poolings,
heads, controls, seeds -- reads the cache and runs in seconds on CPU.
Extract once, iterate forever.

Cache layout (one directory per model x split):

    outputs/activations/<model_tag>/<split>/
        pooled.safetensors      # {pooling: [N, L+1, D] float16}
        token_states.safetensors  # optional, {"layer_<k>": [N, T_max, D]}
        index.json              # example ids, label arrays, prompt hash,
                                # tokenizer/config fingerprint

Disk cost for QEvasion train on Qwen3-4B (~3.4k x 37 layers x 2560d,
3 poolings, fp16): ~2 GB. Cache all layers; never re-extract to try a
new layer.

Reproducibility guards: the cache stores a fingerprint of (model name,
revision, prompt template, max_length). Loading with a mismatched
fingerprint raises instead of silently mixing representations.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from safetensors.numpy import load_file, save_file

from .data_adapter import Example
from .pooling import POOLINGS, answer_token_mask, pool_layer

# ---------------------------------------------------------------------------
# Prompt construction
# ---------------------------------------------------------------------------

# Keep this template IDENTICAL across arms (prompting arm prepends its
# instruction to the same core rendering). Changing it invalidates caches
# -- which the fingerprint enforces.
DEFAULT_TEMPLATE = "Question: {question}\nAnswer: {answer}"


@dataclass
class ExtractionConfig:
    model_name: str
    revision: Optional[str] = None
    template: str = DEFAULT_TEMPLATE
    use_full_question: bool = False   # False: isolated sub-question (recommended)
    use_chat_template: bool = False   # True: wrap in the model's chat template
    max_length: int = 1024
    batch_size: int = 8
    dtype: str = "bfloat16"
    device_map: str = "auto"
    token_layers: tuple[int, ...] = ()  # layers to cache token-level states for
    output_root: str = "outputs/activations"

    @property
    def model_tag(self) -> str:
        return self.model_name.replace("/", "__")

    def fingerprint(self) -> str:
        payload = json.dumps(
            {
                "model": self.model_name,
                "revision": self.revision,
                "template": self.template,
                "full_q": self.use_full_question,
                "chat": self.use_chat_template,
                "max_length": self.max_length,
            },
            sort_keys=True,
        )
        return hashlib.sha256(payload.encode()).hexdigest()[:16]


def render_prompt(cfg: ExtractionConfig, ex: Example, tokenizer=None) -> tuple[str, tuple[int, int]]:
    """Render one example; return (text, answer char span within text)."""
    question = ex.full_question if cfg.use_full_question and ex.full_question else ex.sub_question
    text = cfg.template.format(question=question, answer=ex.answer)
    a_start = text.rindex(ex.answer) if ex.answer and ex.answer in text else len(text)
    span = (a_start, a_start + len(ex.answer))

    if cfg.use_chat_template and tokenizer is not None:
        wrapped = tokenizer.apply_chat_template(
            [{"role": "user", "content": text}],
            tokenize=False,
            add_generation_prompt=True,
        )
        offset = wrapped.index(text)
        span = (span[0] + offset, span[1] + offset)
        text = wrapped
    return text, span


# ---------------------------------------------------------------------------
# Extraction
# ---------------------------------------------------------------------------


def extract(cfg: ExtractionConfig, examples: list[Example], split: str, force: bool = False) -> Path:
    """Run the forward passes and write the cache. Returns the cache dir."""
    out_dir = Path(cfg.output_root) / cfg.model_tag / split
    index_path = out_dir / "index.json"
    if index_path.exists() and not force:
        existing = json.loads(index_path.read_text())
        if existing.get("fingerprint") == cfg.fingerprint():
            print(f"[extract] cache hit: {out_dir}")
            return out_dir
        raise RuntimeError(
            f"Cache at {out_dir} was built with a different fingerprint "
            f"({existing.get('fingerprint')} != {cfg.fingerprint()}). "
            f"Use force=True to overwrite, or change output_root."
        )
    out_dir.mkdir(parents=True, exist_ok=True)

    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name, revision=cfg.revision)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        cfg.model_name,
        revision=cfg.revision,
        torch_dtype=getattr(torch, cfg.dtype),
        device_map=cfg.device_map,
        output_hidden_states=True,
    )
    model.eval()

    # Sort by rendered length for efficient batching; restore order at the end.
    rendered = [render_prompt(cfg, ex, tokenizer) for ex in examples]
    order = sorted(range(len(examples)), key=lambda i: len(rendered[i][0]))

    n = len(examples)
    pooled_buffers: dict[str, Optional[np.ndarray]] = {p: None for p in POOLINGS}
    token_buffers: dict[int, list] = {k: [None] * n for k in cfg.token_layers}
    truncated = 0

    with torch.inference_mode():
        for start in range(0, n, cfg.batch_size):
            batch_ids = order[start : start + cfg.batch_size]
            texts = [rendered[i][0] for i in batch_ids]
            spans = [rendered[i][1] for i in batch_ids]

            enc = tokenizer(
                texts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=cfg.max_length,
                return_offsets_mapping=True,
            )
            offsets = enc.pop("offset_mapping")
            n_tok_untrunc = [len(tokenizer(t).input_ids) for t in texts]
            truncated += sum(1 for t in n_tok_untrunc if t > cfg.max_length)

            enc = {k: v.to(model.device) for k, v in enc.items()}
            out = model(**enc)
            hidden_states = out.hidden_states  # tuple of L+1 tensors [B, T, D]

            attn = enc["attention_mask"].cpu()
            ans_mask = answer_token_mask(offsets, attn, spans)

            n_layers = len(hidden_states)
            for layer, h in enumerate(hidden_states):
                pooled = pool_layer(h.cpu(), attn, ans_mask)
                for pname, vec in pooled.items():
                    if pooled_buffers[pname] is None:
                        d = vec.shape[-1]
                        pooled_buffers[pname] = np.zeros((n, n_layers, d), dtype=np.float16)
                    for row, i in enumerate(batch_ids):
                        pooled_buffers[pname][i, layer] = vec[row].numpy().astype(np.float16)
                if layer in cfg.token_layers:
                    for row, i in enumerate(batch_ids):
                        keep = attn[row].bool()
                        token_buffers[layer][i] = h[row][keep.to(h.device)].float().cpu().numpy().astype(np.float16)

            if (start // cfg.batch_size) % 20 == 0:
                print(f"[extract] {min(start + cfg.batch_size, n)}/{n}")

    save_file({p: b for p, b in pooled_buffers.items() if b is not None}, str(out_dir / "pooled.safetensors"))

    if cfg.token_layers:
        # ragged -> pad to the max length among cached examples
        for layer, rows in token_buffers.items():
            t_max = max(r.shape[0] for r in rows)
            d = rows[0].shape[1]
            arr = np.zeros((n, t_max, d), dtype=np.float16)
            lens = np.zeros(n, dtype=np.int32)
            for i, r in enumerate(rows):
                arr[i, : r.shape[0]] = r
                lens[i] = r.shape[0]
            save_file(
                {"states": arr, "lengths": lens},
                str(out_dir / f"token_states_layer{layer}.safetensors"),
            )

    index = {
        "fingerprint": cfg.fingerprint(),
        "model": cfg.model_name,
        "split": split,
        "n_examples": n,
        "n_layers_incl_embedding": len(hidden_states),
        "truncated_examples": truncated,
        "ids": [ex.id for ex in examples],
        "interview_ids": [ex.interview_id for ex in examples],
        "clarity_labels": [ex.clarity_label for ex in examples],
        "evasion_labels": [ex.evasion_label for ex in examples],
        "position_in_turn": [ex.position_in_turn for ex in examples],
        "turn_size": [ex.turn_size for ex in examples],
    }
    index_path.write_text(json.dumps(index))
    print(f"[extract] wrote {out_dir} (truncated: {truncated}/{n})")
    return out_dir


# ---------------------------------------------------------------------------
# Cache loading
# ---------------------------------------------------------------------------


@dataclass
class ActivationCache:
    pooled: dict[str, np.ndarray]     # {pooling: [N, L+1, D]}
    index: dict

    @property
    def n_layers(self) -> int:
        return self.index["n_layers_incl_embedding"]

    def labels(self, task: str) -> np.ndarray:
        key = {"clarity": "clarity_labels", "evasion": "evasion_labels"}[task]
        return np.array([-1 if v is None else v for v in self.index[key]])

    def groups(self) -> np.ndarray:
        return np.array(self.index["interview_ids"])

    def features(self, pooling: str, layer: int) -> np.ndarray:
        return self.pooled[pooling][:, layer, :].astype(np.float32)


def load_cache(output_root: str, model_name: str, split: str, expected_fingerprint: Optional[str] = None) -> ActivationCache:
    out_dir = Path(output_root) / model_name.replace("/", "__") / split
    index = json.loads((out_dir / "index.json").read_text())
    if expected_fingerprint and index["fingerprint"] != expected_fingerprint:
        raise RuntimeError(f"Fingerprint mismatch for {out_dir}")
    pooled = load_file(str(out_dir / "pooled.safetensors"))
    return ActivationCache(pooled=pooled, index=index)

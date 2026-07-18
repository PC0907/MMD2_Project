"""Residual-stream activation capture for HuggingFace decoder-only models.

Registers forward hooks on the specified decoder layers and records the
hidden state (residual stream output of the layer) for the tokens of the
most recent forward pass. Pooling to a single vector per (layer, segment)
happens at read time so the same capture supports last-token and mean
pooling.

Works with Llama/Qwen/Mistral-style models where decoder layers live at
`model.model.layers[i]`. Adjust `_get_layer` for other architectures.
"""

from __future__ import annotations

import contextlib
from dataclasses import dataclass, field

import torch


def _get_layer(model: torch.nn.Module, idx: int) -> torch.nn.Module:
    """Return the idx-th decoder layer. Llama/Qwen/Mistral layout."""
    for attr_path in ("model.layers", "transformer.h", "gpt_neox.layers"):
        obj = model
        try:
            for part in attr_path.split("."):
                obj = getattr(obj, part)
            return obj[idx]
        except AttributeError:
            continue
    raise ValueError(
        f"Could not locate decoder layers on {type(model).__name__}; "
        "extend _get_layer for this architecture."
    )


@dataclass
class ActivationCapture:
    """Captures per-layer hidden states for the last forward pass.

    Usage:
        cap = ActivationCapture(model, layers=[16, 24])
        with cap.capture():
            out = model(**inputs)
        vec = cap.pooled(layer=24, token_slice=slice(-50, None), pooling="mean")
    """

    model: torch.nn.Module
    layers: list[int]
    store_device: str = "cpu"
    store_dtype: torch.dtype = torch.float32
    _acts: dict[int, torch.Tensor] = field(default_factory=dict, init=False)
    _handles: list = field(default_factory=list, init=False)

    def _make_hook(self, layer_idx: int):
        def hook(_module, _inputs, output):
            hidden = output[0] if isinstance(output, tuple) else output
            # hidden: (batch, seq, d_model). Keep batch 0; agent loop is bs=1.
            self._acts[layer_idx] = (
                hidden[0].detach().to(self.store_device, self.store_dtype)
            )
        return hook

    @contextlib.contextmanager
    def capture(self):
        self._acts.clear()
        try:
            for idx in self.layers:
                handle = _get_layer(self.model, idx).register_forward_hook(
                    self._make_hook(idx)
                )
                self._handles.append(handle)
            yield self
        finally:
            for h in self._handles:
                h.remove()
            self._handles.clear()

    def raw(self, layer: int) -> torch.Tensor:
        """(seq, d_model) hidden states for `layer` from the last pass."""
        if layer not in self._acts:
            raise KeyError(f"Layer {layer} not captured (have {sorted(self._acts)})")
        return self._acts[layer]

    def pooled(
        self,
        layer: int,
        token_slice: slice = slice(None),
        pooling: str = "mean",
    ) -> torch.Tensor:
        """Pool the captured hidden states of `layer` over `token_slice`.

        With KV caching, an incremental decode pass only exposes the new
        token's hidden state; run collection with full-sequence forward
        passes (or cache disabled for the monitored pass) so slices refer
        to the whole context.
        """
        h = self.raw(layer)[token_slice]
        if h.numel() == 0:
            raise ValueError("Empty token slice for pooling")
        if pooling == "mean":
            return h.mean(dim=0)
        if pooling == "last":
            return h[-1]
        raise ValueError(f"Unknown pooling: {pooling}")

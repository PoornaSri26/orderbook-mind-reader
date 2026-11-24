"""A small causal transformer. Torch is an optional dependency.

Design choices and why:

* **Causal mask only.** No bidirectional attention, ever — the label is in the
  future by construction.
* **2-4 layers, d_model 64-128, context 128-256 events.** Small enough to train
  on one GPU in minutes, which is what makes the ablation grid affordable.
* **No positional encoding beyond relative time.** The ``log_dt`` feature
  already carries the clock; adding learned positions on an irregular event
  stream invents structure that is not there.
* **Single sigmoid head.** The task is binary; a two-head softmax would let the
  model spend capacity on the "no move" class that the label deliberately drops.

The model is registered so ``interp.py`` can name a mid layer and the residual
stream without guessing indices.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

try:  # torch is optional: the baselines must run without it
    import torch
    from torch import nn

    _HAS_TORCH = True
except ImportError:  # pragma: no cover
    _HAS_TORCH = False
    nn = object  # type: ignore[assignment]


@dataclass(frozen=True)
class ModelConfig:
    n_features: int = 8
    d_model: int = 96
    n_layers: int = 3
    n_heads: int = 4
    context: int = 256
    dropout: float = 0.1
    mid_layer: int = 1  # the layer interp.py probes and patches


def available() -> bool:
    return _HAS_TORCH


if _HAS_TORCH:

    class CausalBlock(nn.Module):
        def __init__(self, cfg: ModelConfig):
            super().__init__()
            self.ln1 = nn.LayerNorm(cfg.d_model)
            self.attn = nn.MultiheadAttention(
                cfg.d_model, cfg.n_heads, dropout=cfg.dropout, batch_first=True
            )
            self.ln2 = nn.LayerNorm(cfg.d_model)
            self.mlp = nn.Sequential(
                nn.Linear(cfg.d_model, 4 * cfg.d_model),
                nn.GELU(),
                nn.Dropout(cfg.dropout),
                nn.Linear(4 * cfg.d_model, cfg.d_model),
            )

        def forward(self, x, mask):
            h = self.ln1(x)
            a, _ = self.attn(h, h, h, attn_mask=mask, need_weights=False)
            x = x + a
            return x + self.mlp(self.ln2(x))

    class BookTransformer(nn.Module):
        def __init__(self, cfg: ModelConfig):
            super().__init__()
            self.cfg = cfg
            self.proj = nn.Linear(cfg.n_features, cfg.d_model)
            self.blocks = nn.ModuleList(CausalBlock(cfg) for _ in range(cfg.n_layers))
            self.ln_f = nn.LayerNorm(cfg.d_model)
            self.head = nn.Linear(cfg.d_model, 1)
            m = torch.triu(torch.full((cfg.context, cfg.context), float("-inf")), diagonal=1)
            self.register_buffer("mask", m)

        def forward(self, x, return_residuals: bool = False):
            """x: (B, T, F) -> logits (B,). Reads the last position only."""
            h = self.proj(x)
            residuals = []
            for blk in self.blocks:
                h = blk(h, self.mask[: h.size(1), : h.size(1)])
                residuals.append(h)
            h = self.ln_f(h)
            logit = self.head(h[:, -1, :]).squeeze(-1)
            return (logit, residuals) if return_residuals else logit

        @torch.no_grad()
        def residuals_at(self, x, layer: int):
            """Residual stream at ``layer``'s output, for probes and patching."""
            _, res = self.forward(x, return_residuals=True)
            return res[layer]

        @torch.no_grad()
        def forward_with_injection(self, x, inject: dict[tuple[int, int], "torch.Tensor"]):
            """Forward pass with activations replaced at (layer, token) positions.

            This is what makes activation patching a causal test rather than a
            correlation: the value is written into the residual stream and the
            rest of the network runs normally on top of it.
            """
            h = self.proj(x)
            for i, blk in enumerate(self.blocks):
                h = blk(h, self.mask[: h.size(1), : h.size(1)])
                for (layer, token), value in inject.items():
                    if layer == i:
                        h[:, token, :] = value
            h = self.ln_f(h)
            return self.head(h[:, -1, :]).squeeze(-1)


def windows(x: np.ndarray, y: np.ndarray, context: int, stride: int = 1):  # noqa: D401
    """(N, T, F) windows ending at each labelled event, keeping the last value."""
    n = len(x) - context + 1
    if n <= 0:
        raise ValueError(f"need at least {context} events, have {len(x)}")
    idx = np.arange(context - 1, len(x), stride)
    idx = idx[idx < len(x)]
    # labels are aligned to the window's last event
    keep = ~np.isnan(y[idx])
    idx = idx[keep]
    out_x = np.stack([x[i - context + 1 : i + 1] for i in idx]).astype(np.float32)
    out_y = y[idx].astype(np.float32)
    return out_x, out_y, idx

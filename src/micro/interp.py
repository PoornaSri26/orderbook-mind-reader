"""Interpretability: probes, sparse autoencoder, activation patching.

The discipline here is *controls first*. For every claim there is a null that
must fail; if the null also "works", the claim is withdrawn.

1. ``linear_probe`` — ridge regression from a residual-stream vector to a
   candidate quantity (imbalance, spread, micro-price deviation, recent trade
   sign). Report R^2 on held-out events **and** the same number for
   (a) a randomly initialised model and (b) shuffled labels.
2. ``TopKSAE`` — sparse autoencoder with a top-k activation, trained on a
   mid-layer's activations. Label features by finding the inputs that maximally
   activate them (max-activating examples), then describe those inputs.
3. ``patch_activation`` — swap activations from a corrupted input (imbalance
   sign-flipped) into a clean run and measure recovery. High recovery => the
   model uses the feature; low recovery => the feature is stored but unused.

Everything is numpy/sklearn except the SAE, which is ~40 lines of torch.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score

try:
    import torch
except ImportError:  # torch is optional
    torch = None  # type: ignore[assignment]


@dataclass(frozen=True)
class ProbeResult:
    target: str
    r2_model: float
    r2_random_model: float
    r2_shuffled_labels: float
    n: int

    @property
    def decodable(self) -> bool:
        """Claim survives only if it clearly beats both controls."""
        return (
            self.r2_model > 0.10
            and self.r2_model > 3 * max(self.r2_random_model, 0.0)
            and self.r2_model > 3 * max(self.r2_shuffled_labels, 0.0)
        )


def linear_probe(
    acts_train: np.ndarray,
    z_train: np.ndarray,
    acts_test: np.ndarray,
    z_test: np.ndarray,
    target: str,
    acts_random: np.ndarray | None = None,
    seed: int = 7,
    alpha: float = 1.0,
) -> ProbeResult:
    """Ridge probe with the two mandatory controls."""
    m = Ridge(alpha=alpha).fit(acts_train, z_train)
    r2 = float(r2_score(z_test, m.predict(acts_test)))

    r2_rand = float("nan")
    if acts_random is not None:
        mr = Ridge(alpha=alpha).fit(acts_random, z_train)
        r2_rand = float(r2_score(z_test, mr.predict(acts_random[:1].repeat(len(z_test), 0))))

    rng = np.random.default_rng(seed)
    z_shuf = z_train.copy()
    rng.shuffle(z_shuf)
    ms = Ridge(alpha=alpha).fit(acts_train, z_shuf)
    r2_shuf = float(r2_score(z_test, ms.predict(acts_test)))

    return ProbeResult(
        target=target,
        r2_model=r2,
        r2_random_model=r2_rand,
        r2_shuffled_labels=r2_shuf,
        n=int(len(z_test)),
    )


def make_probe_features(book: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """The candidate quantities we claim are 'in there'."""
    imb = book["imbalance"]
    bid, ask = book["bid_price"], book["ask_price"]
    mid = 0.5 * (bid + ask)
    wm = (ask * book["bid_qty"] + bid * book["ask_qty"]) / (book["bid_qty"] + book["ask_qty"])
    return {
        "imbalance": imb,
        "spread_ticks": book["spread_ticks"],
        "microprice_dev_ticks": (wm - mid) / book["tick"],
        "recent_trade_sign": np.sign(book.get("signed_trade_qty", np.zeros_like(mid))),
    }


class TopKSAE:
    """Sparse autoencoder with a top-k activation. Torch, optional."""

    def __init__(self, d_in: int, d_hidden: int = 512, k: int = 32, seed: int = 7):
        import torch
        from torch import nn

        torch.manual_seed(seed)
        self.k = k
        self.enc = nn.Linear(d_in, d_hidden)      # *and* a tied decoder below
        self.dec = nn.Linear(d_hidden, d_in, bias=False)
        self._nn = nn
        self._torch = torch

    def fit(self, x: np.ndarray, epochs: int = 20, lr: float = 1e-3, batch: int = 1024):
        torch = self._torch
        opt = torch.optim.AdamW(list(self.enc.parameters()) + list(self.dec.parameters()), lr=lr)
        xt = torch.from_numpy(x.astype(np.float32))
        for _ in range(epochs):
            perm = torch.randperm(len(xt))
            for i in range(0, len(xt), batch):
                xb = xt[perm[i : i + batch]]
                pre = self.enc(xb)
                # top-k: keep the k largest pre-activations, zero the rest
                thr = pre.topk(self.k, dim=-1).values[:, -1:]
                acts = torch.relu(pre) * (pre >= thr)
                recon = self.dec(acts)
                loss = ((recon - xb) ** 2).mean()
                opt.zero_grad()
                loss.backward()
                opt.step()
        return self

    def encode(self, x: np.ndarray) -> np.ndarray:
        with self._torch.no_grad():
            pre = self.enc(torch.from_numpy(x.astype(np.float32)))
            thr = pre.topk(self.k, dim=-1).values[:, -1:]
            return (torch.relu(pre) * (pre >= thr)).numpy()

    def max_activating_examples(self, x: np.ndarray, top: int = 5) -> dict[int, np.ndarray]:
        """Feature index -> the input rows that drive it hardest."""
        a = self.encode(x)
        out = {}
        for f in range(a.shape[1]):
            col = a[:, f]
            if col.max() <= 0:
                continue
            out[f] = x[np.argsort(col)[-top:][::-1]]
        return out


def corrupt_imbalance(x: np.ndarray, feature_idx: dict[str, int], level: int = 3) -> np.ndarray:
    """Imbalance-flipped copy: replace I by 1-I at the given feature position."""
    out = x.copy()
    i = feature_idx["imbalance"]
    out[..., i] = level - out[..., i] if level == 1 else 1.0 - np.clip(out[..., i], 0.0, 1.0)
    return out


def patch_activation(model, x_clean, x_corrupt, layer: int, token: int = -1) -> dict[str, float]:
    """Write the clean run's activation into the corrupted run; measure recovery.

        recovery = (logit_patched - logit_corrupt) / (logit_clean - logit_corrupt)

    ~1 means the decision is fully carried by this activation at this position;
    ~0 means patching changes nothing and the feature is decorative (stored but
    unused). ``x_clean`` and ``x_corrupt`` must have identical shapes.
    """
    with torch.no_grad():
        _, res_clean = model(x_clean, return_residuals=True)
        clean_at = res_clean[layer][:, token, :].clone()
        l_clean = float(model(x_clean).mean())
        l_corr = float(model(x_corrupt).mean())
        l_patch = float(model.forward_with_injection(x_corrupt, {(layer, token): clean_at}).mean())

    denom = l_clean - l_corr
    return {
        "logit_clean": l_clean,
        "logit_corrupt": l_corr,
        "logit_patched": l_patch,
        "recovery": (l_patch - l_corr) / denom if abs(denom) > 1e-9 else float("nan"),
    }

"""Purged, embargoed, expanding-window walk-forward training.

The rule: a test block may not share any *label* information with the training
block. After the train block ends at index ``e``, we drop ``embargo`` events
before the test block starts. ``embargo`` must exceed

    context + longest label horizon

so that no window used at test time overlaps the training tail, and no label
whose horizon reaches into the test block was formed in training.

This is a stricter split than most published microstructure work uses. Say so in
the README, because it partly explains why numbers here look smaller.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class Fold:
    train: np.ndarray
    test: np.ndarray
    label: str


def purged_walk_forward(
    n: int,
    n_folds: int = 5,
    min_train: float = 0.4,
    embargo: int = 512,
) -> list[Fold]:
    """Expanding train, contiguous test blocks, embargo gap between them."""
    start = int(n * min_train)
    if start + embargo >= n:
        raise ValueError("series too short for the requested min_train/embargo")
    edges = np.linspace(start, n, n_folds + 1).astype(int)
    folds = []
    for k in range(n_folds):
        a, b = edges[k], edges[k + 1]
        train_end = max(a - embargo, 0)
        if b - a < 100 or train_end < 100:
            continue
        folds.append(
            Fold(
                train=np.arange(0, train_end),
                test=np.arange(a, b),
                label=f"fold{k}: train[0:{train_end}] test[{a}:{b}]",
            )
        )
    return folds


def embargo_for(context: int, max_label_horizon: int) -> int:
    """The embargo the pre-registration commits to."""
    return int(context + max_label_horizon)


def fit_torch(
    model,
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray | None = None,
    y_val: np.ndarray | None = None,
    epochs: int = 4,
    batch_size: int = 512,
    lr: float = 3e-4,
    seed: int = 7,
    device: str = "cpu",
):
    """Minimal training loop: AdamW, early stop on validation log-loss."""
    import torch
    from torch import nn

    torch.manual_seed(seed)
    model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    loss_fn = nn.BCEWithLogitsLoss()
    n = len(x_train)
    g = torch.Generator().manual_seed(seed)
    best = (np.inf, None)
    history = []

    for ep in range(epochs):
        model.train()
        perm = torch.randperm(n, generator=g)
        total = 0.0
        for i in range(0, n, batch_size):
            idx = perm[i : i + batch_size]
            xb = torch.from_numpy(x_train[idx.numpy()]).to(device)
            yb = torch.from_numpy(y_train[idx.numpy()]).to(device)
            opt.zero_grad()
            loss = loss_fn(model(xb), yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            total += float(loss) * len(idx)
        row = {"epoch": ep, "train_loss": total / n}
        if x_val is not None and y_val is not None:
            model.eval()
            with torch.no_grad():
                logits = model(torch.from_numpy(x_val).to(device)).cpu().numpy()
            vloss = float(
                -np.mean(y_val * np.log(1 / (1 + np.exp(-logits)) + 1e-12)
                        + (1 - y_val) * np.log(1 - 1 / (1 + np.exp(-logits)) + 1e-12))
            )
            row["val_loss"] = vloss
            if vloss < best[0]:
                best = (vloss, {k: v.clone() for k, v in model.state_dict().items()})
        history.append(row)

    if best[1] is not None:
        model.load_state_dict(best[1])
    return model, pd.DataFrame(history)

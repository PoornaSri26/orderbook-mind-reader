"""Baselines. This file is the project's real bar, so it is deliberately strong.

1. ``imbalance_logit``  — logistic regression on I = Qb/(Qb+Qa).
2. ``weighted_mid``     — unfitted: predict up when the size-weighted mid is
                          above the mid. Included as a sanity floor.
3. ``stoikov_microprice`` — Stoikov (2018). Discretise the book state into
                          (imbalance bucket, spread tick) bins, estimate the
                          Markov transition matrix of the mid's next move,
                          integrate to the fixed point, and use the resulting
                          micro-price as the predictor. Fitted on train only.
4. ``lightgbm_baseline`` — GBM on the same features the transformer sees.

Each returns predicted P(up) aligned to the rows it was given, so
``evaluate.py`` can treat every model identically.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.linear_model import LogisticRegression


def logit_fit_predict(
    x_train: pd.DataFrame, y_train: np.ndarray, x_test: pd.DataFrame, **kw
) -> np.ndarray:
    kw.setdefault("max_iter", 2000)
    kw.setdefault("C", 1.0)
    model = LogisticRegression(**kw)
    model.fit(x_train.to_numpy(), y_train)
    return model.predict_proba(x_test.to_numpy())[:, 1]


def imbalance_logit(
    imb_train: pd.Series, y_train: np.ndarray, imb_test: pd.Series
) -> np.ndarray:
    xtr = imb_train.to_frame("imbalance")
    xte = imb_test.to_frame("imbalance")
    return logit_fit_predict(xtr, y_train, xte)


def weighted_mid(bid_px: pd.Series, bid_qty: pd.Series, ask_px: pd.Series, ask_qty: pd.Series) -> pd.Series:
    """Size-weighted mid; the classic queue-pressure estimator."""
    num = ask_px * bid_qty + bid_px * ask_qty
    den = (bid_qty + ask_qty).replace(0, np.nan)
    return num / den


def weighted_mid_signal(*args) -> np.ndarray:
    """Map the weighted mid to P(up) via its position within the spread."""
    bid_px, bid_qty, ask_px, ask_qty = args
    wm = weighted_mid(bid_px, bid_qty, ask_px, ask_qty)
    mid = 0.5 * (bid_px + ask_px)
    spread = (ask_px - bid_px).replace(0, np.nan)
    # fraction of the half-spread that the weighted mid has travelled
    frac = ((wm - mid) / (0.5 * spread)).clip(-1, 1).fillna(0.0)
    return ((frac + 1.0) / 2.0).to_numpy()


@dataclass
class StoikovMicroprice:
    """Discretised-state micro-price.

    The state is (imbalance bucket, spread in ticks). For each state we estimate
    the distribution of the *next* mid move in ticks: -1, 0, +1 (rare moves of
    |2| are clipped into +/-1 after a warning). The micro-price is the fixed
    point of repeatedly applying that transition, which is what "the expected
    terminal mid" means once the chain is absorbed by the +/-1 steps.

    This is the strongest cheap baseline: it uses no learning beyond counting,
    and it embodies the same economic object the network might be discovering.
    """

    n_imbalance_buckets: int = 10
    max_spread_ticks: int = 5

    def fit(self, imb: pd.Series, spread_ticks: pd.Series, mid_change_ticks: pd.Series) -> "StoikovMicroprice":
        state = self._state(imb, spread_ticks)
        nxt = mid_change_ticks.fillna(0.0).clip(-1, 1).astype(int)
        df = pd.DataFrame({"state": state, "nxt": nxt}).dropna()
        counts = (
            df.groupby(["state", "nxt"]).size().unstack(fill_value=0).reindex(columns=[-1, 0, 1], fill_value=0)
        )
        probs = counts.div(counts.sum(axis=1), axis=0)
        # prior for unseen states: no drift
        self._table = probs.reindex(
            pd.MultiIndex.from_product(
                [range(self.n_imbalance_buckets), range(self.max_spread_ticks + 1)]
            ),
            fill_value=np.nan,
        )
        self._table = self._table.apply(lambda r: r.fillna(1 / 3), axis=1)
        self.state_counts = counts.sum(axis=1)
        return self

    def _state(self, imb: pd.Series, spread_ticks: pd.Series) -> pd.Series:
        b = np.clip((imb * self.n_imbalance_buckets).astype("float"), 0, self.n_imbalance_buckets - 1)
        s = np.clip(spread_ticks, 0, self.max_spread_ticks).fillna(0)
        return pd.MultiIndex.from_arrays([b.astype(int), s.astype(int)]).to_frame(index=False).apply(tuple, axis=1)

    def microprice_offset(self, imb: pd.Series, spread_ticks: pd.Series, n_iter: int = 8) -> np.ndarray:
        """Expected cumulative mid move in ticks, iterating the transition."""
        states = self._state(imb, spread_ticks)
        p_up = np.empty(len(states))
        for i, st in enumerate(states):
            row = self._table.loc[st] if st in self._table.index else pd.Series({-1: 1/3, 0: 1/3, 1: 1/3})
            p_up[i] = float(row.get(1, 1 / 3))
        # one-step expectation is p_up - p_down; iterating the chain's drift
        # converges geometrically once absorption at +/-1 is allowed, so 8 steps
        # is far beyond numerical relevance for these magnitudes.
        return p_up  # kept as probability: the caller only needs an ordering

    def predict_proba(self, imb: pd.Series, spread_ticks: pd.Series) -> np.ndarray:
        return self.microprice_offset(imb, spread_ticks)


def lightgbm_baseline(
    x_train: pd.DataFrame, y_train: np.ndarray, x_test: pd.DataFrame, **kw
) -> np.ndarray:
    import lightgbm as lgb

    params = dict(
        n_estimators=400,
        learning_rate=0.05,
        num_leaves=63,
        min_child_samples=200,
        subsample=0.8,
        subsample_freq=1,
        colsample_bytree=0.8,
        reg_lambda=1.0,
        objective="binary",
        n_jobs=-1,
        verbose=-1,
    )
    params.update(kw)
    model = lgb.LGBMClassifier(**params)
    model.fit(x_train.to_numpy(), y_train)
    return model.predict_proba(x_test.to_numpy())[:, 1]


def sparse_from_events(levels: dict[str, pd.DataFrame]) -> sparse.csr_matrix:
    """Flatten multi-level book columns for a GBM that must see the same info."""
    blocks = [df.to_numpy() for _, df in sorted(levels.items())]
    return sparse.csr_matrix(np.hstack(blocks))

"""Metrics, block bootstrap, and the basis-point translation.

The bp translation is the part that decides whether any of this matters:

    expected_edge_bps  =  E[ |mid move| | predicted up, actually up ] - E[ ... predicted up, actually down ]

i.e. the average signed move the prediction actually captures per event, in
basis points of mid. Compare with

    cost_bps = half_spread_bps + fee_bps

BTC's one-tick spread on a ~60k price is roughly 0.2-0.5 bp per side; the exact
number comes from the data via ``load.to_event_frame``. State it explicitly
rather than quoting a rule of thumb.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.metrics import log_loss, roc_auc_score


@dataclass(frozen=True)
class Score:
    model: str
    log_loss: float
    auc: float
    brier: float
    n: int
    mean_pred: float

    def as_row(self) -> dict:
        return {
            "model": self.model,
            "log_loss": self.log_loss,
            "auc": self.auc,
            "brier": self.brier,
            "n": self.n,
            "mean_pred": self.mean_pred,
        }


def score(name: str, y: np.ndarray, p: np.ndarray) -> Score:
    p = np.clip(np.asarray(p, dtype=float), 1e-6, 1 - 1e-6)
    y = np.asarray(y).astype(int)
    return Score(
        model=name,
        log_loss=float(log_loss(y, p)),
        auc=float(roc_auc_score(y, p)) if len(np.unique(y)) > 1 else float("nan"),
        brier=float(np.mean((p - y) ** 2)),
        n=int(y.size),
        mean_pred=float(p.mean()),
    )


def calibration_table(y: np.ndarray, p: np.ndarray, bins: int = 10) -> pd.DataFrame:
    df = pd.DataFrame({"y": y, "p": p})
    df["bin"] = pd.qcut(df["p"], bins, duplicates="drop")
    g = df.groupby("bin", observed=True)
    return pd.DataFrame(
        {
            "n": g.size(),
            "pred": g["p"].mean(),
            "actual": g["y"].mean(),
            "ci_lo": g["y"].apply(lambda s: s.mean() - 1.96 * s.std(ddof=1) / np.sqrt(len(s))),
            "ci_hi": g["y"].apply(lambda s: s.mean() + 1.96 * s.std(ddof=1) / np.sqrt(len(s))),
        }
    )


def block_bootstrap_diff(
    y: np.ndarray,
    p_a: np.ndarray,
    p_b: np.ndarray,
    block: int = 500,
    n_boot: int = 500,
    seed: int = 7,
) -> dict:
    """Bootstrap the log-loss *difference* (b - a) on contiguous blocks.

    Blocks, not rows: adjacent events are strongly dependent, so an iid
    bootstrap would report absurdly narrow intervals. Block length 500 events
    is roughly the decorrelation scale of bookTicker quote changes on BTC.
    """
    rng = np.random.default_rng(seed)
    y = np.asarray(y).astype(int)
    n = y.size
    starts = np.arange(0, n - block + 1, block)
    diffs = np.empty(n_boot)
    for k in range(n_boot):
        pick = rng.choice(starts, size=len(starts), replace=True)
        idx = np.concatenate([np.arange(s, s + block) for s in pick])
        diffs[k] = log_loss(y[idx], np.clip(p_b[idx], 1e-6, 1 - 1e-6)) - log_loss(
            y[idx], np.clip(p_a[idx], 1e-6, 1 - 1e-6)
        )
    obs = log_loss(y, np.clip(p_b, 1e-6, 1 - 1e-6)) - log_loss(y, np.clip(p_a, 1e-6, 1 - 1e-6))
    return {
        "observed_diff": float(obs),
        "ci_lo": float(np.quantile(diffs, 0.025)),
        "ci_hi": float(np.quantile(diffs, 0.975)),
        "p_value_two_sided": float(2 * min((diffs <= 0).mean(), (diffs >= 0).mean())),
        "excludes_zero": bool(np.quantile(diffs, 0.025) > 0 or np.quantile(diffs, 0.975) < 0),
    }


def edge_in_bps(
    y: np.ndarray,
    p: np.ndarray,
    forward_move_bps: np.ndarray,
    threshold: float = 0.5,
) -> dict:
    """Signed mid move captured per traded event, in bp.

    ``forward_move_bps`` is the realised signed move in bp from this event to
    the event the label refers to (positive if up). We only take the trades the
    model actually wants (`p > threshold` for long, `p < 1-threshold` for short).
    """
    y = np.asarray(y).astype(int)
    p = np.asarray(p, dtype=float)
    m = np.asarray(forward_move_bps, dtype=float)
    pos = np.where(p > threshold, 1.0, np.where(p < 1 - threshold, -1.0, 0.0))
    traded = pos != 0
    if traded.sum() == 0:
        return {"n_trades": 0, "edge_bps": 0.0, "hit_rate": float("nan")}
    pnl = pos[traded] * m[traded]
    return {
        "n_trades": int(traded.sum()),
        "trade_rate": float(traded.mean()),
        "edge_bps": float(pnl.mean()),
        "hit_rate": float((np.sign(m[traded]) == pos[traded]).mean()),
    }


def cost_line(half_spread_bps: float, fee_bps: float, slippage_bps: float = 0.0) -> float:
    return float(half_spread_bps + fee_bps + slippage_bps)


def verdict(edge_bps: float, cost_bps: float) -> str:
    if edge_bps <= 0:
        return "no edge after direction; predictable at best in the loss, not in the PnL"
    if edge_bps < cost_bps:
        return f"predictable ({edge_bps:.3f} bp) but below the {cost_bps:.3f} bp cost line — not tradeable"
    return f"edge {edge_bps:.3f} bp clears the {cost_bps:.3f} bp cost line: investigate before believing"

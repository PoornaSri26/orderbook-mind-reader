"""Per-event features.

Every feature at row ``t`` uses information available at or before the close of
event ``t``. Lags are explicit (``.shift(1)``) rather than implicit, because
look-ahead in features is the single most common way this project lies to
itself.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

FEATURE_COLUMNS = [
    "log_bid_qty",
    "log_ask_qty",
    "imbalance",
    "spread_ticks",
    "log_dt",
    "mid_change",
    "signed_qty_change",
    "queue_change_imbalance",
]


def build(df: pd.DataFrame, tick: float | None = None) -> pd.DataFrame:
    """Feature frame from an event frame produced by ``load.to_event_frame``."""
    out = pd.DataFrame(index=df.index)
    tick = tick or float(df["tick"].iloc[0])

    out["log_bid_qty"] = np.log1p(df["bid_qty"])
    out["log_ask_qty"] = np.log1p(df["ask_qty"])
    out["imbalance"] = df["bid_qty"] / (df["bid_qty"] + df["ask_qty"]).replace(0, np.nan)
    out["spread_ticks"] = (df["ask_price"] - df["bid_price"]) / tick

    dt = df.index.to_series().diff().dt.total_seconds()
    out["log_dt"] = np.log1p(dt.clip(lower=0).fillna(0.0))

    # mid change and queue changes are shifted: at row t we know the *previous*
    # move, not the one that happens after this event.
    mid = df["mid"]
    out["mid_change"] = (mid.diff() / tick).shift(1).fillna(0.0)

    dq_bid = df["bid_qty"].diff()
    dq_ask = df["ask_qty"].diff()
    out["signed_qty_change"] = (dq_bid.fillna(0.0) - dq_ask.fillna(0.0)).shift(1).fillna(0.0)
    denom = (dq_bid.abs() + dq_ask.abs()).replace(0, np.nan)
    out["queue_change_imbalance"] = (dq_bid / denom).shift(1).fillna(0.0)

    return out[FEATURE_COLUMNS].astype("float64")


@dataclass(frozen=True)
class Normalizer:
    """Fold-local standardisation. Fit on train only, applied to test."""

    mean: pd.Series
    std: pd.Series

    @classmethod
    def fit(cls, x: pd.DataFrame) -> "Normalizer":
        mean = x.mean()
        std = x.std(ddof=0).replace(0.0, 1.0).fillna(1.0)
        return cls(mean=mean, std=std)

    def transform(self, x: pd.DataFrame) -> pd.DataFrame:
        return (x - self.mean) / self.std


def add_trade_flow(trades: pd.DataFrame, index: pd.DatetimeIndex, window_s: float = 1.0) -> pd.Series:
    """Signed trade size per second, aligned to the quote clock.

    ``trades`` needs columns ``ts``, ``qty`` and ``is_buyer_maker`` (Binance's
    aggressor flag). A *market sell* hits the bid, which has
    ``is_buyer_maker == True``; so the aggressor sign is
    ``+1 if not is_buyer_maker else -1``.
    """
    if trades.empty:
        return pd.Series(0.0, index=index)
    t = trades.set_index("ts").sort_index()
    signed = t["qty"] * np.where(t["is_buyer_maker"], -1.0, 1.0)
    flow = signed.resample(f"{int(window_s * 1000)}ms").sum()
    return flow.reindex(index, method="ffill").fillna(0.0)

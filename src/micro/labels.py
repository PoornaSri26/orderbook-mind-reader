"""Labels.

Primary label: the sign of the *next non-zero* mid change, in event time.

Why event time: with a one-tick spread the mid sits at a half-tick grid and many
updates leave it unchanged. A calendar-time label would be a mixture of "no
news" and "news", and the no-news mass would dominate the loss. Event time
instead asks the question a maker actually faces: given a quote change, which
way is the next move?

Consequence, stated in the README: the label set excludes zero-change events,
so the model never learns to predict "nothing happens". The drop rate is
returned by :func:`event_time_labels` and must be reported.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class LabelSet:
    y: pd.Series          # -1 / +1 (0 dropped)
    horizon: pd.Series    # bars until the move, for horizon-dependent costs
    dropped_fraction: float
    kind: str

    def binary(self) -> np.ndarray:
        """1 for up, 0 for down, after dropping zeros."""
        v = self.y.dropna()
        return (v > 0).astype(np.int8).to_numpy()


def event_time_labels(mid: pd.Series) -> LabelSet:
    """Sign of the next non-zero change in ``mid``."""
    m = mid.to_numpy(dtype=float)
    n = m.size
    sign = np.zeros(n)
    horizon = np.full(n, np.nan)

    # Walk backwards: nearest future index i where m[i] != m[t].
    nxt = np.full(n, np.nan)
    last_change_idx = np.nan
    for t in range(n - 1, -1, -1):
        nxt[t] = last_change_idx
        if t > 0 and m[t] != m[t - 1]:
            last_change_idx = t

    for t in range(n):
        j = nxt[t]
        if np.isnan(j):
            continue
        j = int(j)
        d = m[j] - m[t]
        if d == 0:
            continue
        sign[t] = np.sign(d)
        horizon[t] = j - t

    y = pd.Series(sign, index=mid.index).replace(0.0, np.nan)
    return LabelSet(
        y=y,
        horizon=pd.Series(horizon, index=mid.index),
        dropped_fraction=float(np.mean(pd.isna(y))),
        kind="next_nonzero_event",
    )


def fixed_horizon_labels(mid: pd.Series, horizon_events: int = 100) -> LabelSet:
    """Sign of the mid change over the next ``horizon_events`` updates."""
    fwd = mid.shift(-horizon_events)
    d = (fwd - mid).to_numpy()
    y = pd.Series(np.sign(d), index=mid.index).replace(0.0, np.nan)
    return LabelSet(
        y=y,
        horizon=pd.Series(float(horizon_events), index=mid.index),
        dropped_fraction=float(np.mean(pd.isna(y))),
        kind=f"fixed_{horizon_events}_events",
    )


def fixed_horizon_labels_time(mid: pd.Series, seconds: float = 1.0) -> LabelSet:
    """Sign of the mid change ``seconds`` ahead, on the event clock."""
    target = mid.index + pd.Timedelta(seconds=seconds)
    idx = mid.index.searchsorted(target, side="left")
    idx = np.clip(idx, 0, mid.size - 1)
    fwd = pd.Series(mid.to_numpy()[idx], index=mid.index)
    y = pd.Series(np.sign((fwd - mid).to_numpy()), index=mid.index).replace(0.0, np.nan)
    return LabelSet(
        y=y,
        horizon=pd.Series(float(seconds), index=mid.index),
        dropped_fraction=float(np.mean(pd.isna(y))),
        kind=f"fixed_{seconds}s",
    )


def class_balance(y: pd.Series) -> dict[str, float]:
    v = y.dropna()
    up = float((v > 0).mean())
    return {"n": int(v.size), "p_up": up, "p_down": 1 - up}

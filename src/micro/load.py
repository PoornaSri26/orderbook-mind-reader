"""Loading Binance USD-M ``bookTicker`` archives into a tidy frame.

Verified schema (futures/um bookTicker CSV, headerless):

    update_id, best_bid_price, best_bid_qty, best_ask_price, best_ask_qty,
    transaction_time, event_time

``transaction_time`` and ``event_time`` are epoch **milliseconds**. Futures
timestamps did not switch to microseconds (that change was spot-only), but the
loader normalises anyway so a spot file can be dropped in without surprises.

Coverage note: the archive on data.binance.vision ends 2024-03-30 (daily) /
2024-04 (monthly) as of Nov 2026. ``available_range()`` reads the bucket index
rather than hard-coding dates, so the code reports the truth even if the host
extends coverage later.
"""

from __future__ import annotations

import io
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data"
RAW = DATA / "raw"
PROCESSED = DATA / "processed"

BT_COLUMNS = [
    "update_id",
    "bid_price",
    "bid_qty",
    "ask_price",
    "ask_qty",
    "transaction_time",
    "event_time",
]

BASE = "https://data.binance.vision/data/futures/um"

DTYPES = {
    "update_id": "int64",
    "bid_price": "float64",
    "bid_qty": "float64",
    "ask_price": "float64",
    "ask_qty": "float64",
    "transaction_time": "int64",
    "event_time": "int64",
}


def _to_utc(ms: pd.Series) -> pd.Series:
    v = ms.to_numpy(dtype="int64")
    unit = np.where(v > 10**14, "us", "ms")
    out = np.empty(v.size, dtype="datetime64[ns]")
    # normalise per-file: a single file is one unit in practice
    if (unit == "us").all():
        out = pd.to_datetime(v, unit="us", utc=True)
    else:
        out = pd.to_datetime(v, unit="ms", utc=True)
    return pd.Series(out)


def read_bookticker_zip(path: Path) -> pd.DataFrame:
    with zipfile.ZipFile(path) as zf:
        name = zf.namelist()[0]
        with zf.open(name) as fh:
            df = pd.read_csv(
                io.TextIOWrapper(fh, encoding="utf-8"),
                header=None,
                names=BT_COLUMNS,
                dtype=DTYPES,
                on_bad_lines="warn",
            )
    df["ts"] = _to_utc(df["event_time"])
    df = df.set_index("ts").sort_index()
    df = df[~df.index.duplicated(keep="first")]

    bad = (df["bid_price"] <= 0) | (df["ask_price"] <= 0) | (df["bid_price"] >= df["ask_price"])
    if bad.any():
        # crossed/zero quotes are real in the feed (and are a finding in their own
        # right); count them, then drop, rather than silently keeping them.
        print(f"  dropped {int(bad.sum())} crossed/zero quotes ({bad.mean():.4%})")
    df = df.loc[~bad]
    return df[["bid_price", "bid_qty", "ask_price", "ask_qty", "update_id"]]


def available_range(symbol: str = "BTCUSDT", timeout: int = 60) -> tuple[str, str]:
    """Earliest and latest month available in the monthly bucket, from the index."""
    import re
    import urllib.request

    url = f"{BASE}/monthly/bookTicker/{symbol}/"
    with urllib.request.urlopen(url, timeout=timeout) as r:
        html = r.read().decode("utf-8", "replace")
    months = sorted(set(re.findall(rf"{symbol}-bookTicker-(\d{{4}}-\d{{2}})\.zip", html)))
    if not months:
        raise RuntimeError(f"could not parse months from {url}")
    return months[0], months[-1]


def to_event_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Add mid, spread and tick-relative quantities used by every downstream module.

    ``tick`` is *estimated* as the gcd-like minimum positive price increment seen
    in the sample, because the archive does not ship the exchange filter. State
    this estimate wherever spread-in-ticks is reported.
    """
    out = df.copy()
    out["mid"] = 0.5 * (out["bid_price"] + out["ask_price"])
    out["spread"] = out["ask_price"] - out["bid_price"]
    inc = out["mid"].diff().abs()
    inc = inc[inc > 0]
    tick = float(inc.min()) if not inc.empty else np.nan
    out["tick"] = tick
    out["spread_ticks"] = out["spread"] / tick if tick and np.isfinite(tick) else np.nan
    out["imbalance"] = out["bid_qty"] / (out["bid_qty"] + out["ask_qty"]).replace(0, np.nan)
    return out


def save_parquet(df: pd.DataFrame, name: str) -> Path:
    PROCESSED.mkdir(parents=True, exist_ok=True)
    path = PROCESSED / f"{name}.parquet"
    df.to_parquet(path)
    return path


def load_parquet(name: str) -> pd.DataFrame:
    return pd.read_parquet(PROCESSED / f"{name}.parquet")

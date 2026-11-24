"""Paths, seeds and the frozen experimental constants."""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data"
RAW = DATA / "raw"
PROCESSED = DATA / "processed"
OUT = ROOT / "outputs"
FIG = OUT / "figures"

SEED = 7
SYMBOL = "BTCUSDT"
CONTEXT = 256
LABEL_HORIZON_EVENTS = 100
EMBARGO = CONTEXT + LABEL_HORIZON_EVENTS
N_FOLDS = 5
MIN_TRAIN_FRAC = 0.4

# Cost line. The BTC perp fee is 2 bp taker with BNB discount, 4.5 bp without;
# use 4.5 to avoid an optimistic assumption. Slippage on a one-tick book is the
# half-spread itself, counted separately.
FEE_BPS = 4.5
SLIPPAGE_BPS = 0.0

LOG_LOSS_IMPROVEMENT_BAR = 0.005
AUC_FLOOR = 0.55


def ensure_dirs() -> None:
    for p in (DATA, RAW, PROCESSED, OUT, FIG):
        p.mkdir(parents=True, exist_ok=True)

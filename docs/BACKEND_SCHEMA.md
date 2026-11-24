# BACKEND_SCHEMA — Order Book Mind Reader

No database. The schema is the on-disk contract between stages.

## 1. `data/raw/BTCUSDT-bookTicker-YYYY-MM-DD.zip`

Headerless CSV inside the zip, futures `bookTicker` schema:

| # | Column | Type | Notes |
|---|---|---|---|
| 0 | update_id | int64 | exchange update sequence |
| 1 | best_bid_price | float64 | |
| 2 | best_bid_qty | float64 | base units |
| 3 | best_ask_price | float64 | |
| 4 | best_ask_qty | float64 | base units |
| 5 | transaction_time | int64 | epoch **ms** (futures did not move to µs) |
| 6 | event_time | int64 | epoch ms; used as the index |

**Constraints:** `bid_price < ask_price` and both `> 0`, else the row is dropped
and counted. Duplicate `event_time` → keep first.

## 2. `data/processed/bookticker_features.parquet`

Index: UTC `DatetimeIndex` (from `event_time`).

| Column | Type | Definition |
|---|---|---|
| `log_bid_qty` | float64 | `log1p(bid_qty)` |
| `log_ask_qty` | float64 | `log1p(ask_qty)` |
| `imbalance` | float64 | `Qb/(Qb+Qa)`, NaN if both zero |
| `spread_ticks` | float64 | `(ask-bid)/tick`, tick estimated from min positive mid increment |
| `log_dt` | float64 | `log1p(seconds since previous event)` |
| `mid_change` | float64 | previous mid change in ticks (`shift(1)`) |
| `signed_qty_change` | float64 | previous `ΔQb − ΔQa` (`shift(1)`) |
| `queue_change_imbalance` | float64 | previous `ΔQb/(|ΔQb|+|ΔQa|)` |
| `mid` | float64 | `(bid+ask)/2` — target construction only |
| `spread` | float64 | absolute spread |
| `spread_ticks` | float64 | duplicate of the feature, kept for cost maths |
| `imbalance` | float64 | duplicate of the feature |
| `label_event` | float64 | ±1 sign of next non-zero mid change, NaN if none |
| `label_fixed100` | float64 | ±1 sign of mid change 100 events ahead |

**Invariant (tested):** every column in `FEATURE_COLUMNS` is a function of data
at or before its own timestamp. The only negative shift allowed anywhere is
inside `labels.py`.

## 3. `outputs/baselines.csv`

| Column | Meaning |
|---|---|
| `model` | `imbalance_logit` \| `weighted_mid` \| `stoikov` \| `lightgbm` |
| `log_loss` | mean over folds |
| `auc` | mean over folds |
| `brier` | mean over folds |
| `n` | summed scored events |

## 4. `outputs/model_vs_baseline.csv`

Same columns, `model ∈ {transformer, lightgbm}`, scored on the **paired
intersection** of events (so the comparison is legitimate).

## 5. `outputs/probes.csv`

| Column | Meaning |
|---|---|
| `target` | `imbalance`, `spread_ticks`, `microprice_dev_ticks`, `recent_trade_sign` |
| `r2_model` | held-out R² of a ridge probe on the mid layer |
| `r2_random` | same probe on a randomly initialised model |
| `r2_shuffled` | same probe with shuffled labels |
| `decodable` | `r2_model > 0.10` and `> 3×` both controls |

## 6. `outputs/history_foldK.csv`

`epoch, train_loss, val_loss` — the early-stopping trace, kept so a suspiciously
good fold can be inspected.

## 7. `outputs/sae_examples.npy`

Array of feature indices that produced a non-zero top-k activation. The
max-activating input rows are printed, not stored; storing 512 × 5 × 256 floats
would be 2.6 MB of noise.

## 8. Access rules

| Path | Read | Write |
|---|---|---|
| `data/raw/` | any stage | downloader only |
| `data/processed/` | any stage | `build_dataset.py` only |
| `outputs/baselines.csv` | `run_model.py` (as a gate) | `run_baselines.py` only |
| `outputs/probes.csv` | report | `run_interp.py` only |

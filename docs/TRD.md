# TRD — Order Book Mind Reader

## 1. Project overview

| Item | Value |
|---|---|
| Package | `micro` |
| Language | Python 3.11+ |
| Core | numpy, pandas, scipy, scikit-learn, lightgbm, pyarrow, matplotlib |
| Optional | torch (CPU wheel is enough) |
| Storage | Parquet |
| Compute | CPU for baselines; one CPU/GPU for the transformer, minutes per fold |

## 2. Technical objectives

- **Strong baselines first.** The GBM must see the same information set as the
  transformer, flattened. No straw men.
- **Leak-free by construction.** Purge + embargo, fold-local normalisation.
- **Cheap to iterate.** Two weeks of data must train a fold in minutes.

## 3. Functional requirements

| ID | Requirement |
|---|---|
| F1 | Read bookTicker zips (headerless CSV, 7 columns) into an event frame |
| F2 | Estimate tick size from the data (archives do not ship the exchange filter) |
| F3 | Count and drop crossed/zero quotes, reporting the rate |
| F4 | Build next-non-zero-mid labels + two robustness labels |
| F5 | Build 8 per-event features with explicit lags |
| F6 | Fit and score 4 baselines per fold |
| F7 | Train a 3-layer causal transformer with early stopping on an inner split |
| F8 | Score with log-loss, AUC, Brier, calibration bins |
| F9 | Block-bootstrap the log-loss difference |
| F10 | Translate the prediction into bp and compare with half-spread + fee |
| F11 | Probe, SAE, patch — each with a control |

## 4. Non-functional requirements

| Requirement | Target |
|---|---|
| Dataset build (14 days) | ≤ 10 min, ≤ 4 GB RAM |
| One transformer fold | ≤ 5 min CPU at context 256 |
| Determinism | same seed → same fold scores (torch seeded, `num_workers=0`) |
| Memory ceiling | 8 GB for the default context and feature set |

## 5. Stack and responsibility

| Layer | Technology | Responsibility |
|---|---|---|
| Ingest | stdlib zipfile + pandas | decode archives |
| Frames | pandas | event frame, joins, resampling |
| Models | scikit-learn, lightgbm | baselines |
| Models | torch | transformer, SAE |
| Numerics | scipy | Poisson/statistical helpers, distances |
| Storage | pyarrow | features parquet |
| Charts | matplotlib | baseline table figure, imbalance-response curve |
| Tests | pytest | label and leak invariants |

## 6. Integrations (verified Nov 2026)

| Service | Endpoint | Auth | Notes |
|---|---|---|---|
| Binance public data | `data.binance.vision/data/futures/um/{daily,monthly}/bookTicker/BTCUSDT/` | none | coverage ends **2024-03-30** (daily) / 2024-04 (monthly, truncated) |
| Hyperliquid (fallback feed) | `wss://api.hyperliquid.xyz/ws`, `l2Book` + `trades` | none | see project 3's recorder |

No API keys. No private endpoints.

## 7. Constraints and assumptions

- **The archive is frozen.** Out-of-sample means "later 2023-2024", not "recent".
  Any claim of currency is false and must not be made.
- Spread is one tick for most of the sample, so mid moves are discrete and the
  binary label is clean — and the attainable edge is tiny.
- Tick size is *estimated*; if Binance changed it mid-sample, `spread_ticks` is
  wrong for part of the data. The estimator uses the minimum positive mid
  increment, which is robust but coarse; a change-point check is listed as
  follow-up.
- Zero-change events are dropped for the primary label. The model therefore does
  not learn "nothing happens", and the drop rate must be quoted.
- Fees: 4.5 bp taker assumed. If you trade maker, the sign of the edge changes
  and the whole bp translation must be redone — say so rather than reusing it.

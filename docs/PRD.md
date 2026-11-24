# PRD — Order Book Mind Reader

## 1. Overview

An out-of-sample test of whether a small causal transformer beats strong
microstructure baselines at predicting the next mid move in BTCUSDT, plus a
mechanical explanation of what the network actually uses.

## 2. Problem statement

Two claims circulate about order-book ML:

- "Transformers read the book and predict the next tick." Usually measured
  against a weak baseline, in-sample, after a random split.
- "There is no edge." Usually asserted without ever building the strong
  baseline that would settle it.

Neither is an answer. The question only has content when the baseline is strong,
the split is purged, and the gain is translated into basis points against the
spread.

## 3. Goals (measurable)

| # | Goal | Measure |
|---|---|---|
| G1 | A baseline strong enough to be a real bar | 4 baselines incl. LightGBM on multi-level features and a Stoikov micro-price |
| G2 | No leakage | purged expanding folds with embargo = context + label horizon |
| G3 | An honest comparison | block-bootstrap CI on the log-loss difference |
| G4 | A tradeability answer | edge in bp vs half-spread + fee, reported either way |
| G5 | Mechanistic evidence | probes with two controls, SAE features, patching recovery |

## 4. Target users

- The author, as a portfolio artefact for quant-research interviews.
- A reader who wants a template for *honest* microstructure ML.

## 5. Features and priorities

| Priority | Feature |
|---|---|
| Must | bookTicker ingest with integrity checks and crossed-quote accounting |
| Must | Event-time label + fixed-horizon robustness labels |
| Must | Four baselines, each fitted on train only |
| Must | Purged, embargoed walk-forward |
| Must | Log-loss/AUC/calibration + block bootstrap + bp translation |
| Must | Linear probes with random-init and shuffled-label controls |
| Should | Top-k SAE on a mid layer, labelled by max-activating examples |
| Should | Activation patching (causal, via injection into the residual stream) |
| Could | Second symbol with a wider spread than one tick |
| Could | Cross-venue comparison (Binance vs Hyperliquid) |
| Won't | Any live trading, order placement or execution simulation |
| Won't | Claims about the article's 82% imbalance figure |
| Won't | A dashboard or web UI |

## 6. User stories

**US-1** — *As the author, I want a baseline I cannot hand-wave away.*
Acceptance: `outputs/baselines.csv` exists before any model number is reported;
`run_model.py` refuses to run without it.

**US-2** — *As a reviewer, I want to know if the edge is tradeable.*
Acceptance: `evaluate.verdict` prints one of three sentences, and the README
quotes it verbatim including the bp numbers.

**US-3** — *As a sceptic, I want the interpretability claims to have controls.*
Acceptance: `outputs/probes.csv` has `r2_random` and `r2_shuffled` columns, and
the script withdraws any claim where the control also scores.

**US-4** — *As a reader, I want to know the data's limits.*
Acceptance: the README states in its first screen that the archive ends
2024-03-30 and what that means for the holdout.

## 7. Success metrics

- Bar met: log-loss improvement ≥ 0.005 nats, CI excludes zero, AUC ≥ 0.55.
- Bar missed: reported as the result, with the bp number showing *why* (usually:
  the gain is smaller than half the spread).
- Runtime: dataset build ≤ 10 min for two weeks of data; one fold ≤ 5 min on CPU.

## 8. Out of scope

Execution, queue position, fee tiers, multi-venue routing, L3 (order-level)
data, and any statement about profitability of a real strategy.

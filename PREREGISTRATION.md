# Pre-registration — Order Book Mind Reader

**Registered at (UTC):** `2026-11-24T16:42:49Z`
**Frozen bar hash:** see `PREREGISTRATION.sha256` (written by `make stamp`)

## The question

Does a small causal transformer beat strong microstructure baselines at
predicting the next mid-price move, out of sample — and what does it actually
use to do it?

## Data (verified against data.binance.vision, Nov 2026)

| Item | Value |
|---|---|
| Symbol / venue | BTCUSDT, Binance USD-M perpetual |
| Feed | `bookTicker` (best bid/ask size and price) |
| Path | `data/futures/um/monthly/bookTicker/BTCUSDT/` |
| **Coverage** | monthly archives appear to stop at **2024-04**, daily at **2024-03-30**. Verified on the bucket index. There is no live alternative on that host, so the usable window is bounded. |
| Size | 3–8 GB per month compressed — plan storage before starting |
| Fallback | record Hyperliquid `l2Book` + `trades` yourself (see project 3's recorder) |

Because the Binance archive is frozen in early 2024, **the holdout is a time
block inside 2023–2024, not "the future"**. This is stated up front rather than
discovered later, because it changes what the result means: a model trained on
2023 and tested on early 2024 has not survived a regime change.

## Labels

- **Primary:** sign of the next non-zero mid change (event time). Zero-change
  events are dropped for this label and the drop rate is reported.
- **Robustness:** fixed horizon of 100 events, and of 1 second.

## Baselines (the real bar)

1. Logistic regression on imbalance `I = Qb/(Qb+Qa)`.
2. Weighted mid as a direct (unfitted) predictor.
3. Stoikov micro-price: 10 imbalance buckets x spread, Markov transition matrix
   on the discretised state, iterated to its fixed point.
4. LightGBM on hand-built features (multi-level imbalance, queue changes,
   trade-flow imbalance, recent mid returns).

The transformer is only interesting if it beats **baseline 4**.

## Pass / fail bar

The transformer "wins" if and only if:

1. **Log-loss improves on LightGBM by ≥ 0.005 nats** on the out-of-sample
   block, and the 95% block-bootstrap CI on that difference excludes zero, and
2. **AUC ≥ 0.55** out of sample (a floor, not a triumph).

A *separate, non-negotiable* reporting requirement: the bp translation. The
predicted edge is compared against **half the spread plus fees**. If it does not
clear that, the project reports "predictable but not tradeable" as its finding
and does not soften it.

## Interpretability claims and their controls

Each of these is a claim with a null attached:

| Claim | Control that must fail |
|---|---|
| Imbalance is linearly decodable from layer L | probe on a randomly initialised model; probe on shuffled labels |
| Some SAE features correspond to interpretable book states | feature activation on shuffled inputs |
| The model *uses* imbalance | activation patching from a corrupted (imbalance-flipped) book recovers < 50% of the prediction shift |

If the control also "works", the claim is withdrawn, not reworded.

## Splits

Expanding-window walk-forward, purged, with an embargo **longer than the
context window** (256 events + the label horizon). No random row splits, ever.
Normalisation statistics come from the training fold only.

## Not claimed

That the article's imbalance figure (~82%) holds. It is a hypothesis with an
unknown sample behind it; the whole point of baselines 1-4 is to find out.

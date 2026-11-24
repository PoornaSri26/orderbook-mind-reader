# Order Book Mind Reader

**Question.** Does a small causal transformer beat strong microstructure
baselines at predicting the next mid move, out of sample — and what does it
actually use?

**Bar.** Beat LightGBM by ≥ 0.005 nats log-loss with a block-bootstrap CI
excluding zero, AUC ≥ 0.55, and report the bp edge against half-spread + fees
whether or not it clears. Frozen in `PREREGISTRATION.md`.

**Result.** _Not yet run._

---

## Read this before you start

Two facts that change the plan:

1. **The Binance `bookTicker` archives stop at 2024-03-30** (verified on the
   bucket index). Monthly files exist through 2024-04. So this is a
   *historical* dataset with a fixed end date, not a live feed. Your "out of
   sample" is a later slice of 2023–24, and you must say so in the README
   rather than implying it is recent.
2. **The archives are 3–8 GB per month compressed.** Budget disk before you
   download a year. `scripts/download_bookticker.py --days 14` is the sane
   start.

## Documentation

Built and maintained per `AGENTS.md`. The 12 documents live in `docs/`:
`PRD` (scope, bar) · `TRD` (stack, verified endpoints) · `APP_FLOW` (stage order)
· `ARCHITECTURE` (module boundaries) · `BACKEND_SCHEMA` (parquet/CSV shapes) ·
`DESIGN_SYSTEM` (figures) · `CODE_STYLE` · `IMPLEMENTATION_PLAN` (3 weeks) ·
`SECURITY` (leakage controls) · `TESTING` (cases + statuses).
The frozen bar is `PREREGISTRATION.md`; its hash is `PREREGISTRATION.sha256`.

## Layout

```
docs/                   the 12 project documents
AGENTS.md               agent operating manual
PREREGISTRATION.md      the bar (+ .sha256)
src/micro/
  load.py        zip -> parquet, bookTicker schema, integrity checks
  labels.py      event-time and fixed-horizon labels, drop accounting
  features.py    per-event features + fold-local normalisation
  baselines.py   logit / weighted mid / Stoikov micro-price / LightGBM
  model.py       causal transformer (torch, optional dependency)
  train.py       purged expanding-window walk-forward
  evaluate.py    log-loss, AUC, calibration, bp translation, block bootstrap
  interp.py      linear probes, top-k SAE, activation patching
configs/base.yaml
scripts/
  download_bookticker.py
  build_dataset.py    (load -> labels -> features -> one parquet)
  run_baselines.py
  run_model.py
tests/
```

## Reproduce

```bash
pip install -r requirements.txt
make stamp
make data DAYS=14          # ~1 GB, two weeks of bookTicker
make dataset
make baselines             # the bar. Do this before the model. Always.
make model                 # needs torch: pip install torch --index-url https://download.pytorch.org/whl/cpu
make interp
```

No `make` on this machine? `bash reproduce.sh [stage...]`.

## The thing that makes or breaks this project

Weak baselines. If the GBM is a straw man, the transformer "win" is
uninformative and the whole project is worthless — the article says this and it
is correct. So `run_baselines.py` is designed to be *annoyingly* strong:

- the GBM gets the same information set as the transformer (multi-level book,
  queue deltas, trade flow, recent returns), just flattened;
- the Stoikov micro-price baseline is fitted on the *training fold only*;
- every baseline is evaluated on the same purged folds with the same embargo.

If the transformer loses, that is the result, and the interpretability section
becomes more interesting, not less: "the network learns the micro-price and
nothing else" is a finding with evidence behind it.

## Cost of being wrong about the label

BTC's tick is large relative to price, so the spread is usually one tick and the
mid is nearly discrete. That makes the binary label clean (good) and makes the
edge tiny (the honest problem). The bp translation in `evaluate.py` converts
the log-loss gain into "expected move captured" and compares it with:

```
half_spread_bps + fee_bps
```

If the gain is below that line, no amount of AUC saves the strategy. Report it
plainly, with the number.

## What I'd do next

- Extend to a second symbol with a *smaller* tick relative to price (e.g. a
  mid-cap perp) where the spread is wider than one tick; if the model's edge
  clears costs there but not on BTC, that is a clean, quotable result.
- Cross-venue: Binance bookTicker vs Hyperliquid l2Book on the same clock, to
  test whether the model's features are venue-specific.
- Replace the SAE with a supervised dictionary (label the features by their
  correlation with the micro-price residual) and see whether the unsupervised
  features are worse — a real experiment, not a demo.

## Attribution

Scaffold generated with an AI coding agent; equations are documented above their
implementations so they can be re-derived by hand.

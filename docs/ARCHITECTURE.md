# ARCHITECTURE — Order Book Mind Reader

## 1. System overview

```
data.binance.vision (bookTicker zips)
        |
        v
scripts/download_bookticker.py
        |
        v
micro.load.read_bookticker_zip  ->  to_event_frame (mid, spread, tick, imbalance)
        |
        v
micro.labels  (next non-zero mid change)
micro.features (8 lagged features)     [scripts/build_dataset.py]
        |
        v
data/processed/bookticker_features.parquet
        |
   +----+------------------------------------+
   v                                         v
micro.baselines                         micro.model (torch)
 (logit, weighted mid,                  (causal transformer)
  Stoikov, LightGBM)                          |
   |                                          |
   +------------------+-----------------------+
                      v
        micro.train.purged_walk_forward  (folds + embargo)
                      |
                      v
     micro.evaluate (log-loss, AUC, block bootstrap, bp)
                      |
        +-------------+-------------+
        v                           v
 outputs/baselines.csv       outputs/model_vs_baseline.csv
                                    |
                                    v
                            micro.interp (probes / SAE / patching)
                                    |
                                    v
                            outputs/probes.csv
```

## 2. Layers and responsibilities

| Layer | Module | Does | Must never |
|---|---|---|---|
| Ingest | `load.py` | bytes → event frame, integrity accounting | compute labels |
| Labels | `labels.py` | forward-looking targets | be imported by `features.py` |
| Features | `features.py` | backward-looking inputs | contain a negative shift |
| Models | `baselines.py`, `model.py` | fit/predict | normalise with test statistics |
| Splits | `train.py` | folds, embargo, loop | score models |
| Metrics | `evaluate.py` | scores, CIs, bp | decide what "wins" (config does) |
| Explanation | `interp.py` | probes, SAE, patching | assert a claim without a control |
| Orchestration | `scripts/*` | wire stages | hold formulas |

## 3. Data flow: one fold

1. `purged_walk_forward` yields train/test index arrays with a gap.
2. `Normalizer.fit(train)` then `.transform(both)` — test statistics never leak.
3. Baselines fit on the train block, predict on the test block.
4. The transformer slices context windows ending at each labelled event,
   trains with an inner 85/15 split of the train block for early stopping.
5. Predictions are aligned on the intersection of the two index sets (the GBM
   sees every event; the transformer only events with a full context).
6. `evaluate.block_bootstrap_diff` compares paired predictions on contiguous
   blocks.
7. `evaluate.edge_in_bps` converts direction into bp and compares with the cost
   line.

## 4. Folder structure

```
02-orderbook-mind-reader/
├── docs/             the 12 documents
├── AGENTS.md  README.md  PREREGISTRATION.md (+ .sha256)
├── configs/base.yaml frozen experiment config
├── Makefile  reproduce.sh  reproduce.ps1  requirements.txt
├── data/raw  data/processed
├── src/micro/  load labels features baselines model train evaluate interp config style
├── scripts/    download_bookticker build_dataset run_baselines run_model run_interp stamp_prereg
├── tests/
└── outputs/
```

## 5. Architectural decisions

| Decision | Why | Cost |
|---|---|---|
| Event-time labels | removes the "nothing happened" mass that dominates calendar-time loss | the model never learns to abstain |
| Baselines fit per fold | a global fit leaks future distribution into the past | 5× the fit time; worth it |
| One `ModelConfig` dataclass | the ablation grid is readable and loggable | slightly more plumbing |
| Torch optional | baselines must run on any machine; the bar cannot depend on a GPU | two code paths in `scripts/` |
| Patching via `forward_with_injection` | a swap must be causal, not a correlation | an extra forward method on the model |
| Align predictions on the intersection | comparing models on different rows is the classic silent error | fewer scored events |
| Tick estimated from data | archives omit exchange filters | coarse if tick changed mid-sample |
| Probes must beat two controls | the only defence against "decodable means nothing" | some true-but-weak findings get dropped |

## 6. Extension points

- **New baseline:** add a function with signature `(train, y_train, test) ->
  p` in `baselines.py`, register it in `run_baselines.py`, and add a row to
  `docs/TESTING.md`. Do not change `evaluate.py`.
- **New feature:** add it to `FEATURE_COLUMNS`, update `configs/base.yaml` and
  `docs/BACKEND_SCHEMA.md`, and confirm the lag test still passes.
- **Second symbol:** parameterise `config.SYMBOL` and the archive path; do not
  fork the repo — a fork makes the comparison unverifiable.

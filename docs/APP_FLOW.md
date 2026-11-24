# APP_FLOW — Order Book Mind Reader

No UI; the "screens" are pipeline stages and the "journey" is the required order.

## 1. Stage list

| Stage | Command | Reads | Writes |
|---|---|---|---|
| 0. Register | `make stamp` | `PREREGISTRATION.md` | `PREREGISTRATION.sha256` |
| 1. Acquire | `make data DAYS=7` | network | `data/raw/*.zip` |
| 2. Build | `make dataset` | zips | `data/processed/bookticker_features.parquet` |
| 3. Baselines | `make baselines` | parquet | `outputs/baselines.csv` |
| 4. Model | `make model` | parquet + baselines | `outputs/model_vs_baseline.csv` |
| 5. Interpret | `make interp` | parquet | `outputs/probes.csv`, `sae_examples.npy` |

## 2. Main journey

```
stamp → data → dataset → baselines → model → interp
```

`baselines` must come before `model`. `run_model.py` exits with an error if
`outputs/baselines.csv` is missing — the comparison is the whole product.

## 3. Per-stage detail

### Stage 1 — Acquire
- **Sees:** `coverage: 2023-12-09 .. 2024-03-30`, then one line per day.
- **Actions:** `DAYS=n` or `--months n`; `--end` to move the window earlier.
- **Empty state:** no archives → explicit message naming the script.
- **Failure:** 404 for an unpublished day → `skip`, not a crash.
- **Warning:** ~1.5 GB for two weeks; the script prints sizes as it goes.

### Stage 2 — Build
- **Sees:** event count, date range, estimated tick, median spread in ticks,
  one-sided spread in bp, dropped-label fraction, class balance.
- **Empty state:** `no archives; run scripts/download_bookticker.py first`.
- **Check:** if `dropped_fraction` is very high (> 90%), the sample is mostly
  unchanged books and the label is nearly "the rare event" — the script warns.

### Stage 3 — Baselines
- **Sees:** a per-fold line, then a table of 4 models, then three bootstrap
  comparisons against LightGBM.
- **Reading it:** if `imbalance_logit` and `lightgbm` are indistinguishable, the
  dataset cannot support the project; stop and say so.

### Stage 4 — Model
- **Guard:** exits early if torch is absent, with the install command.
- **Sees:** per-fold scores, then `model vs bar`, then the bar verdict.
- **Budget state:** folds with < 1000 train windows or < 200 test windows are
  skipped with a message rather than silently scored.

### Stage 5 — Interpret
- **Sees:** probe table with `decodable`, SAE feature count, patching recovery.
- **Withdrawal state:** if a probe fails its control, `decodable` is False; if
  patching recovery is < 0.5, the script prints that the feature is stored but
  not necessarily used.

## 4. States

| State | Meaning | Where |
|---|---|---|
| Loading | download / dataset build | stdout progress |
| Empty | no archives, no baselines, no torch | stderr + exit 1 |
| Success | artefact written | one summary line |
| Warning | crossed quotes, high drop rate, weak baseline | stdout, prefixed |
| Withdrawn | control also passed | explicit line in the interp output |

## 5. Edge cases

| Case | Behaviour |
|---|---|
| Crossed or zero quotes | counted, reported as a rate, dropped |
| `bid_qty + ask_qty == 0` | imbalance set to NaN, row excluded from that feature |
| Series shorter than the context | `model.windows` raises with the numbers |
| Fold with too few windows | skipped, reported |
| No torch | model/interp stages exit 1 with the install command |
| Windows event stream (`\r\n`) | pandas handles it; timestamps are epoch ints |
| Archive extends past 2024-04 in future | downloader uses the live range and prints it; the change is a re-registration |

## 6. Access rules

| Artefact | Access |
|---|---|
| `data/raw/` | any stage; read-only |
| `data/processed/` | written by `build_dataset.py` only |
| `outputs/` | written by its own stage only |
| the model checkpoint | written by `run_model.py` into `outputs/`, gitignored |

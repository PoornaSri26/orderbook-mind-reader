# IMPLEMENTATION_PLAN — Order Book Mind Reader

**Objective:** an honest model-vs-baseline comparison plus a mechanistic
explanation, in three weeks part-time.
**Team:** one person. **Approach:** baselines before models, always.

## Phase 0 — Freeze (day 1)

| Task | Deliverable |
|---|---|
| Write the bar (log-loss, AUC, bp disclosure) | `PREREGISTRATION.md` |
| Record the archive's coverage limit | `docs/TRD.md` §6, `data/README.md` |
| Skeleton + Makefile | repo runs `make help` |

**Gate:** the bar is committed before `src/` exists.

## Phase 1 — Ingest (days 2-4)

| Task | Deliverable |
|---|---|
| zip reader with crossed-quote accounting | `load.read_bookticker_zip` |
| event frame with estimated tick | `load.to_event_frame` |
| downloader with live coverage check | `scripts/download_bookticker.py` |
| 7 days downloaded | `data/raw/` |

**Gate:** a day file round-trips to a frame with a monotone index and a printed
crossed-quote rate.

## Phase 2 — Labels and features (days 5-6)

| Task | Deliverable |
|---|---|
| next-non-zero label + drop accounting | `labels.event_time_labels` |
| two robustness labels | `labels.fixed_horizon_*` |
| 8 lagged features | `features.build` |
| leak tests | `tests/test_labels.py` |

**Gate:** the `shift` direction test passes, and the drop fraction is printed by
`build_dataset.py`.

## Phase 3 — Baselines (days 7-9)

| Task | Deliverable |
|---|---|
| logit, weighted mid | `baselines.py` |
| Stoikov micro-price (10 buckets × spread) | `baselines.StoikovMicroprice` |
| LightGBM on the same feature set | `baselines.lightgbm_baseline` |
| fold loop + bootstrap comparison | `outputs/baselines.csv` |

**Gate:** the baseline table exists and the LightGBM-vs-logit difference is
either clearly non-zero or the dataset is declared too noisy. Do not proceed
until this is written down.

## Phase 4 — Transformer (days 10-14)

| Task | Deliverable |
|---|---|
| causal model, 3 layers, d_model 96 | `model.BookTransformer` |
| purged walk-forward with inner val split | `run_model.py` |
| paired comparison | `outputs/model_vs_baseline.csv` |
| bp translation vs cost line | printed verdict |

**Gate:** the bar verdict is printed (`PASS`/`FAIL`) and copied to the README.

## Phase 5 — Interpretability (days 15-19)

| Task | Deliverable |
|---|---|
| ridge probes + two controls | `outputs/probes.csv` |
| top-k SAE + max-activating examples | `sae_examples.npy` + printout |
| causal patching via injection | recovery per layer |
| imbalance-response curve | one figure |

**Gate:** every claim in the README has its control number next to it.

## Phase 6 — Write-up (days 20-21)

| Task | Deliverable |
|---|---|
| README: question, bar, result, bp verdict | top ten lines |
| baseline table and response curve | figures |
| what I'd do next | one page |
| interview prep re-derived by hand | personal, not committed |

## Risks

| Risk | Mitigation |
|---|---|
| Weak baseline invalidates the result | GBM gets the same information set; print the comparison first |
| Archive ends 2024-03 | state it; do not imply currency; offer the Hyperliquid fallback |
| 40 GB download surprise | start with 7 days, size printed per file |
| Interpretability theatre | controls are computed in the same script that makes the claim |
| Silence about costs | `evaluate.verdict` cannot return without a bp number |

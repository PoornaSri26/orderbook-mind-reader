# AGENTS.md — Order Book Mind Reader

## Project

Predict the next BTCUSDT mid move with a small causal transformer and prove —
or refute — that it beats strong microstructure baselines. Python 3.11+,
numpy/pandas/sklearn/lightgbm, optional torch. No server, no deployment.

## Commands

```bash
pip install -r requirements.txt
make stamp                     # freeze + hash PREREGISTRATION.md
make data DAYS=7               # ~1.5 GB of bookTicker (archive ends 2024-03-30)
make dataset                   # load -> labels -> features
make baselines                 # the bar. Must run before the model.
make model                     # needs: pip install torch --index-url .../cpu
make interp                    # probes, SAE, patching
make test
```

Windows without `make`: `bash reproduce.sh` or `pwsh reproduce.ps1`.
Entry points need `PYTHONPATH=src`.

## Documentation map

| Question | File |
|---|---|
| Scope, bar, priorities | `docs/PRD.md` |
| Stack, endpoints, coverage limits | `docs/TRD.md` |
| Stage order and guards | `docs/APP_FLOW.md` |
| Module boundaries and data flow | `docs/ARCHITECTURE.md` |
| Parquet/CSV schemas | `docs/BACKEND_SCHEMA.md` |
| Figure and table rules | `docs/DESIGN_SYSTEM.md` |
| Naming, lag comments, commits | `docs/CODE_STYLE.md` |
| Three-week phase plan | `docs/IMPLEMENTATION_PLAN.md` |
| Leakage controls, provider terms | `docs/SECURITY.md` |
| Test cases and statuses | `docs/TESTING.md` |
| **The frozen pass/fail bar** | `PREREGISTRATION.md` |

## Rules — Always

1. Read `docs/` first, starting with `PRD.md` and `ARCHITECTURE.md`.
2. **Baselines before the model.** Never report a transformer number without
   `outputs/baselines.csv` present.
3. Any prediction gain is reported with its bp value and the cost line.
4. `features.py` must not import `labels.py`. Keep that direction.
5. Fold-local normalisation only. Never fit a scaler on the full series.
6. `make test` passes before you say done.
7. Update `docs/BACKEND_SCHEMA.md` with any column change, same commit.

## Rules — Ask first

- Changing the bar, the embargo, `FEE_BPS`, or the label definition.
- Adding a dependency (especially anything that pulls a GPU wheel by default).
- Changing `FEATURE_COLUMNS` (it invalidates every stored result).
- Downloading more than two weeks of archives (disk).
- Editing `PREREGISTRATION.md` (needs a `RE-REGISTER` commit).

## Rules — Never

- Introduce a negative shift outside `labels.py`.
- Report AUC, log-loss or a "win" without the bp translation.
- Describe a probe as "the model uses X" — that requires patching recovery.
- Claim the data is recent: the archive ends 2024-03-30.
- Commit `data/`, `outputs/`, torch checkpoints, or `.env`.
- `torch.load` an untrusted file.
- Reuse the 82% / 59% figures from the source article as benchmarks.

## Definition of done

Tests pass, the baseline table exists, the bar verdict is printed and copied
verbatim into the README, and every interpretability claim carries its control
number. Generating a model is not the deliverable; a comparison is.

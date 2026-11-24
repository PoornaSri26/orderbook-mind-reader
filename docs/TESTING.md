# TESTING — Order Book Mind Reader

## 1. Overview

| Type | Checks | Tool |
|---|---|---|
| Unit | labels, features, metrics against closed forms | pytest |
| Leak test | no feature uses the future | pytest |
| Split test | embargo is respected | pytest |
| Integration | each stage writes the artefact the next reads | pytest + `tmp_path` |
| Manual | figure legibility, patching plausibility | human |

## 2. Goals

1. A look-ahead bug fails a test rather than raising the score.
2. The metrics agree with hand-computable cases (ln 2 for a coin flip).
3. The baseline gate cannot be bypassed.
4. Interpretability output always carries its control columns.

## 3. Test cases

| ID | Scenario | Expected | Type | Status |
|---|---|---|---|---|
| TC-01 | Next-non-zero label on `[100,100,101,101,100]` | +1,+1,−1,−1 | unit | Pass |
| TC-02 | Last event | label is NaN | unit | Pass |
| TC-03 | Flat series `[100,100,100,101]` | dropped fraction 0.5 | unit | Pass |
| TC-04 | Purged split, embargo 356 | train.max < test.min − 300 | unit | Pass |
| TC-05 | `embargo_for(256, 100)` | 356 | unit | Pass |
| TC-06 | Feature with constant mid | `mid_change == 0` everywhere | leak | Pass |
| TC-07 | First row of features | lagged columns are 0 | leak | Pass |
| TC-08 | Perfect predictor | log-loss < 1e-5 (clip floor) | unit | Pass |
| TC-09 | Coin flip | log-loss = ln 2 | unit | Pass |
| TC-10 | Random predictions | AUC in (0.47, 0.53) | unit | Pass |
| TC-11 | Clear improvement | bootstrap diff < 0, CI excludes 0 | unit | Pass |
| TC-12 | `p == 0.5` everywhere | 0 trades, edge 0 | unit | Pass |
| TC-13 | Correct side, 2 bp moves | edge = 2 bp, hit rate 1 | unit | Pass |
| TC-14 | `verdict(0.05, 0.3)` | says "not tradeable" | unit | Pass |
| TC-15 | `run_model.py` without `baselines.csv` | exit 1 with a message | integration | **Pending** |
| TC-16 | Crossed quotes in a synthetic zip | counted and dropped | unit | **Pending** |
| TC-17 | Probe with shuffled labels only | `decodable` is False | unit | **Pending** |
| TC-18 | Patching with identical clean/corrupt input | recovery is NaN, not 0 | unit | **Pending** |
| TC-19 | Two runs, same seed | identical fold log-loss | reproduc. | **Pending** |

## 4. Running

```bash
make test
PYTHONPATH=src python -m pytest -q tests -k labels
bash reproduce.sh dataset baselines      # stage-level integration
```

## 5. Checklist

- [ ] Tests need no network and no GPU.
- [ ] Tests never write to `data/` or `outputs/`.
- [ ] Every metric has at least one case with a closed-form answer.
- [ ] The leak tests would fail if a feature were shifted the wrong way
      (verify by introducing `shift(-1)` on a branch).
- [ ] The bar gate is tested by running the model stage without baselines.

## 6. What is not tested

- **Model quality.** A test cannot tell you whether 0.004 nats matters; only
  the bootstrap CI and the bp translation can.
- **Stoikov micro-price correctness in the limit.** The Markov fixed point is
  implemented as a one-step expectation because that is what the predictor
  needs; a full iterative solution is listed as follow-up, not claimed.
- **Figures.** Reviewed by eye against `docs/DESIGN_SYSTEM.md`.
- **`.CHECKSUM` verification** is exercised only on a real download.

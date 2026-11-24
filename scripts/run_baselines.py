#!/usr/bin/env python
"""Run the baseline table: every cheap detector, walk-forward, scored.

Writes outputs/baselines.csv — the bar the transformer must beat
(run_model.py refuses to run without it).
"""

from __future__ import annotations

import sys

import numpy as np
import pandas as pd

from micro import baselines, config, evaluate, features, train


def main() -> int:
    config.ensure_dirs()

    from micro import load

    df = load.load_parquet("bookticker_features")
    y_all = df["label_event"].to_numpy(dtype=float)
    x_all = df[features.FEATURE_COLUMNS]
    folds = train.purged_walk_forward(
        len(df), n_folds=config.N_FOLDS, min_train=config.MIN_TRAIN_FRAC, embargo=config.EMBARGO
    )

    rows = []
    paired = {"y": [], "imbalance_logit": [], "logistic_features": [], "lightgbm": []}

    for f in folds:
        print(f"  {f.label}")
        tr, te = f.train, f.test
        ytr_raw = y_all[tr]
        ytr = y_all[tr]
        valid_tr = ~np.isnan(ytr_raw)
        valid_te = ~np.isnan(y_all[te])
        if valid_tr.sum() < 1000 or valid_te.sum() < 200:
            print("    skip: not enough labelled rows in this fold")
            continue

        tr_idx = tr[valid_tr]
        te_idx = te[valid_te]
        ytr_v = y_all[tr_idx]
        yte_v = y_all[te_idx]

        preds = {}

        # 1. imbalance-only logistic
        try:
            preds["imbalance_logit"] = baselines.imbalance_logit(
                df["imbalance"].iloc[tr_idx], ytr_v, df["imbalance"].iloc[te_idx]
            )
        except Exception as e:
            print(f"    imbalance_logit failed: {e}")

        # 2. logistic on the full feature set (cheap linear bar)
        try:
            preds["logistic_features"] = baselines.logit_fit_predict(
                x_all.iloc[tr_idx], ytr_v, x_all.iloc[te_idx]
            )
        except Exception as e:
            print(f"    logistic_features failed: {e}")

        # 3. LightGBM on the same features the transformer sees
        try:
            preds["lightgbm"] = baselines.lightgbm_baseline(
                x_all.iloc[tr_idx], ytr_v, x_all.iloc[te_idx]
            )
        except Exception as e:
            print(f"    lightgbm failed: {e}")

        # 4. Stoikov micro-price, if the needed columns exist
        for col in ("imbalance", "spread_ticks", "mid_change_ticks"):
            if col not in df.columns:
                break
        else:
            try:
                mp = baselines.StoikovMicroprice().fit(
                    df["imbalance"].iloc[tr_idx],
                    df["spread_ticks"].iloc[tr_idx],
                    df["mid_change_ticks"].iloc[tr_idx],
                )
                preds["stoikov_microprice"] = mp.predict_proba(
                    df["imbalance"].iloc[te_idx], df["spread_ticks"].iloc[te_idx]
                )
            except Exception as e:
                print(f"    stoikov_microprice failed: {e}")

        for name, p in preds.items():
            rows.append({**evaluate.score(name, yte_v, p).as_row(), "fold": f.label})
            paired[name].append(p)
        paired["y"].append(yte_v)

    if not rows:
        print("no baseline rows produced — check the features parquet", file=sys.stderr)
        return 1

    table = pd.DataFrame(rows)
    table.to_csv(config.OUT / "baselines.csv", index=False)

    pooled_y = np.concatenate(paired["y"])
    print("\npooled out-of-fold scores:")
    for name in ("imbalance_logit", "logistic_features", "lightgbm", "stoikov_microprice"):
        if name in paired and paired[name]:
            p = np.concatenate(paired[name])
            s = evaluate.score(name, pooled_y, p)
            print(f"  {name:22s} logloss={s.log_loss:.5f} auc={s.auc:.4f} brier={s.brier:.5f} n={s.n}")

    print(f"\nwrote {config.OUT / 'baselines.csv'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

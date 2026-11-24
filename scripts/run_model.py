#!/usr/bin/env python
"""Train the causal transformer on every fold and compare it to LightGBM.

Refuses to run if the baseline table is missing: the whole point is comparison,
and a transformer number without a bar is not a result.
"""

from __future__ import annotations

import sys

import numpy as np
import pandas as pd

from micro import baselines, config, evaluate, features, model, train


def main() -> int:
    config.ensure_dirs()
    if not model.available():
        print(
            "torch is not installed. This is a hard requirement for the model stage:\n"
            "  pip install torch --index-url https://download.pytorch.org/whl/cpu",
            file=sys.stderr,
        )
        return 1
    if not (config.OUT / "baselines.csv").exists():
        print("run scripts/run_baselines.py first: no bar to beat", file=sys.stderr)
        return 1

    from micro import load

    df = load.load_parquet("bookticker_features")
    y_all = df["label_event"]
    x = df[features.FEATURE_COLUMNS]
    folds = train.purged_walk_forward(
        len(df), n_folds=config.N_FOLDS, min_train=config.MIN_TRAIN_FRAC, embargo=config.EMBARGO
    )

    import torch

    rows, paired = [], {"y": [], "gbm": [], "transformer": []}
    for f in folds:
        print(f"  {f.label}")
        norm = features.Normalizer.fit(x.iloc[f.train])
        xtr_all = norm.transform(x).to_numpy(dtype=np.float64)
        y_raw = y_all.to_numpy(dtype=float)

        xtr, ytr, idx_tr = model.windows(xtr_all, y_raw, config.CONTEXT)
        xte, yte, idx_te = model.windows(xtr_all, y_raw, config.CONTEXT)
        m_tr = np.isin(idx_tr, f.train)
        m_te = np.isin(idx_te, f.test)
        if m_tr.sum() < 1000 or m_te.sum() < 200:
            print("    skip: not enough windows in this fold")
            continue

        # inner validation split for early stopping, taken from the train block only
        cut = int(m_tr.sum() * 0.85)
        tr_idx = np.flatnonzero(m_tr)[:cut]
        va_idx = np.flatnonzero(m_tr)[cut:]
        te_idx = np.flatnonzero(m_te)

        cfg = model.ModelConfig(n_features=len(features.FEATURE_COLUMNS), context=config.CONTEXT)
        net = model.BookTransformer(cfg)
        net, hist = train.fit_torch(
            net,
            xtr[tr_idx],
            ytr[tr_idx],
            xtr[va_idx],
            ytr[va_idx],
            epochs=4,
            batch_size=512,
            lr=3e-4,
            seed=config.SEED,
        )

        net.eval()
        with torch.no_grad():
            logits = net(torch.from_numpy(xtr[te_idx])).numpy()
        p_t = 1.0 / (1.0 + np.exp(-logits))

        xgb_tr = x.iloc[f.train].to_numpy()
        xgb_te = x.iloc[f.test].to_numpy()
        keep_tr = ~np.isnan(y_raw[f.train])
        keep_te = ~np.isnan(y_raw[f.test])
        p_g = baselines.lightgbm_baseline(
            pd.DataFrame(xgb_tr[keep_tr]), (y_raw[f.train][keep_tr] > 0).astype(int),
            pd.DataFrame(xgb_te[keep_te]),
        )

        # align: the GBM sees every event, the transformer every context window
        common = np.intersect1d(idx_te[te_idx], f.test[keep_te])
        if common.size < 200:
            continue
        pos_t = {v: k for k, v in enumerate(idx_te[te_idx])}
        pos_g = {v: k for k, v in enumerate(f.test[keep_te])}
        yt = (y_raw[common] > 0).astype(int)
        pt = np.array([p_t[pos_t[c]] for c in common])
        pg = np.array([p_g[pos_g[c]] for c in common])

        rows.append({**evaluate.score("transformer", yt, pt).as_row(), "fold": f.label})
        rows.append({**evaluate.score("lightgbm", yt, pg).as_row(), "fold": f.label})
        paired["y"].append(yt)
        paired["gbm"].append(pg)
        paired["transformer"].append(pt)
        hist.to_csv(config.OUT / f"history_{f.label.split(':')[0]}.csv", index=False)

    if not rows:
        print("no fold produced a comparable prediction", file=sys.stderr)
        return 1

    table = pd.DataFrame(rows).groupby("model", as_index=False).agg(
        log_loss=("log_loss", "mean"), auc=("auc", "mean"), n=("n", "sum")
    )
    print("\nmodel vs bar:")
    print(table.to_string(index=False))

    y = np.concatenate(paired["y"])
    d = evaluate.block_bootstrap_diff(y, np.concatenate(paired["gbm"]), np.concatenate(paired["transformer"]))
    gain = -d["observed_diff"]
    print(f"\nlog-loss(lightgbm) - log-loss(transformer) = {gain:+.5f} "
          f"[{-d['ci_hi']:+.5f}, {-d['ci_lo']:+.5f}]")
    print(f"bar: {config.LOG_LOSS_IMPROVEMENT_BAR} nats with CI excluding zero -> "
          f"{'PASS' if (gain >= config.LOG_LOSS_IMPROVEMENT_BAR and d['excludes_zero']) else 'FAIL'}")

    table.to_csv(config.OUT / "model_vs_baseline.csv", index=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

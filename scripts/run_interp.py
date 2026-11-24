#!/usr/bin/env python
"""Probes with controls, top-k SAE, and activation patching.

Every claim is printed next to its control. If the control also scores well, the
claim is explicitly withdrawn in the output — the script does that itself so it
cannot be forgotten in the write-up.
"""

from __future__ import annotations

import sys

import numpy as np
import pandas as pd

from micro import config, features, interp, load, model


def main() -> int:
    config.ensure_dirs()
    if not model.available():
        print("torch required for the interpretability stage", file=sys.stderr)
        return 1
    try:
        df = load.load_parquet("bookticker_features").tail(60_000)
    except FileNotFoundError:
        print("run scripts/build_dataset.py first", file=sys.stderr)
        return 1

    import torch

    x = features.Normalizer.fit(df[features.FEATURE_COLUMNS]).transform(df[features.FEATURE_COLUMNS]).to_numpy()
    ctx = config.CONTEXT
    cfg = model.ModelConfig(n_features=len(features.FEATURE_COLUMNS), context=ctx)
    net = model.BookTransformer(cfg).eval()
    rnd = model.BookTransformer(cfg).eval()  # random init = the required control

    xw = np.stack([x[i - ctx + 1 : i + 1] for i in range(ctx - 1, len(x), 25)]).astype(np.float32)
    split = int(len(xw) * 0.7)
    xt = torch.from_numpy(xw)
    acts = net.residuals_at(xt, cfg.mid_layer).numpy()
    acts_r = rnd.residuals_at(xt, cfg.mid_layer).numpy()

    book = {
        "imbalance": xw[:, -1, features.FEATURE_COLUMNS.index("imbalance")],
        "spread_ticks": xw[:, -1, features.FEATURE_COLUMNS.index("spread_ticks")],
        "bid_price": np.zeros(len(xw)),
        "ask_price": np.zeros(len(xw)),
        "bid_qty": np.expm1(xw[:, -1, features.FEATURE_COLUMNS.index("log_bid_qty")]),
        "ask_qty": np.expm1(xw[:, -1, features.FEATURE_COLUMNS.index("log_ask_qty")]),
        "tick": np.ones(len(xw)),
        "signed_trade_qty": xw[:, -1, features.FEATURE_COLUMNS.index("signed_qty_change")],
    }
    targets = interp.make_probe_features(book)

    rows = []
    for name, z in targets.items():
        if not np.isfinite(z).all():
            continue
        r = interp.linear_probe(
            acts[:split], z[:split], acts[split:], z[split:], target=name, acts_random=acts_r[:split]
        )
        rows.append(
            {
                "target": r.target,
                "r2_model": r.r2_model,
                "r2_random": r.r2_random_model,
                "r2_shuffled": r.r2_shuffled_labels,
                "decodable": r.decodable,
            }
        )
    table = pd.DataFrame(rows)
    print("\nlinear probes (mid layer):")
    print(table.to_string(index=False))
    table.to_csv(config.OUT / "probes.csv", index=False)

    # --- SAE -------------------------------------------------------------
    sae = interp.TopKSAE(d_in=acts.shape[1], d_hidden=512, k=32, seed=config.SEED).fit(
        acts[:split], epochs=10
    )
    examples = sae.max_activating_examples(acts[split:], top=5)
    print(f"\nSAE: {len(examples)} active features out of 512; top-5 examples stored per feature")
    np.save(config.OUT / "sae_examples.npy", np.array(list(examples.keys())))

    # --- patching --------------------------------------------------------
    feat_idx = {k: i for i, k in enumerate(features.FEATURE_COLUMNS)}
    clean = xt[-256:]
    corrupt = torch.from_numpy(interp.corrupt_imbalance(clean.numpy(), feat_idx, level=1))
    pat = interp.patch_activation(net, clean, corrupt, layer=cfg.mid_layer)
    print("\nactivation patching (imbalance flipped):")
    for k, v in pat.items():
        print(f"  {k:16s} {v:+.4f}")
    if abs(pat["recovery"]) < 0.5:
        print("  -> model does NOT rely on this layer's imbalance code at the last token:")
        print("     the feature is stored, not necessarily used. Report as such.")

    print("\nreminder: any claim whose control also passes is withdrawn, not reworded.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

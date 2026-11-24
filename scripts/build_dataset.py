#!/usr/bin/env python
"""load -> labels -> features -> one parquet. The only place raw data is read."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

from micro import config, features, labels, load


def main() -> int:
    config.ensure_dirs()
    zips = sorted(config.RAW.glob("BTCUSDT-bookTicker-*.zip"))
    if not zips:
        print("no archives; run scripts/download_bookticker.py first", file=sys.stderr)
        return 1

    frames = []
    for z in zips:
        print(f"reading {z.name}")
        frames.append(load.read_bookticker_zip(z))
    df = pd.concat(frames).sort_index()
    df = df[~df.index.duplicated(keep="last")]
    print(f"{len(df):,} events  {df.index.min()} -> {df.index.max()}")

    ev = load.to_event_frame(df)
    tick = float(ev["tick"].iloc[0])
    print(f"estimated tick {tick:g}; median spread {ev['spread_ticks'].median():.2f} ticks")
    print(f"one-sided spread in bp: {(0.5 * ev['spread'] / ev['mid']).median() * 1e4:.4f}")

    ev["label_event"] = labels.event_time_labels(ev["mid"]).y
    ev["label_fixed100"] = labels.fixed_horizon_labels(ev["mid"], 100).y
    print(f"dropped (no future change): {labels.event_time_labels(ev['mid']).dropped_fraction:.2%}")
    print("class balance:", labels.class_balance(ev["label_event"]))

    feats = features.build(ev, tick=tick)
    out = pd.concat([feats, ev[["mid", "spread", "spread_ticks", "imbalance", "label_event", "label_fixed100"]]], axis=1)
    path = load.save_parquet(out, "bookticker_features")
    print(f"wrote {path}  shape={out.shape}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

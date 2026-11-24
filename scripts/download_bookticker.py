#!/usr/bin/env python
"""Download Binance USD-M BTCUSDT bookTicker archives.

Coverage warning, verified Nov 2026: the archive stops in early 2024
(daily through 2024-03-30, monthly through 2024-04). This script asks the
bucket for its actual range and refuses dates outside it instead of producing
silent 404s.

Usage:
    python scripts/download_bookticker.py --days 7 --end 2024-03-30
    python scripts/download_bookticker.py --months 2 --end 2024-03

Sizes: 80-230 MB per *day*, 3-8 GB per month. Two weeks is ~1.5 GB.
"""

from __future__ import annotations

import argparse
import hashlib
import sys
import urllib.error
import urllib.request
from datetime import date, timedelta
from pathlib import Path

BASE = "https://data.binance.vision/data/futures/um"
RAW = Path(__file__).resolve().parents[1] / "data" / "raw"
SYMBOL = "BTCUSDT"


def fetch(url: str, timeout: int = 300) -> bytes:
    with urllib.request.urlopen(url, timeout=timeout) as r:
        return r.read()


def verify(payload: bytes, checksum_url: str, name: str) -> None:
    expected = fetch(checksum_url).decode().split()[0].strip()
    got = hashlib.sha256(payload).hexdigest()
    if expected != got:
        raise RuntimeError(f"checksum mismatch for {name}")


def check_available(granularity: str) -> list[str]:
    import re

    url = f"{BASE}/{granularity}/bookTicker/{SYMBOL}/"
    try:
        html = fetch(url, timeout=60).decode("utf-8", "replace")
    except urllib.error.HTTPError as exc:
        raise SystemExit(f"cannot list {url}: {exc}") from exc
    if granularity == "daily":
        pat = rf"{SYMBOL}-bookTicker-(\d{{4}}-\d{{2}}-\d{{2}})\.zip"
    else:
        pat = rf"{SYMBOL}-bookTicker-(\d{{4}}-\d{{2}})\.zip"
    found = sorted(set(re.findall(pat, html)))
    return found


def day_range(days: int, end: date) -> list[str]:
    return [(end - timedelta(days=i)).isoformat() for i in range(days - 1, -1, -1)]


def download_daily(days: int, end: date) -> int:
    available = check_available("daily")
    if not available:
        raise SystemExit("no daily files listed")
    lo, hi = available[0], available[-1]
    print(f"coverage: {lo} .. {hi}")
    wanted = day_range(days, min(end, date.fromisoformat(hi)))
    ok = 0
    for d in wanted:
        if d not in available:
            print(f"  skip {d} (not published)")
            continue
        name = f"{SYMBOL}-bookTicker-{d}.zip"
        dest = RAW / name
        if dest.exists():
            print(f"  have {d}")
            ok += 1
            continue
        url = f"{BASE}/daily/bookTicker/{SYMBOL}/{name}"
        try:
            payload = fetch(url)
            verify(payload, f"{url}.CHECKSUM", name)
            dest.write_bytes(payload)
            print(f"  ok   {d}  {len(payload/1e6):.0f} MB" if False else f"  ok   {d}  {len(payload)/1e6:.0f} MB")
            ok += 1
        except Exception as exc:  # noqa: BLE001
            print(f"  FAIL {d}: {exc}", file=sys.stderr)
    return ok


def download_monthly(months: int, end: str) -> int:
    available = check_available("monthly")
    print(f"coverage: {available[0]} .. {available[-1]}")
    wanted = available[-months:] if months else available
    ok = 0
    for m in wanted:
        name = f"{SYMBOL}-bookTicker-{m}.zip"
        dest = RAW / name
        if dest.exists():
            print(f"  have {m}")
            ok += 1
            continue
        url = f"{BASE}/monthly/bookTicker/{SYMBOL}/{name}"
        try:
            payload = fetch(url, timeout=1800)
            verify(payload, f"{url}.CHECKSUM", name)
            dest.write_bytes(payload)
            print(f"  ok   {m}  {len(payload)/1e9:.2f} GB")
            ok += 1
        except Exception as exc:  # noqa: BLE001
            print(f"  FAIL {m}: {exc}", file=sys.stderr)
    return ok


def main() -> int:
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group()
    g.add_argument("--days", type=int, default=7)
    g.add_argument("--months", type=int, default=0)
    ap.add_argument("--end", default="2024-03-30", help="YYYY-MM-DD or YYYY-MM")
    args = ap.parse_args()

    RAW.mkdir(parents=True, exist_ok=True)
    if args.months:
        n = download_monthly(args.months, args.end)
    else:
        n = download_daily(args.days, date.fromisoformat(args.end))
    print(f"\n{n} archive(s) in {RAW}")
    return 0 if n else 1


if __name__ == "__main__":
    raise SystemExit(main())
